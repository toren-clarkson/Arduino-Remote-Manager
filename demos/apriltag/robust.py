"""Color -> AprilTag approach with confirmed detections and bounded loss handling.

Run --camera-only --tag-only first; --video clip.avi always disables the Arduino.
See DESIGN.md for research, timing limits, recording and tuning.
"""

import argparse
import asyncio
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import phonesense

from robotics_lab.april_navigation import Motion, Navigator, Observation, Settings
from robotics_lab.motor_control import SmoothMotorController
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


# /////////////////////// Edit HERE
ARDUINO_IP = "10.99.223.68"
PHONE_IP = None  # PC interface shared with the phone, e.g. "192.168.137.1"
TARGET_TAG_ID = 0  # None locks onto the first tag confirmed over several frames
DETECTOR_BACKEND = "apriltag3"  # "opencv" also works with the existing OpenCV install
TAG_QUAD_DECIMATE = 1.0
TAG_QUAD_SIGMA = 0.0
TAG_THREADS = 2
TAG_DECODE_SHARPENING = 0.25
TAG_MIN_DECISION_MARGIN = 20.0  # starting point to tune with real footage, not a probability
TAG_MAX_HAMMING = 0
TAG_STOP_FRACTION = 0.24  # mean tag side length / image width; NOT a distance in meters
COLOR_STOP_FRACTION = 0.125  # enclosing-circle radius / image width
MIN_COLOR_AREA = 100
MIN_PWM = 150
PWM_ACCELERATION = 160
PWM_DECELERATION = 240
CONTROL_INTERVAL = 0.05
CAMERA_TIMEOUT = 0.6
PERCEPTION_TIMEOUT = 0.6
STAGE_PAUSE_SECONDS = 0.5
RECORD_FPS = 20.0
NAVIGATION = Settings(
    acquire_frames=3, acquire_seconds=0.12,
    grace_seconds=0.30, search_after=0.65, memory_seconds=4.0,
    base_speed=0.22, steering_gain=0.35, max_turn=0.22,
    search_speed=0.18, search_turn=1.1, search_pause=0.8,
)
# /////////////////////// Edit HERE END


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int


LEFT_MOTOR = MotorPins(7, 4, 5)
RIGHT_MOTOR = MotorPins(9, 8, 6)


@dataclass
class VisionResult:
    frame: np.ndarray
    observation: Observation | None
    corners: np.ndarray | None
    sharpness: float
    processing_ms: float
    seen_ids: str = ""
    margin: float | None = None
    hamming: int | None = None


class Perception:
    """Owned by one worker at a time. No motor commands or navigation decisions."""

    def __init__(self, backend):
        self.backend = backend
        if backend == "apriltag3":
            try:
                from pupil_apriltags import Detector
            except ImportError as exc:
                raise RuntimeError("Install pupil-apriltags or use --backend opencv") from exc
            self.detector = Detector(
                families="tag36h11", nthreads=TAG_THREADS,
                quad_decimate=TAG_QUAD_DECIMATE, quad_sigma=TAG_QUAD_SIGMA,
                refine_edges=1, decode_sharpening=TAG_DECODE_SHARPENING,
            )
        else:
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            parameters.aprilTagQuadDecimate = TAG_QUAD_DECIMATE
            parameters.aprilTagQuadSigma = TAG_QUAD_SIGMA
            self.detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), parameters,
            )

    def process(self, jpeg, kind, hsv, tag_id):
        started = time.monotonic()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid camera JPEG")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        observation = corners = None
        seen_ids, margin, hamming = "", None, None
        height, width = gray.shape
        if kind == "color" and hsv is not None:
            image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue, saturation, value = hsv
            # Circular hue distance handles red at the hue boundary.
            delta = np.abs(image_hsv[:, :, 0].astype(np.int16) - hue)
            mask = ((np.minimum(delta, 180 - delta) <= 10)
                    & (np.abs(image_hsv[:, :, 1].astype(np.int16) - saturation) <= 80)
                    & (np.abs(image_hsv[:, :, 2].astype(np.int16) - value) <= 80)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) >= MIN_COLOR_AREA:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    observation = Observation("color", 2*x/width-1, 2*y/height-1, radius/width)
                    corners = cv2.boxPoints(cv2.minAreaRect(contour))
        elif kind == "tag":
            candidates = []
            if self.backend == "apriltag3":
                tags = self.detector.detect(np.ascontiguousarray(gray))
                seen_ids = ";".join(str(tag.tag_id) for tag in tags)
                for tag in tags:
                    if tag_id is not None and tag.tag_id != tag_id:
                        continue
                    # Log rejected quality too, so a threshold can be tuned from evidence.
                    margin, hamming = float(tag.decision_margin), int(tag.hamming)
                    if margin >= TAG_MIN_DECISION_MARGIN and hamming <= TAG_MAX_HAMMING:
                        candidates.append((int(tag.tag_id), tag.corners.copy(), margin, hamming))
            else:
                all_corners, ids, _ = self.detector.detectMarkers(gray)
                if ids is not None:
                    seen_ids = ";".join(str(int(i)) for i in ids.flatten())
                    candidates = [(int(i), c.reshape(4, 2), None, None)
                                  for i, c in zip(ids.flatten(), all_corners)
                                  if tag_id is None or int(i) == tag_id]
            if candidates:
                best_id, corners, margin, hamming = max(
                    candidates, key=lambda item: abs(cv2.contourArea(item[1].astype(np.float32))),
                )
                x, y = corners.mean(axis=0)
                side = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).mean()
                observation = Observation(f"tag:{best_id}", float(2*x/width-1),
                                          float(2*y/height-1), float(side/width))
        return VisionResult(frame, observation, corners, sharpness,
                            1000*(time.monotonic()-started), seen_ids, margin, hamming)


def make_navigator(kind, tag_id):
    expected = "color" if kind == "color" else (f"tag:{tag_id}" if tag_id is not None else None)
    return Navigator(COLOR_STOP_FRACTION if kind == "color" else TAG_STOP_FRACTION,
                     expected_key=expected, settings=NAVIGATION)


async def main(args):
    perception = Perception(args.backend)  # validate dependency before connecting to motors
    board = control = cam = capture = recorder = csv_file = task = None
    connected = False
    try:
        if args.video:
            capture = cv2.VideoCapture(str(args.video))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot read video: {args.video}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else RECORD_FPS
        else:
            cam = phonesense.start()
            if PHONE_IP:
                cam.select_ip(PHONE_IP)
            print("Phone camera URLs:", *cam.phone_urls, sep="\n  ")
            print("On the phone: Connect, Start camera, and allow camera access.")
        if not (args.camera_only or args.video):
            print(f"Connecting to Arduino at {ARDUINO_IP}:31336...", flush=True)
            board = TelemetrixUnoR4WiFiAio(
                transport_address=ARDUINO_IP, autostart=False,
                loop=asyncio.get_running_loop(), close_loop_on_shutdown=False,
                shutdown_on_exception=False, hard_reset_on_shutdown=False,
            )
            await asyncio.wait_for(board.start_aio(), timeout=15)
            connected = True
            control = SmoothMotorController(
                board, (LEFT_MOTOR, RIGHT_MOTOR), min_pwm=MIN_PWM,
                acceleration=PWM_ACCELERATION, deceleration=PWM_DECELERATION,
            )
            await board.disable_all_reporting()
            for motor in (LEFT_MOTOR, RIGHT_MOTOR):
                await board.set_pin_mode_digital_output(motor.in1)
                await board.set_pin_mode_digital_output(motor.in2)
                await board.set_pin_mode_analog_output(motor.enable)
            await control.stop()

        fields = ["receive_time", "result_time", "receive_to_result_ms", "processing_ms",
                  "sharpness", "mission", "phase", "accepted_packet", "detected_key",
                  "seen_ids", "decision_margin", "hamming", "raw_x", "raw_size",
                  "filtered_x", "last_detection_age", "requested_left", "requested_right"]
        if args.log_csv:
            csv_file = Path(args.log_csv).open("x", newline="", encoding="utf-8")
            logger = csv.DictWriter(csv_file, fieldnames=fields)
            logger.writeheader()
        display = np.zeros((480, 640, 3), np.uint8)
        selection = {"hsv": None}
        previous_hsv = None
        window = "Robust AprilTag robot"

        def click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= y < display.shape[0] and 0 <= x < display.shape[1]:
                pixel = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)[y, x]
                selection["hsv"] = tuple(int(value) for value in pixel)

        if not args.headless:
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, click)
        mission = "tag" if args.tag_only else "wait"
        navigator = make_navigator(mission, args.tag_id)
        revision = 0
        motion = Motion("WAIT")
        current = submitted = None
        received_at = vision_at = -math.inf
        task_frame_at = task_revision = task_kind = None
        next_read = last_control = 0.0
        pause_until = None
        eof = False
        result = None
        print("Preview/replay only." if args.camera_only or args.video else "Motor control enabled.")
        print("Click a color to start color -> tag, or use --tag-only. Esc/Q quits.")

        while True:
            now = time.monotonic()
            if capture is not None and not eof and now >= next_read:
                ok, frame = capture.read()
                next_read = now + 1/fps
                if not ok:
                    eof = True
                else:
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if ok:
                        current, received_at = encoded.tobytes(), now
            elif cam is not None:
                jpeg = cam.jpeg
                if jpeg is not None and jpeg is not current:
                    current, received_at = jpeg, now

            if selection["hsv"] != previous_hsv:
                previous_hsv = selection["hsv"]
                mission, revision, pause_until = "color", revision + 1, None
                navigator = make_navigator(mission, args.tag_id)
                vision_at = -math.inf

            if task is not None and task.done():
                result = task.result()  # inference errors trigger the cleanup motor stop
                task = None
                accepted = False
                display = result.frame
                if task_revision == revision:
                    if mission in ("color", "tag"):
                        accepted = navigator.ingest(result.observation, task_frame_at, now)
                    else:
                        accepted = now - task_frame_at <= NAVIGATION.max_result_age
                    if accepted:
                        vision_at = task_frame_at
                if args.record:
                    if recorder is None:
                        if Path(args.record).exists():
                            raise FileExistsError(args.record)
                        recorder = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"MJPG"),
                                                   RECORD_FPS, (result.frame.shape[1], result.frame.shape[0]))
                        if not recorder.isOpened():
                            raise RuntimeError("Cannot create recording; use an .avi filename")
                    recorder.write(result.frame)
                if csv_file:
                    obs = result.observation
                    logger.writerow(dict(zip(fields, [
                        task_frame_at, now, (now-task_frame_at)*1000, result.processing_ms,
                        result.sharpness, task_kind, motion.phase, accepted, obs.key if obs else "",
                        result.seen_ids, result.margin, result.hamming, obs.x if obs else "",
                        obs.size if obs else "", navigator.x, now-navigator.last_seen,
                        motion.left, motion.right,
                    ])))
                    csv_file.flush()

            if (task is None and current is not None and current is not submitted
                    and now-received_at <= CAMERA_TIMEOUT):
                submitted, task_frame_at, task_revision, task_kind = current, received_at, revision, mission
                locked_id = (int(navigator.locked_key.split(":")[1])
                             if mission == "tag" and navigator.locked_key else args.tag_id)
                task = asyncio.create_task(asyncio.to_thread(
                    perception.process, current, mission, selection["hsv"], locked_id,
                ))
            if eof and task is None and current is submitted:
                break

            source_live = now-received_at <= CAMERA_TIMEOUT and (cam is None or cam.is_streaming)
            pipeline_live = source_live and now-vision_at <= PERCEPTION_TIMEOUT
            if now-last_control >= CONTROL_INTERVAL:
                last_control = now
                stopped = control.is_stopped if control else True
                if mission in ("wait", "done"):
                    motion = Motion(mission.upper(), hard_stop=True)
                elif pause_until is not None:
                    motion = Motion("STAGE_PAUSE")
                    if now >= pause_until:
                        mission, revision, pause_until = "tag", revision + 1, None
                        navigator = make_navigator(mission, args.tag_id)
                        vision_at = -math.inf
                else:
                    motion = navigator.tick(now, camera_live=pipeline_live, stopped=stopped)
                    if motion.phase == "ARRIVED" and stopped:
                        if mission == "color":
                            pause_until = now + STAGE_PAUSE_SECONDS
                        else:
                            mission = "done"
                if control:
                    async def apply_motion():
                        if motion.hard_stop:
                            await control.stop()
                        else:
                            for motor, speed in zip((LEFT_MOTOR, RIGHT_MOTOR), (motion.left, motion.right)):
                                await control.set_speed(motor, speed)
                    await asyncio.wait_for(apply_motion(), timeout=0.5)

            if not args.headless:
                preview = display.copy()
                label = f"{mission.upper()} / {motion.phase}"
                if motion.phase == "SEARCH":
                    label += f" / {navigator.search.phase} / {'left' if navigator.search_side < 0 else 'right'}"
                cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if result is not None:
                    if result.corners is not None:
                        cv2.polylines(preview, [result.corners.astype(np.int32)], True, (0, 255, 0), 2)
                    age = now-navigator.last_seen
                    text = f"vision {result.processing_ms:.0f}ms | sharpness {result.sharpness:.0f} | age {age:.2f}s"
                    cv2.putText(preview, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
                cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            await asyncio.sleep(0.005)
    finally:
        try:
            if control:
                await asyncio.wait_for(control.stop(), timeout=1)
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if cam:
                cam.stop()
            if capture:
                capture.release()
            if recorder:
                recorder.release()
            if csv_file:
                csv_file.close()
            cv2.destroyAllWindows()
            try:
                if connected:
                    await asyncio.wait_for(board.shutdown(), timeout=1)
            finally:
                if board is not None and board.sock is not None and board.sock.writer is not None:
                    board.sock.writer.close()
                    await board.sock.writer.wait_closed()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--tag-only", action="store_true")
    parser.add_argument("--tag-id", type=int, default=TARGET_TAG_ID)
    parser.add_argument("--backend", choices=("apriltag3", "opencv"), default=DETECTOR_BACKEND)
    parser.add_argument("--video", type=Path, help="Replay a recording; always disables Arduino connection")
    parser.add_argument("--headless", action="store_true", help="No preview; requires --video and --tag-only")
    parser.add_argument("--record", type=Path, help="Save unannotated processed frames to a new .avi file")
    parser.add_argument("--log-csv", type=Path, help="Write timing and tracking metrics to a new CSV file")
    args = parser.parse_args()
    if args.headless and not (args.video and args.tag_only):
        parser.error("--headless requires --video and --tag-only")
    return args


if __name__ == "__main__":
    try:
        asyncio.run(main(arguments()))
    except KeyboardInterrupt:
        pass
