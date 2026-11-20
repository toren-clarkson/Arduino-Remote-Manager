"""Follow objects with local YOLO weights trained on a Roboflow dataset.

Course: https://purwar-lab.github.io/ml-for-robotics-/?lesson=projA-local-inference
OpenCV only decodes/displays video; Ultralytics YOLO performs detection.
"""

import argparse
import asyncio
import math
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import phonesense

from robotics_lab.motor_control import SmoothMotorController
from robotics_lab.paths import MODELS
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int


# /////////////////////// Edit HERE
ARDUINO_IP = "10.13.96.68"
LEFT_MOTOR = MotorPins(in1=7, in2=4, enable=5)
RIGHT_MOTOR = MotorPins(in1=9, in2=8, enable=6)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models/best.pt"  # trained weights from Exercise A
TARGET_CLASS: str | None = None  # e.g. "ball"; None follows the largest detected object
CONFIDENCE = 0.5
IMAGE_SIZE = 416
DEVICE = "cpu"  # use "0" for a compatible CUDA GPU
PHONE_IP: str | None = None  # PC address shared with the phone, if auto choice is wrong

BASE_SPEED = 0.35
STEERING_GAIN = 0.5
MIN_PWM = 120
PWM_ACCELERATION = 180
PWM_DECELERATION = 240
MOTOR_UPDATE_INTERVAL = 0.05
INFERENCE_INTERVAL = 0.1  # at most ten detections per second; no queued frames
TARGET_HOLD_SECONDS = 0.25
MAX_RESULT_AGE = 0.75  # discard detections older than this, measured from source frame
CAMERA_TIMEOUT = 0.75
# /////////////////////// Edit HERE END


class Detection(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def detect(model, frame: np.ndarray) -> list[Detection]:
    """Run in one worker thread so inference cannot block the motor-update loop."""
    result = model.predict(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                           device=DEVICE, verbose=False)[0]
    if result.boxes is None:
        return []
    coordinates = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    detections = []
    for xyxy, confidence, class_id in zip(coordinates, scores, classes):
        box = Detection(*map(float, xyxy), float(confidence), str(result.names[int(class_id)]))
        if (all(math.isfinite(value) for value in box[:5])
                and box.x2 > box.x1 and box.y2 > box.y1
                and box.confidence >= CONFIDENCE):
            detections.append(box)
    return detections


def choose_target(detections: list[Detection]) -> Detection | None:
    matching = [box for box in detections
                if TARGET_CLASS is None or box.label.casefold() == TARGET_CLASS.casefold()]
    return max(matching, key=lambda box: box.area, default=None)


def steering_speeds(target: Detection, width: int) -> tuple[float, float]:
    offset = max(-1.0, min(1.0, ((target.x1 + target.x2) / 2 - width / 2) / (width / 2)))
    steering = STEERING_GAIN * offset
    return (max(0.0, min(1.0, BASE_SPEED + steering)),
            max(0.0, min(1.0, BASE_SPEED - steering)))


def target_is_fresh(now: float, source_time: float, last_result: float,
                    missing: bool) -> bool:
    return (now - source_time <= MAX_RESULT_AGE
            and (not missing or now - last_result <= TARGET_HOLD_SECONDS))


async def main(model, *, camera_only=False) -> None:
    board = None
    control = None
    connected = False
    cam = None
    inference = None
    try:
        cam = phonesense.start()
        if PHONE_IP:
            cam.select_ip(PHONE_IP)
        print("Phone camera URLs (choose the PC address on your phone's network):")
        for url in cam.phone_urls:
            print(" ", url)
        print("On the phone: accept the local HTTPS certificate, Connect, Start camera.")
        if not camera_only:
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

        print("Following:", TARGET_CLASS or "largest detected object")
        print("Camera-only preview." if camera_only else "Motor control enabled.")
        print("Press Esc or Q in the preview to quit.")
        window = "Phone - YOLO tracking"
        cv2.namedWindow(window)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        last_jpeg = submitted_jpeg = None
        frame_at = source_at = target_at = result_at = -float("inf")
        next_inference = last_motor_update = 0.0
        detections = []
        target = None
        missing = True
        target_width = frame.shape[1]
        result_shape = frame.shape[:2]
        detections_at = -float("inf")
        detections_shape = frame.shape[:2]
        status = "Waiting for phone camera"

        while True:
            now = time.monotonic()
            jpeg = cam.jpeg
            if jpeg is not None and jpeg is not last_jpeg:
                last_jpeg = jpeg
                decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    frame, frame_at = decoded, now
            camera_live = cam.is_streaming and now - frame_at <= CAMERA_TIMEOUT

            if inference is not None and inference.done():
                try:
                    detections = inference.result()
                except Exception as exc:
                    if control:
                        await control.stop()
                    raise RuntimeError("YOLO inference failed; check the model and DEVICE setting.") from exc
                inference = None
                detections_at, detections_shape = source_at, result_shape
                if camera_live and now - source_at <= MAX_RESULT_AGE:
                    selected = choose_target(detections)
                    missing = selected is None
                    if selected is not None:
                        result_at = now
                        target, target_at = selected, source_at
                        target_width = result_shape[1]
                    status = "Tracking" if selected else "Target missing"
                else:
                    detections, target, missing = [], None, True
                    status = "Detection too old; try IMAGE_SIZE = 320"

            if (camera_live and inference is None and now >= next_inference
                    and last_jpeg is not submitted_jpeg):
                submitted_jpeg = last_jpeg
                source_at, result_shape = frame_at, frame.shape[:2]
                inference = asyncio.create_task(asyncio.to_thread(detect, model, frame.copy()))
                next_inference = now + INFERENCE_INTERVAL

            if not camera_live:
                target, detections, missing = None, [], True
                status = "Camera disconnected / stale"
            if control and now - last_motor_update >= MOTOR_UPDATE_INTERVAL:
                last_motor_update = now
                if not camera_live:
                    await control.stop()
                else:
                    speeds = (steering_speeds(target, target_width)
                              if target is not None and target_is_fresh(now, target_at, result_at, missing)
                              else (0.0, 0.0))
                    for motor, speed in zip((LEFT_MOTOR, RIGHT_MOTOR), speeds):
                        await control.set_speed(motor, speed)

            preview = frame.copy()
            # Cached boxes are scaled to the current preview; expire with their source frame.
            if now - detections_at <= MAX_RESULT_AGE:
                sx, sy = frame.shape[1] / detections_shape[1], frame.shape[0] / detections_shape[0]
                for box in detections:
                    color = (0, 255, 0) if box == target else (180, 180, 180)
                    p1 = (int(box.x1 * sx), int(box.y1 * sy))
                    p2 = (int(box.x2 * sx), int(box.y2 * sy))
                    cv2.rectangle(preview, p1, p2, color, 2)
                    cv2.putText(preview, f"{box.label} {box.confidence:.0%}", p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
            await asyncio.sleep(0.01)
    finally:
        try:
            if control:
                await control.stop()
        finally:
            if inference is not None:
                inference.cancel()
                await asyncio.gather(inference, return_exceptions=True)
            if cam:
                cam.stop()
            cv2.destroyAllWindows()
            try:
                if connected:
                    await board.shutdown()
            finally:
                if board is not None and board.sock is not None and board.sock.writer is not None:
                    board.sock.writer.close()
                    await board.sock.writer.wait_closed()


def load_model(path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install YOLO first: .venv\\Scripts\\python.exe -m pip install ultralytics") from exc
    weights = Path(path)
    if not weights.is_file() and weights.name == "yolo26n.pt":
        weights = MODELS / "yolo26n.pt"
    if not weights.is_file() and weights.name != "yolo26n.pt":
        raise SystemExit(f"Model not found: {weights}. Copy your trained best.pt here, "
                         "or use --model yolo26n.pt for the pretrained COCO detector.")
    model = YOLO(str(weights))
    if model.task != "detect":
        raise SystemExit("Choose object-detection weights, not a classification/pose/segmentation model.")
    print("Model classes:", model.names)
    if TARGET_CLASS is not None and TARGET_CLASS.casefold() not in {str(v).casefold() for v in model.names.values()}:
        raise SystemExit(f"TARGET_CLASS {TARGET_CLASS!r} is not in this model's class names.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_PATH), help="Local best.pt weights, or yolo26n.pt for COCO.")
    parser.add_argument("--camera-only", action="store_true", help="Preview YOLO without connecting to the Arduino.")
    args = parser.parse_args()
    model = load_model(args.model)
    try:
        asyncio.run(main(model, camera_only=args.camera_only))
    except KeyboardInterrupt:
        pass
