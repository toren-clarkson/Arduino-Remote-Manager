import asyncio
import time
from enum import Enum, auto
from typing import NamedTuple

import cv2
import numpy as np
import phonesense
from robotics_lab.motor_control import SmoothMotorController, TimedSearch
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int

# /////////////////////// Edit HERE: Define your motor pins and board address
LEFT_MOTOR = MotorPins(in1=7, in2=4, enable=5)
RIGHT_MOTOR = MotorPins(in1=9, in2=8, enable=6)

board = TelemetrixUnoR4WiFiAio(transport_address="10.13.96.68")
# /////////////////////// Edit HERE END

# Tuning values
BASE_SPEED = 0.3
STEERING_GAIN = 0.5
SEARCH_SPEED = 0.7  # in-place turning speed while searching
SEARCH_TURN_SECONDS = 0.7  # turn-command duration, including smooth acceleration
SEARCH_PAUSE_SECONDS = 0.4  # stationary detection time AFTER PWM has ramped to zero
MIN_PWM = 120  # lowest PWM (0-255) at which the motors actually turn
MOTOR_UPDATE_INTERVAL = 0.05  # seconds between motor commands
TARGET_HOLD_SECONDS = 0.6  # bridge blur briefly using remembered position, slowing down
TARGET_MEMORY_SECONDS = 4.0  # remember which side to search after losing a target
MEMORY_FORWARD_SCALE = 0.5  # fraction of BASE_SPEED at the start of a detection gap
MEMORY_STEERING_SCALE = 0.5  # reduce steering during blur to avoid overshooting
MEMORY_DIRECTION_DEADBAND = 0.12  # keep previous search side when nearly centered
DETECTION_FRESH_SECONDS = 0.25  # cached frames are not new sightings
TARGET_LOST_TIMEOUT = 1.0  # seconds without the target before searching again
STOP_PAUSE = 0.5  # seconds to stand still between stages

COLOR_STOP_RADIUS = 80  # px, stop at the colored object when its circle reaches this
TAG_STOP_SIZE = 150  # px, stop at the tag when its average side length reaches this
TARGET_TAG_ID: int | None = None  # None accepts any tag in the dictionary

HUE_TOLERANCE = 10
SATURATION_TOLERANCE = 80
VALUE_TOLERANCE = 80
MIN_CONTOUR_AREA = 100

tag_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
    cv2.aruco.DetectorParameters(),
)


class Stage(Enum):
    WAIT_FOR_CLICK = auto()
    SEARCH_COLOR = auto()
    APPROACH_COLOR = auto()
    SEARCH_TAG = auto()
    APPROACH_TAG = auto()
    DONE = auto()


class ColorTracker:
    selected_hsv: tuple[int, int, int] | None = None
    frame_hsv: np.ndarray | None = None


# A detected target: (center_x, center_y, size_in_pixels)
Target = tuple[int, int, float]


class TargetMemory:
    """Last real sighting; memory use never refreshes its timestamp."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.target: Target | None = None
        self.last_seen = -float("inf")
        self.offset = 0.0
        self.direction = 1  # default right if we have no useful recent position

    def observe(self, target: Target, width: int, now: float):
        if now - self.last_seen > TARGET_MEMORY_SECONDS:
            self.direction = 1
        self.target = target
        self.last_seen = now
        self.offset = max(-1.0, min(1.0, (target[0] - width / 2) / (width / 2)))
        if abs(self.offset) > MEMORY_DIRECTION_DEADBAND:
            self.direction = 1 if self.offset > 0 else -1

    def search_direction(self, now: float) -> int:
        return self.direction if now - self.last_seen <= TARGET_MEMORY_SECONDS else 1

    def remembered_speeds(self, now: float) -> tuple[float, float]:
        age = now - self.last_seen
        if self.target is None or age >= TARGET_HOLD_SECONDS:
            return (0.0, 0.0)
        # Fade both forward motion and steering; never extrapolate approach distance.
        fade = max(0.0, 1.0 - age / TARGET_HOLD_SECONDS)
        forward = BASE_SPEED * MEMORY_FORWARD_SCALE * fade
        steering = STEERING_GAIN * self.offset * MEMORY_STEERING_SCALE * fade
        return (max(0.0, min(1.0, forward + steering)),
                max(0.0, min(1.0, forward - steering)))


def on_color_click(
    event: int, x: int, y: int, _: int, tracker: ColorTracker | None
) -> None:
    if (
        event == cv2.EVENT_LBUTTONDOWN
        and tracker is not None
        and tracker.frame_hsv is not None
        and 0 <= x < tracker.frame_hsv.shape[1]
        and 0 <= y < tracker.frame_hsv.shape[0]
    ):
        pixel = tracker.frame_hsv[y, x]
        tracker.selected_hsv = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        print(f"Tracking selected HSV color: {tracker.selected_hsv}")


def color_mask(hsv_frame: np.ndarray, selected_hsv: tuple[int, int, int]) -> np.ndarray:
    hue, saturation, value = selected_hsv
    lower_s = max(0, saturation - SATURATION_TOLERANCE)
    upper_s = min(255, saturation + SATURATION_TOLERANCE)
    lower_v = max(0, value - VALUE_TOLERANCE)
    upper_v = min(255, value + VALUE_TOLERANCE)
    lower_h = hue - HUE_TOLERANCE
    upper_h = hue + HUE_TOLERANCE

    def in_range(h_low: int, h_high: int) -> np.ndarray:
        return cv2.inRange(
            hsv_frame,
            np.array([h_low, lower_s, lower_v], dtype=np.uint8),
            np.array([h_high, upper_s, upper_v], dtype=np.uint8),
        )

    # Hue wraps around at 180, so a range crossing 0 needs two masks
    if lower_h < 0:
        return cv2.bitwise_or(in_range(0, upper_h), in_range(180 + lower_h, 179))
    if upper_h > 179:
        return cv2.bitwise_or(in_range(lower_h, 179), in_range(0, upper_h - 180))
    return in_range(lower_h, upper_h)


def find_color(frame: np.ndarray, hsv_frame: np.ndarray, hsv: tuple[int, int, int]) -> Target | None:
    """Return the largest blob of the selected color, with its enclosing-circle radius as size."""
    mask = color_mask(hsv_frame, hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
    cv2.circle(frame, center, 4, (0, 0, 255), -1)
    return center[0], center[1], float(radius)


def find_april_tag(frame: np.ndarray) -> Target | None:
    """Return the largest matching AprilTag, with its average side length as size."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = tag_detector.detectMarkers(gray)
    if ids is None:
        return None

    best: Target | None = None
    best_corners: np.ndarray | None = None
    best_id = 0
    for tag_corners, tag_id in zip(corners, ids.flatten()):
        if TARGET_TAG_ID is not None and int(tag_id) != TARGET_TAG_ID:
            continue
        points = tag_corners.reshape(4, 2)
        center = points.mean(axis=0)
        sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        size = float(sides.mean())
        if best is None or size > best[2]:
            best = (int(center[0]), int(center[1]), size)
            best_corners, best_id = tag_corners, tag_id

    if best is not None and best_corners is not None:
        cv2.aruco.drawDetectedMarkers(frame, (best_corners,), np.array([[best_id]]))
        cv2.circle(frame, (best[0], best[1]), 4, (0, 0, 255), -1)
    return best


# PWM counts per second; lower values make starts/stops more gradual.
PWM_ACCELERATION = 180
PWM_DECELERATION = 240
motor_controller = SmoothMotorController(
    board, (LEFT_MOTOR, RIGHT_MOTOR), min_pwm=MIN_PWM,
    acceleration=PWM_ACCELERATION, deceleration=PWM_DECELERATION,
)


async def setup_board() -> None:
    await board.disable_all_reporting()
    for motor in (LEFT_MOTOR, RIGHT_MOTOR):
        await board.set_pin_mode_digital_output(motor.in1)
        await board.set_pin_mode_digital_output(motor.in2)
        await board.set_pin_mode_analog_output(motor.enable)
    await motor_controller.stop()


async def move_motor(motor: MotorPins, speed: float) -> None:
    """Request speed from -1 to 1; PWM changes gradually on repeated calls."""
    await motor_controller.set_speed(motor, speed)


async def move_motors(left_speed: float, right_speed: float) -> None:
    await move_motor(LEFT_MOTOR, left_speed)
    await move_motor(RIGHT_MOTOR, right_speed)


async def main() -> None:
    cam = phonesense.start()
    print("On your phone, open:", cam.phone_url)
    tracker = ColorTracker()
    window_name = "phone"

    stage = Stage.WAIT_FOR_CLICK
    target: Target | None = None
    frame_width = 0
    color_memory = TargetMemory()
    tag_memory = TargetMemory()
    selected_color = None
    detected_at = -float("inf")
    last_jpeg = None
    last_motor_update = 0.0
    last_speeds = (0.0, 0.0)
    search = TimedSearch(SEARCH_TURN_SECONDS, SEARCH_PAUSE_SECONDS)
    search_stage = None
    search_direction = 1

    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_color_click, tracker)
        await setup_board()
        while stage is not Stage.DONE:
            # Yield even when the motor-update interval causes an early continue.
            await asyncio.sleep(0.01)
            jpeg = cam.jpeg
            if jpeg is not None and jpeg is not last_jpeg:
                last_jpeg = jpeg
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    tracker.frame_hsv = hsv_frame
                    frame_width = frame.shape[1]

                    if stage in (Stage.SEARCH_COLOR, Stage.APPROACH_COLOR) and tracker.selected_hsv:
                        target = find_color(frame, hsv_frame, tracker.selected_hsv)
                    elif stage in (Stage.SEARCH_TAG, Stage.APPROACH_TAG):
                        target = find_april_tag(frame)
                    else:
                        target = None

                    if target is not None:
                        detected_at = time.monotonic()
                        memory = color_memory if stage in (Stage.SEARCH_COLOR, Stage.APPROACH_COLOR) else tag_memory
                        memory.observe(target, frame_width, detected_at)

                    label = "Click the object to track" if stage is Stage.WAIT_FOR_CLICK else stage.name
                    if stage in (Stage.SEARCH_COLOR, Stage.SEARCH_TAG):
                        label += f" ({search.phase})"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if target is not None:
                        cv2.putText(
                            frame, f"size: {target[2]:.0f}px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        )
                    elif stage in (Stage.APPROACH_COLOR, Stage.APPROACH_TAG):
                        memory = color_memory if stage is Stage.APPROACH_COLOR else tag_memory
                        age = time.monotonic() - memory.last_seen
                        if age <= TARGET_MEMORY_SECONDS:
                            side = "left" if memory.search_direction(time.monotonic()) < 0 else "right"
                            cv2.putText(frame, f"Memory: {age:.1f}s, last seen {side}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
                    cv2.imshow(window_name, frame)

            if cv2.waitKey(1) == 27:
                break

            now = time.monotonic()
            if tracker.selected_hsv != selected_color:
                selected_color = tracker.selected_hsv
                color_memory.reset()
                if stage in (Stage.SEARCH_COLOR, Stage.APPROACH_COLOR):
                    target = None
                    stage = Stage.SEARCH_COLOR
                    search_stage = None
            if not frame_width or now - last_motor_update < MOTOR_UPDATE_INTERVAL:
                continue
            last_motor_update = now

            if not cam.is_streaming:
                target = None
                last_speeds = (0.0, 0.0)
                await motor_controller.stop()
                search.reset()
                search_stage = None
                color_memory.reset()
                tag_memory.reset()
                continue

            # Reprocessing the same JPEG must not keep a target alive indefinitely.
            visible = target is not None and now - detected_at <= DETECTION_FRESH_SECONDS
            if stage not in (Stage.SEARCH_COLOR, Stage.SEARCH_TAG):
                search_stage = None

            if stage is Stage.WAIT_FOR_CLICK:
                last_speeds = (0.0, 0.0)
                if tracker.selected_hsv is not None:
                    stage = Stage.SEARCH_COLOR

            elif stage in (Stage.SEARCH_COLOR, Stage.SEARCH_TAG):
                memory = color_memory if stage is Stage.SEARCH_COLOR else tag_memory
                if search_stage is not stage:
                    search.reset()
                    search_stage = stage
                    search_direction = memory.search_direction(now)
                if visible:
                    stage = Stage.APPROACH_COLOR if stage is Stage.SEARCH_COLOR else Stage.APPROACH_TAG
                    last_speeds = (0.0, 0.0)
                else:
                    previous_phase = search.phase
                    turning = search.update(now, stopped=motor_controller.is_stopped)
                    if turning and previous_phase == "pause":
                        search_direction = memory.search_direction(now)
                    last_speeds = ((search_direction * SEARCH_SPEED, -search_direction * SEARCH_SPEED)
                                   if turning else (0.0, 0.0))

            elif stage in (Stage.APPROACH_COLOR, Stage.APPROACH_TAG):
                is_color = stage is Stage.APPROACH_COLOR
                memory = color_memory if is_color else tag_memory
                stop_size = COLOR_STOP_RADIUS if is_color else TAG_STOP_SIZE
                if visible:
                    if target[2] >= stop_size:
                        await motor_controller.ramp_stop()
                        await asyncio.sleep(STOP_PAUSE)
                        print("Reached the object" if is_color else "Reached the AprilTag")
                        stage = Stage.SEARCH_TAG if is_color else Stage.DONE
                        if is_color:
                            tag_memory.reset()
                        target = None
                        last_speeds = (0.0, 0.0)
                    else:
                        # -1.0 at the left edge of the frame, +1.0 at the right edge
                        offset = (target[0] - frame_width / 2) / (frame_width / 2)
                        steering = STEERING_GAIN * offset
                        last_speeds = (
                            max(0.0, min(1.0, BASE_SPEED + steering)),
                            max(0.0, min(1.0, BASE_SPEED - steering)),
                        )
                else:
                    last_speeds = memory.remembered_speeds(now)
                    if now - memory.last_seen > max(TARGET_LOST_TIMEOUT, TARGET_HOLD_SECONDS):
                        stage = Stage.SEARCH_COLOR if is_color else Stage.SEARCH_TAG

            await move_motors(*last_speeds)
    finally:
        try:
            await motor_controller.stop()
        finally:
            cam.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # The board connected on this loop when it was created, so main() must run on it too
    board.loop.run_until_complete(main())
