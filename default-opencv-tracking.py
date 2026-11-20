import asyncio
import time
from typing import NamedTuple

import cv2
import numpy as np
import phonesense
from robotics_lab.motor_control import SmoothMotorController
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int


class ColorTracker:
    selected_hsv: tuple[int, int, int] | None = None
    frame_hsv: np.ndarray | None = None


LEFT_MOTOR = MotorPins(in1=7, in2=4, enable=5)
RIGHT_MOTOR = MotorPins(in1=9, in2=8, enable=6)

board = TelemetrixUnoR4WiFiAio(transport_address="10.13.96.68")

HUE_TOLERANCE = 10
SATURATION_TOLERANCE = 80
VALUE_TOLERANCE = 80
MIN_CONTOUR_AREA = 100

BASE_SPEED = 0.5
MIN_PWM = 120  # lowest PWM (0-255) at which the motors actually turn; tune this
STEERING_GAIN = 0.8
MOTOR_UPDATE_INTERVAL = 0.05  # seconds between motor commands
TARGET_HOLD_SECONDS = 0.25  # tolerate brief detection gaps before slowing down


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
    lower_saturation = max(0, saturation - SATURATION_TOLERANCE)
    upper_saturation = min(255, saturation + SATURATION_TOLERANCE)
    lower_value = max(0, value - VALUE_TOLERANCE)
    upper_value = min(255, value + VALUE_TOLERANCE)
    lower_hue = hue - HUE_TOLERANCE
    upper_hue = hue + HUE_TOLERANCE

    if lower_hue < 0:
        low_hue_mask = cv2.inRange(
            hsv_frame,
            np.array([0, lower_saturation, lower_value], dtype=np.uint8),
            np.array([upper_hue, upper_saturation, upper_value], dtype=np.uint8),
        )
        wrap_hue_mask = cv2.inRange(
            hsv_frame,
            np.array([180 + lower_hue, lower_saturation, lower_value], dtype=np.uint8),
            np.array([179, upper_saturation, upper_value], dtype=np.uint8),
        )
        return cv2.bitwise_or(low_hue_mask, wrap_hue_mask)
    if upper_hue > 179:
        high_hue_mask = cv2.inRange(
            hsv_frame,
            np.array([lower_hue, lower_saturation, lower_value], dtype=np.uint8),
            np.array([179, upper_saturation, upper_value], dtype=np.uint8),
        )
        wrap_hue_mask = cv2.inRange(
            hsv_frame,
            np.array([0, lower_saturation, lower_value], dtype=np.uint8),
            np.array([upper_hue - 180, upper_saturation, upper_value], dtype=np.uint8),
        )
        return cv2.bitwise_or(high_hue_mask, wrap_hue_mask)

    return cv2.inRange(
        hsv_frame,
        np.array([lower_hue, lower_saturation, lower_value], dtype=np.uint8),
        np.array([upper_hue, upper_saturation, upper_value], dtype=np.uint8),
    )


def track_color(frame: np.ndarray, tracker: ColorTracker) -> tuple[int, int] | None:
    """Draw tracking results on frame and return the target center in pixel coordinates."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tracker.frame_hsv = hsv_frame
    center: tuple[int, int] | None = None

    if tracker.selected_hsv is not None:
        mask = color_mask(hsv_frame, tracker.selected_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= MIN_CONTOUR_AREA:
                (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(center_x), int(center_y))
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Target: {center}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        status = f"Tracking HSV {tracker.selected_hsv} (click to change)"
    else:
        status = "Click a color to track"

    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return center


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
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_color_click, tracker)

    try:
        await setup_board()
        
        latest_obj_pos: tuple[int, int] | None = None
        frame_width = 0
        last_motor_update = 0.0
        last_jpeg = None
        last_target_seen = -float("inf")
        desired_speeds = (0.0, 0.0)

        while True:
            jpeg = cam.jpeg
            if jpeg is not None and jpeg is not last_jpeg:
                last_jpeg = jpeg
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    latest_obj_pos = track_color(frame, tracker)
                    if latest_obj_pos is not None:
                        last_target_seen = time.monotonic()
                    frame_width = frame.shape[1]
                    cv2.imshow(window_name, frame)

            now = time.monotonic()
            if frame_width and now - last_motor_update >= MOTOR_UPDATE_INTERVAL:
                last_motor_update = now
                if not cam.is_streaming:
                    desired_speeds = (0.0, 0.0)
                    await motor_controller.stop()
                elif latest_obj_pos is not None:
                    # -1.0 at the left edge of the frame, +1.0 at the right edge
                    offset = (latest_obj_pos[0] - frame_width / 2) / (frame_width / 2)
                    steering = STEERING_GAIN * offset
                    left_speed = max(0.0, min(1.0, BASE_SPEED + steering))
                    right_speed = max(0.0, min(1.0, BASE_SPEED - steering))
                    desired_speeds = (left_speed, right_speed)
                elif now - last_target_seen > TARGET_HOLD_SECONDS:
                    desired_speeds = (0.0, 0.0)
                await move_motors(*desired_speeds)

            if cv2.waitKey(1) == 27:
                break
            # Socket writes can finish without yielding; let asyncio service Wi-Fi I/O.
            await asyncio.sleep(0.01)

    finally:
        try:
            await motor_controller.stop()
        finally:
            cam.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # The board connected on this loop when it was created, so main() must run on it too
    board.loop.run_until_complete(main())
