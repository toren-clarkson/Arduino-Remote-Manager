import asyncio
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

# /////////////////////// Edit HERE: Define your motor pins and board address
LEFT_MOTOR = MotorPins(in1=4, in2=7, enable=5)
RIGHT_MOTOR = MotorPins(in1=8, in2=9, enable=6)

board = TelemetrixUnoR4WiFiAio(transport_address="192.168.137.182")
# /////////////////////// Edit HERE END

MIN_PWM = 120  # Tune to the lowest PWM (0-255) that reliably starts your motors.


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

    try:
        await setup_board()
        while True:
            jpeg = cam.jpeg
            if jpeg is not None:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow("phone", frame)
            if cv2.waitKey(1) == 27:
                break
            
            # //////////////////////////// Edit HERE
            # Example motor control logic (replace with your own)
            if cam.is_streaming:
                # await move_motors(0.5, 0.5)
                pass
            else:
                await motor_controller.stop()
            # //////////////////////////// Edit HERE END
            # Socket writes may complete without yielding to other asyncio tasks.
            await asyncio.sleep(0.01)
    finally:
        try:
            await motor_controller.stop()
        finally:
            cam.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Keep camera/motor work on the loop used to open the board's Wi-Fi socket.
    board.loop.run_until_complete(main())
