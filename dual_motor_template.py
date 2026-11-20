import asyncio
from typing import NamedTuple

import cv2
import numpy as np
import phonesense
from robotics_lab.network import ARDUINO_IP, TELEMETRIX_PORT, configure_phone_camera
from robotics_lab.motor_control import SmoothMotorController
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int

# /////////////////////// Edit HERE: Motor pins; AP address comes from robotics_lab.network
LEFT_MOTOR = MotorPins(in1=4, in2=7, enable=5)
RIGHT_MOTOR = MotorPins(in1=8, in2=9, enable=6)

board = None
board_ready = False
# /////////////////////// Edit HERE END

MIN_PWM = 120  # Tune to the lowest PWM (0-255) that reliably starts your motors.


# PWM counts per second; lower values make starts/stops more gradual.
PWM_ACCELERATION = 180
PWM_DECELERATION = 240
motor_controller = None


async def setup_board() -> None:
    global board, board_ready, motor_controller
    board_ready = False
    motor_controller = None
    board = TelemetrixUnoR4WiFiAio(
        transport_address=ARDUINO_IP, ip_port=TELEMETRIX_PORT, autostart=False, loop=asyncio.get_running_loop(),
        shutdown_on_exception=False, close_loop_on_shutdown=False, hard_reset_on_shutdown=False,
    )
    await asyncio.wait_for(board.start_aio(), timeout=15)
    board_ready = True
    motor_controller = SmoothMotorController(
        board, (LEFT_MOTOR, RIGHT_MOTOR), min_pwm=MIN_PWM,
        acceleration=PWM_ACCELERATION, deceleration=PWM_DECELERATION,
    )
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
    configure_phone_camera(cam)

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
            if motor_controller is not None:
                await motor_controller.stop()
        finally:
            cam.stop()
            cv2.destroyAllWindows()
            if board is not None:
                try:
                    if board_ready:
                        await asyncio.wait_for(board.shutdown(), timeout=1)
                finally:
                    sock = getattr(board, 'sock', None)
                    if sock is not None and sock.writer is not None:
                        sock.writer.close()
                        await sock.writer.wait_closed()


if __name__ == "__main__":
    from robotics_lab.console import configure_script
    configure_script(globals())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
