"""A timed step always finishes at zero PWM before the next observation begins."""

import asyncio
import time
from typing import NamedTuple

from robotics_lab.motor_control import SmoothMotorController
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


class Pins(NamedTuple):
    in1: int
    in2: int
    enable: int


class PulseDriver:
    def __init__(self, cfg, *, drive=False, clock=time.monotonic, sleep=asyncio.sleep):
        self.cfg, self.drive, self.clock, self.sleep = cfg, drive, clock, sleep
        self.board = self.control = None
        self.motors = (Pins(*cfg.left_pins), Pins(*cfg.right_pins))

    async def open(self):
        if not self.drive:
            return
        self.board = TelemetrixUnoR4WiFiAio(
            transport_address=self.cfg.arduino_ip, autostart=False,
            loop=asyncio.get_running_loop(), shutdown_on_exception=False,
            close_loop_on_shutdown=False, hard_reset_on_shutdown=False,
        )
        await asyncio.wait_for(self.board.start_aio(), 15)
        self.control = SmoothMotorController(
            self.board, self.motors, min_pwm=self.cfg.min_pwm,
            acceleration=self.cfg.acceleration, deceleration=self.cfg.deceleration,
            clock=self.clock,
        )
        await self.board.disable_all_reporting()
        for motor in self.motors:
            await self.board.set_pin_mode_digital_output(motor.in1)
            await self.board.set_pin_mode_digital_output(motor.in2)
            await self.board.set_pin_mode_analog_output(motor.enable)
        await self.stop()

    def target_pwm(self, speed):
        return 0 if speed == 0 else (self.cfg.min_pwm+abs(speed)*(255-self.cfg.min_pwm))*(1 if speed > 0 else -1)

    async def stop(self):
        if self.control:
            await asyncio.wait_for(self.control.stop(), 1)

    async def _wait(self, seconds, phase, hook):
        deadline = self.clock()+seconds
        while self.clock() < deadline:
            await hook(phase)  # polls connection and keyboard only; never performs inference
            await self.sleep(self.cfg.motor_interval)

    async def _ramp(self, speeds, phase, hook):
        targets = [self.target_pwm(speed) for speed in speeds]
        if not self.control:
            rate = self.cfg.acceleration if any(speeds) else self.cfg.deceleration
            ramp_speeds = speeds if any(speeds) else getattr(self, "last_speeds", speeds)
            magnitude = max(abs(self.target_pwm(s)) for s in ramp_speeds)
            await self._wait(magnitude/rate, phase, hook)
            self.last_speeds = speeds
            return
        deadline = self.clock()+self.cfg.ramp_timeout
        while True:
            await hook(phase)
            for motor, speed in zip(self.motors, speeds):
                await asyncio.wait_for(self.control.set_speed(motor, speed), .5)
            if all(abs(self.control.states[m].pwm-target) < .01 for m, target in zip(self.motors, targets)):
                return
            if self.clock() >= deadline:
                raise TimeoutError("Motor ramp timed out; increase ramp_timeout or check acceleration")
            await self.sleep(self.cfg.motor_interval)

    async def pulse(self, step, hook):
        try:
            await self._ramp((step.left, step.right), "RAMP_UP", hook)
            await self._wait(step.hold_seconds, "MOVE", hook)
            await self._ramp((0, 0), "BRAKE", hook)
        finally:
            await self.stop()

    async def close(self):
        try:
            await self.stop()
        finally:
            if self.board is not None:
                try:
                    if self.control:
                        await asyncio.wait_for(self.board.shutdown(), 1)
                finally:
                    sock = self.board.sock
                    if sock is not None and sock.writer is not None:
                        sock.writer.close()
                        await sock.writer.wait_closed()
