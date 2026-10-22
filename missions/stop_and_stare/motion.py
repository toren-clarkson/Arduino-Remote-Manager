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
        self.last_impulse = None
        self._integral = None

    def _sample_impulse(self):
        if self._integral is None or self.control is None:
            return
        now = self.clock()
        pwm = [float(self.control.states[m].written_pwm or 0)*(self.control.states[m].direction or 0)
               for m in self.motors]
        dt = max(0,now-self._impulse_time)
        for i in range(2):
            self._integral[i] += dt*(self._impulse_pwm[i]+pwm[i])/(2*255)
        self._impulse_time, self._impulse_pwm = now,pwm

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
            self._sample_impulse()
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
            self._sample_impulse()
            await hook(phase)
            for motor, speed in zip(self.motors, speeds):
                await asyncio.wait_for(self.control.set_speed(motor, speed), .5)
                self._sample_impulse()
            if all(abs(self.control.states[m].pwm-target) < .01 for m, target in zip(self.motors, targets)):
                return
            if self.clock() >= deadline:
                raise TimeoutError("Motor ramp timed out; increase ramp_timeout or check acceleration")
            await self.sleep(self.cfg.motor_interval)

    async def pulse(self, step, hook):
        self.last_impulse = None
        self._integral = [0.,0.]
        self._impulse_time = self.clock()
        self._impulse_pwm = [0.,0.]
        completed = False
        try:
            await self._ramp((step.left, step.right), "RAMP_UP", hook)
            if step.wheel_holds is None or self.control is None:
                await self._wait(step.hold_seconds, "MOVE", hook)
            else:
                started = self.clock()
                while self.clock()-started < step.hold_seconds:
                    self._sample_impulse()
                    await hook('MOVE')
                    for motor,speed,hold in zip(self.motors,(step.left,step.right),step.wheel_holds):
                        await asyncio.wait_for(self.control.set_speed(motor,speed if self.clock()-started < hold else 0),.5)
                        self._sample_impulse()
                    await self.sleep(self.cfg.motor_interval)
            await self._ramp((0, 0), "BRAKE", hook)
            completed = True
        finally:
            self._sample_impulse()
            await self.stop()
            self._sample_impulse()
            if completed and self.drive and self.control is not None:
                self.last_impulse = tuple(self._integral)
            self._integral = None

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
