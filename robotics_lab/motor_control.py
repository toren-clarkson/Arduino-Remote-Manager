"""PWM ramping shared by the camera demos. No hardware connection on import."""

import asyncio
import math
import time
from dataclasses import dataclass


@dataclass
class _MotorState:
    pwm: float = 0.0
    written_pwm: int | None = None
    direction: int | None = None
    updated: float | None = None


class SmoothMotorController:
    def __init__(self, board, motors, *, min_pwm=120, acceleration=180,
                 deceleration=240, clock=time.monotonic):
        self.board = board
        self.motors = tuple(motors)
        self.min_pwm = min_pwm
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.clock = clock
        self.states = {motor: _MotorState() for motor in self.motors}

    @property
    def is_stopped(self):
        """True once both commanded PWM ramps have reached zero (not encoder feedback)."""
        return all(state.pwm == 0 for state in self.states.values())

    async def set_speed(self, motor, speed, *, immediate=False):
        state = self.states[motor]
        now = self.clock()
        dt = 0.05 if state.updated is None else min(0.1, max(0.0, now - state.updated))
        state.updated = now
        speed = max(-1.0, min(1.0, speed))
        target = 0.0 if speed == 0 else (self.min_pwm + abs(speed) * (255 - self.min_pwm)) * (1 if speed > 0 else -1)
        # Reverse only after ramping the previous direction down to zero.
        if state.pwm * target < 0:
            target = 0.0
        if immediate:
            if speed != 0:
                raise ValueError("Only a stop can bypass the PWM ramp")
            state.pwm = 0.0
        else:
            rate = self.acceleration if abs(target) > abs(state.pwm) else self.deceleration
            change = rate * dt
            state.pwm += max(-change, min(change, target - state.pwm))

        pwm = round(abs(state.pwm))
        direction = 0 if pwm == 0 else (1 if state.pwm > 0 else -1)
        if direction != state.direction:
            # Disable the bridge before changing direction, never on steady updates.
            if state.written_pwm != 0:
                await self.board.analog_write(motor.enable, 0)
                state.written_pwm = 0
            await self.board.digital_write(motor.in1, int(direction > 0))
            await self.board.digital_write(motor.in2, int(direction < 0))
            state.direction = direction
        if pwm != state.written_pwm:
            await self.board.analog_write(motor.enable, pwm)
            state.written_pwm = pwm

    async def stop(self):
        for motor in self.motors:
            await self.set_speed(motor, 0, immediate=True)

    async def ramp_stop(self):
        while not self.is_stopped:
            for motor in self.motors:
                await self.set_speed(motor, 0)
            await asyncio.sleep(0.02)


class TimedSearch:
    """Nonblocking turn -> ramp down -> stationary pause -> repeat schedule.

    Call update every motor tick, passing the motor controller's is_stopped flag.
    A False result requests zero speed through the normal smooth ramp.
    """

    def __init__(self, turn_seconds, pause_seconds):
        if not all(math.isfinite(value) and value > 0
                   for value in (turn_seconds, pause_seconds)):
            raise ValueError("Search turn and pause durations must be positive finite seconds")
        self.turn_seconds = turn_seconds
        self.pause_seconds = pause_seconds
        self.reset()

    def reset(self):
        self.phase = "turn"
        self.started = None

    def update(self, now, *, stopped):
        if self.started is None:
            self.started = now
        if self.phase == "turn" and now - self.started >= self.turn_seconds:
            self.phase = "slowing"
        if self.phase == "slowing" and stopped:
            self.phase = "pause"
            self.started = now
        if self.phase == "pause" and now - self.started >= self.pause_seconds:
            self.phase = "turn"
            self.started = now
        return self.phase == "turn"
