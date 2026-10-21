"""Hardware-free checks: python -m unittest test_motor_control -v."""

import unittest
from typing import NamedTuple

from robotics_lab.motor_control import SmoothMotorController


class Pins(NamedTuple):
    in1: int
    in2: int
    enable: int


class FakeBoard:
    def __init__(self):
        self.commands = []

    async def analog_write(self, pin, value):
        self.commands.append(("pwm", pin, value))

    async def digital_write(self, pin, value):
        self.commands.append(("direction", pin, value))


class MotionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 0.0
        self.motor = Pins(4, 7, 5)
        self.board = FakeBoard()
        self.control = SmoothMotorController(
            self.board, [self.motor], clock=lambda: self.now,
        )
        await self.control.stop()

    async def step(self, speed):
        self.now += 0.05
        await self.control.set_speed(self.motor, speed)
        return self.control.states[self.motor].pwm

    async def test_acceleration_and_deceleration_are_bounded(self):
        previous = 0.0
        for _ in range(30):
            value = await self.step(0.5)
            self.assertGreaterEqual(value, previous)
            self.assertLessEqual(value - previous, 9.000001)
            previous = value
        self.assertAlmostEqual(previous, 187.5)
        for _ in range(20):
            value = await self.step(0)
            self.assertLessEqual(value, previous)
            self.assertLessEqual(previous - value, 12.000001)
            previous = value
        self.assertEqual(previous, 0)

    async def test_running_updates_do_not_pulse_pwm_to_zero(self):
        self.board.commands.clear()
        for _ in range(40):
            await self.step(0.5)
        running_pwm = [value for kind, _, value in self.board.commands if kind == "pwm"]
        self.assertTrue(running_pwm)
        self.assertNotIn(0, running_pwm)
        self.board.commands.clear()
        for _ in range(10):
            await self.step(0.5)
        self.assertEqual(self.board.commands, [])

    async def test_reverse_passes_through_zero(self):
        for _ in range(30):
            await self.step(0.5)
        values = [await self.step(-0.5) for _ in range(50)]
        self.assertIn(0, values)
        zero = values.index(0)
        self.assertTrue(all(value >= 0 for value in values[:zero]))
        self.assertTrue(any(value < 0 for value in values[zero + 1:]))
        # Each direction change is made with PWM already disabled.
        pwm = 0
        for kind, _, value in self.board.commands:
            if kind == "pwm":
                pwm = value
            else:
                self.assertEqual(pwm, 0)

    async def test_emergency_stop_bypasses_ramp(self):
        for _ in range(30):
            await self.step(1)
        await self.control.stop()
        self.assertEqual(self.control.states[self.motor].pwm, 0)
        self.assertIn(("pwm", 5, 0), self.board.commands[-3:])


if __name__ == "__main__":
    unittest.main()
