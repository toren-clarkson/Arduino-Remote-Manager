"""Search timing checks, with no connection to the robot."""

import unittest
from typing import NamedTuple
from unittest.mock import AsyncMock

from robotics_lab.motor_control import SmoothMotorController, TimedSearch


class SearchTests(unittest.TestCase):
    def test_pause_starts_after_ramp_down(self):
        search = TimedSearch(0.6, 0.8)
        self.assertTrue(search.update(0, stopped=True))
        self.assertTrue(search.update(0.59, stopped=False))
        self.assertFalse(search.update(0.6, stopped=False))
        self.assertEqual(search.phase, "slowing")
        self.assertFalse(search.update(2, stopped=False))
        self.assertFalse(search.update(2.1, stopped=True))
        self.assertEqual(search.phase, "pause")
        self.assertFalse(search.update(2.89, stopped=True))
        self.assertTrue(search.update(2.91, stopped=True))
        self.assertFalse(search.update(3.52, stopped=False))

    def test_reset_starts_a_new_turn(self):
        search = TimedSearch(0.6, 0.8)
        search.update(0, stopped=True)
        search.update(1, stopped=True)
        self.assertEqual(search.phase, "pause")
        search.reset()
        self.assertTrue(search.update(20, stopped=True))
        self.assertTrue(search.update(20.5, stopped=False))

    def test_invalid_durations(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TimedSearch(value, 1)
            with self.assertRaises(ValueError):
                TimedSearch(1, value)


class RampIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_turns_include_a_full_zero_pwm_pause(self):
        class Pins(NamedTuple):
            in1: int
            in2: int
            enable: int

        motors = (Pins(7, 4, 5), Pins(9, 8, 6))
        now = 0.0
        control = SmoothMotorController(AsyncMock(), motors, min_pwm=80, clock=lambda: now)
        search = TimedSearch(0.6, 0.8)
        await control.stop()
        pause_started = None
        completed_pauses = 0
        for tick in range(120):
            now = tick * 0.05
            previous = search.phase
            turning = search.update(now, stopped=control.is_stopped)
            if search.phase == "pause" and previous != "pause":
                pause_started = now
            if previous == "pause" and turning:
                self.assertGreaterEqual(now - pause_started, 0.8)
                completed_pauses += 1
            for motor, speed in zip(motors, (0.005, -0.005) if turning else (0, 0)):
                await control.set_speed(motor, speed)
            if search.phase == "pause":
                self.assertTrue(control.is_stopped)
        self.assertGreaterEqual(completed_pauses, 2)


if __name__ == "__main__":
    unittest.main()
