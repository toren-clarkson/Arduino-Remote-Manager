"""Mission planning and phase separation tests; no Arduino connection is permitted."""

import asyncio
import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

from missions.stop_and_stare.camera import Packet
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Planner, Sighting, Step, confirmed_sighting
from missions.stop_and_stare.motion import PulseDriver
from missions.stop_and_stare.run import Runner
from robotics_lab.motor_control import SmoothMotorController


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.planner = Planner(self.cfg)

    def test_complete_object_then_tag_mission(self):
        self.assertEqual(self.planner.plan(Sighting("object", "bottle", -.6, .02), 1).name, "turn_left")
        self.assertEqual(self.planner.plan(Sighting("object", "bottle", 0, .02), 2).name, "forward")
        self.assertEqual(self.planner.plan(Sighting("object", "bottle", 0, .13), 3).name, "object_reached")
        self.assertEqual(self.planner.leg, "tag")
        self.assertEqual(self.planner.plan(Sighting("tag", "tag:0", .5, .1), 4).name, "turn_right")
        self.assertEqual(self.planner.plan(Sighting("tag", "tag:0", 0, .25), 5).name, "done")

    def test_large_off_center_target_must_align_before_arrival(self):
        self.assertEqual(self.planner.plan(Sighting("object", "bottle", -.5, .8), 1).name, "turn_left")
        self.assertEqual(self.planner.leg, "object")

    def test_loss_searches_toward_last_side_after_confirmed_empty_batches(self):
        self.planner.plan(Sighting("object", "bottle", -.5, .02), 1)
        self.assertEqual(self.planner.plan(None, 2).name, "look_again")
        self.assertEqual(self.planner.plan(None, 3).name, "search_left")
        self.assertEqual(self.planner.plan(None, 20).name, "search_right")

    def test_wrong_class_or_tag_cannot_complete_a_leg(self):
        self.planner.plan(Sighting("object", "person", 0, .9), 1)
        self.assertEqual(self.planner.leg, "object")
        self.planner.leg = "tag"
        self.planner.plan(Sighting("tag", "tag:7", 0, .9), 2)
        self.assertEqual(self.planner.leg, "tag")

    def test_forward_steps_shorten_near_target(self):
        far = self.planner.plan(Sighting("object", "bottle", 0, .01), 1)
        near = self.planner.plan(Sighting("object", "bottle", 0, .11), 2)
        self.assertGreater(far.hold_seconds, near.hold_seconds)

    def test_burst_confirmation_keeps_ids_and_rejects_outliers(self):
        sample = Sighting("object", "bottle", .1, .05, 4)
        self.assertIsNone(confirmed_sighting([sample, None, None], self.cfg))
        self.assertEqual(confirmed_sighting([sample, sample, None], self.cfg), sample)
        self.assertIsNone(confirmed_sighting([sample, Sighting("object", "bottle", .1, .05, 9)], self.cfg))
        self.assertIsNone(confirmed_sighting([sample, Sighting("object", "bottle", -.8, .05, 4)], self.cfg))
        small = Sighting("object", "bottle", .1, .11, 4)
        large = Sighting("object", "bottle", .1, .20, 4)
        self.assertEqual(confirmed_sighting([small, large], self.cfg).size, .11)


class MotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pulse_ramps_holds_and_reaches_exact_zero(self):
        now = 0.0
        async def sleep(seconds):
            nonlocal now
            now += seconds
            await asyncio.sleep(0)
        cfg = Config()
        driver = PulseDriver(cfg, clock=lambda: now, sleep=sleep)
        driver.control = SmoothMotorController(AsyncMock(), driver.motors, min_pwm=cfg.min_pwm,
                                               acceleration=cfg.acceleration, deceleration=cfg.deceleration,
                                               clock=lambda: now)
        await driver.control.stop()
        phases = []
        async def hook(phase):
            phases.append(phase)
        await driver.pulse(Step("forward", .18, .18, .2), hook)
        self.assertEqual(list(dict.fromkeys(phases)), ["RAMP_UP", "MOVE", "BRAKE"])
        self.assertTrue(driver.control.is_stopped)
        self.assertGreater(now, .2)

    async def test_camera_failure_aborts_to_zero(self):
        cfg = Config()
        driver = PulseDriver(cfg)
        driver.control = SmoothMotorController(AsyncMock(), driver.motors)
        async def fail(_):
            raise RuntimeError("simulated lost camera")
        with self.assertRaises(RuntimeError):
            await driver.pulse(Step("turn", .18, -.18, .2), fail)
        self.assertTrue(driver.control.is_stopped)

    async def test_camera_failure_during_hold_stops_active_motors(self):
        now = 0.0
        async def sleep(seconds):
            nonlocal now
            now += seconds
        driver = PulseDriver(Config(), clock=lambda: now, sleep=sleep)
        driver.control = SmoothMotorController(AsyncMock(), driver.motors, clock=lambda: now)
        await driver.control.stop()
        async def hook(phase):
            if phase == "MOVE":
                self.assertFalse(driver.control.is_stopped)
                raise RuntimeError("camera disconnected during hold")
        with self.assertRaises(RuntimeError):
            await driver.pulse(Step("forward", .18, .18, .2), hook)
        self.assertTrue(driver.control.is_stopped)

    async def test_preview_does_not_skip_later_ramp_up(self):
        now = 0.0
        async def sleep(seconds):
            nonlocal now
            now += seconds
        driver = PulseDriver(Config(), clock=lambda: now, sleep=sleep)
        async def hook(_):
            pass
        durations = []
        for _ in range(2):
            started = now
            await driver.pulse(Step("forward", .18, .18, .1), hook)
            durations.append(now-started)
        self.assertAlmostEqual(durations[0], durations[1])


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_inference_never_becomes_search_evidence(self):
        import time
        import numpy as np
        cfg = replace(Config(), settle_seconds=.01, discard_frames=1, max_result_age=.001,
                      observation_timeout=.08)
        class Source:
            latest = None
            sequence = 0
            eof = False
            def poll(self, now):
                self.sequence += 1
                self.latest = Packet(self.sequence, now, b"fresh")
            def healthy(self, now):
                return True
        class Detector:
            def begin_burst(self, preferred):
                pass
            def detect(self, jpeg, leg):
                time.sleep(.02)
                return None, np.zeros((30, 40, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            driver = PulseDriver(cfg)
            driver.pulse = AsyncMock()
            runner = Runner(cfg, Source(), Detector(), driver, Path(folder), headless=True)
            with self.assertRaises(TimeoutError):
                await runner.run()
            driver.pulse.assert_not_awaited()

    async def test_no_inference_during_motion_and_post_settle_frames_only(self):
        cfg = replace(Config(), settle_seconds=.02, discard_frames=2, motor_interval=.001,
                      observe_frames=2, confirmations=2)
        holder = {}
        class Source:
            latest = None
            sequence = 0
            eof = False
            def poll(self, now):
                self.sequence += 1
                self.latest = Packet(self.sequence, now, str(self.sequence).encode())
            def healthy(self, now):
                return True
        source = Source()
        observations = []
        class Detector:
            def begin_burst(self, preferred):
                pass
            def detect(self, jpeg, leg):
                import numpy as np
                runner = holder["runner"]
                self_test.assertEqual(runner.phase, "OBSERVE")
                observations.append((leg, int(jpeg)))
                size = cfg.object_stop_area+.01 if leg == "object" else cfg.tag_stop_width+.01
                key = cfg.target_class if leg == "object" else "tag:0"
                return Sighting(leg, key, 0, size, 1 if leg == "object" else None), np.zeros((30, 40, 3), np.uint8)
        self_test = self
        with tempfile.TemporaryDirectory() as folder:
            driver = PulseDriver(cfg)
            runner = Runner(cfg, source, Detector(), driver, Path(folder), headless=True)
            holder["runner"] = runner
            outcome = await runner.run()
            self.assertEqual(outcome, "object and AprilTag reached")
            self.assertEqual([leg for leg, _ in observations], ["object", "object", "tag", "tag"])
            with (Path(folder)/"events.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for index, row in enumerate(rows):
                if row["event"] == "stationary_frame":
                    previous = rows[:index]
                    flush = next(r for r in reversed(previous) if r["phase"] == "FLUSH")
                    self.assertGreaterEqual(int(row["sequence"])-int(flush["sequence"]), cfg.discard_frames)

    async def test_runner_executes_pulse_before_next_inference(self):
        cfg = replace(Config(), settle_seconds=.01, motor_interval=.001, observe_frames=2,
                      confirmations=2, discard_frames=1, acceleration=10000, deceleration=10000,
                      forward_min_seconds=.005, forward_max_seconds=.005)
        holder, events = {}, []
        class Source:
            latest = None
            sequence = 0
            eof = False
            def poll(self, now):
                self.sequence += 1
                self.latest = Packet(self.sequence, now, b"fresh")
            def healthy(self, now):
                return True
        class Detector:
            def begin_burst(self, preferred):
                pass
            def detect(self, jpeg, leg):
                import numpy as np
                events.append(holder["runner"].phase)
                return Sighting("object", "bottle", 0, .01, 1), np.zeros((30, 40, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            driver = PulseDriver(cfg)
            runner = Runner(cfg, Source(), Detector(), driver, Path(folder), headless=True)
            holder["runner"] = runner
            await runner.run(max_batches=2)
            with (Path(folder)/"events.csv").open() as handle:
                phases = [row["phase"] for row in csv.DictReader(handle) if row["event"] == "phase"]
            first_move = phases.index("MOVE")
            next_observe = phases.index("OBSERVE", first_move)
            self.assertIn("BRAKE", phases[first_move:next_observe])
            self.assertIn("SETTLE", phases[first_move:next_observe])
            self.assertIn("FLUSH", phases[first_move:next_observe])
            self.assertEqual(set(events), {"OBSERVE"})


if __name__ == "__main__":
    unittest.main()
