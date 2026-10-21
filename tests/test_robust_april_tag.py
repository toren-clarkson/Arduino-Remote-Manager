"""Controller regressions plus real detector/video tests. Never connects to Arduino."""

import asyncio
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from robotics_lab.april_navigation import Navigator, Observation, Settings


spec = importlib.util.spec_from_file_location("robust_april", Path(__file__).resolve().parents[1] / "demos/apriltag/robust.py")
robot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = robot
spec.loader.exec_module(robot)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.nav = Navigator(.3, expected_key="tag:0")

    def observe(self, t, x=-.3, size=.1, key="tag:0"):
        return self.nav.ingest(Observation(key, x, 0, size), t, t)

    def acquired(self, x=-.3, size=.1):
        for t in (0, .1, .2):
            self.observe(t, x=x, size=size)
        self.assertTrue(self.nav.acquired)

    def tick(self, t, live=True, stopped=True):
        return self.nav.tick(t, camera_live=live, stopped=stopped)

    def test_confirmation_requires_distinct_consistent_frames(self):
        self.observe(0)
        for _ in range(5):
            self.assertFalse(self.nav.ingest(Observation("tag:0", 0, 0, .1), 0, .1))
        self.assertFalse(self.nav.acquired)
        self.nav.ingest(None, .1, .1)
        self.observe(.2)
        self.observe(.3)
        self.assertFalse(self.nav.acquired)
        self.observe(.4)
        self.assertTrue(self.nav.acquired)

    def test_track_coast_settle_then_left_search(self):
        self.acquired()
        self.assertEqual(self.tick(.21).phase, "TRACK")
        coast = self.tick(.4)
        self.assertEqual(coast.phase, "COAST")
        self.assertGreater(coast.right, coast.left)
        self.assertEqual(self.tick(.55).phase, "SETTLE")
        search = self.tick(.9)
        self.assertEqual(search.phase, "SEARCH")
        self.assertLess(search.left, 0)
        self.assertGreater(search.right, 0)

    def test_inconsistent_candidates_do_not_acquire(self):
        self.observe(0, x=-.8)
        self.observe(.1, x=.8)
        self.observe(.2, x=-.8)
        self.assertFalse(self.nav.acquired)

    def test_no_forward_motion_during_near_target_loss(self):
        self.acquired(x=.2, size=.25)
        coast = self.tick(.4)
        self.assertEqual(coast.phase, "COAST")
        self.assertAlmostEqual(coast.left + coast.right, 0)

    def test_reacquisition_after_long_gap_requires_confirmation(self):
        self.acquired()
        self.tick(.9)
        self.observe(1)
        self.assertEqual(self.tick(1.01).phase, "ACQUIRE")
        self.observe(1.1)
        self.assertFalse(self.nav.acquired)
        self.observe(1.2)
        self.assertEqual(self.tick(1.21).phase, "TRACK")

    def test_wrong_id_and_old_results_cannot_drive_or_finish(self):
        self.observe(0, key="tag:9", size=.5)
        self.assertFalse(self.nav.acquired)
        self.assertEqual(self.nav.last_seen, -float("inf"))
        self.assertFalse(self.nav.ingest(Observation("tag:0", 0, 0, .5), .1, 1))
        self.assertFalse(self.nav.reached)

    def test_automatic_id_lock_prevents_switching_tags(self):
        self.nav = Navigator(.3)
        self.acquired()
        self.observe(.3, key="tag:7")
        self.assertEqual(self.nav.locked_key, "tag:0")
        self.assertEqual(self.nav.last_seen, .2)

    def test_outlier_position_and_size_are_rejected(self):
        self.acquired(x=-.6)
        self.observe(.3, x=.8)
        self.assertEqual(self.nav.last_seen, .2)
        self.observe(.4, x=-.6, size=.8)
        self.assertEqual(self.nav.last_seen, .2)
        self.assertFalse(self.nav.reached)

    def test_filter_and_prediction_have_bounded_effect(self):
        self.acquired(x=0)
        self.observe(.3, x=.3)
        self.assertGreater(self.nav.x, 0)
        self.assertLess(self.nav.x, .3)
        self.nav.vx = 100  # even an unreasonable velocity cannot escape the bound
        self.assertLessEqual(abs(self.nav.predicted_x(1)-self.nav.x), .150001)

    def test_arrival_needs_real_centered_size_confirmations(self):
        self.observe(0, x=.02, size=.31)
        self.assertFalse(self.nav.reached)
        self.observe(.1, x=.02, size=.31)
        self.assertFalse(self.nav.reached)
        self.observe(.2, x=.02, size=.31)
        self.assertTrue(self.nav.reached)
        self.assertEqual(self.tick(.21).phase, "ARRIVED")

    def test_memory_cannot_confirm_arrival(self):
        self.acquired(x=.01, size=.28)
        for t in (.3, .4, .5, .6, .7):
            self.tick(t)
        self.assertFalse(self.nav.reached)

    def test_camera_watchdog_clears_estimate_and_stops(self):
        self.acquired()
        motion = self.tick(.3, live=False)
        self.assertTrue(motion.hard_stop)
        self.assertEqual((motion.left, motion.right), (0, 0))
        self.assertFalse(self.nav.acquired)

    def test_search_pause_waits_for_motor_ramp_to_stop(self):
        self.tick(0)
        self.assertEqual(self.tick(1.2, stopped=False).left, 0)
        self.assertEqual(self.nav.search.phase, "slowing")
        self.tick(2, stopped=True)
        self.assertEqual(self.nav.search.phase, "pause")
        self.assertEqual(self.tick(2.7).left, 0)
        self.assertGreater(self.tick(2.81).left, 0)


def marker_frame(tag_id=0, width=120, x=320):
    frame = np.full((480, 640, 3), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, width,
    )
    frame[180:180+width, x-width//2:x-width//2+width] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


class PerceptionTests(unittest.TestCase):
    def test_generated_tags_with_both_real_detectors(self):
        for backend in ("apriltag3", "opencv"):
            with self.subTest(backend=backend):
                detector = robot.Perception(backend)
                _, jpeg = cv2.imencode(".jpg", marker_frame())
                result = detector.process(jpeg.tobytes(), "tag", None, 0)
                self.assertEqual(result.observation.key, "tag:0")
                self.assertAlmostEqual(result.observation.x, 0, delta=.02)
                self.assertAlmostEqual(result.observation.size, 120/640, delta=.01)
                wrong = detector.process(jpeg.tobytes(), "tag", None, 1)
                self.assertIsNone(wrong.observation)
                self.assertIn("0", wrong.seen_ids)

    def test_color_detector_preserves_red_hue_wrap(self):
        detector = robot.Perception("opencv")
        frame = np.zeros((200, 200, 3), np.uint8)
        cv2.circle(frame, (100, 100), 25, (0, 0, 255), -1)
        _, jpeg = cv2.imencode(".jpg", frame)
        result = detector.process(jpeg.tobytes(), "color", (179, 255, 255), None)
        self.assertEqual(result.observation.key, "color")


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_color_to_tag_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            video, log = Path(folder)/"mission.avi", Path(folder)/"mission.csv"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
            self.assertTrue(writer.isOpened())
            for i in range(60):
                frame = np.full((480, 640, 3), 255, np.uint8)
                if i < 14:
                    cv2.circle(frame, (320, 240), 100, (0, 0, 255), -1)
                elif i >= 24:
                    frame = marker_frame(width=180)
                writer.write(frame)
            writer.release()
            callback = None
            clicked = False

            def register(window, cb):
                nonlocal callback
                callback = cb

            def show(window, image):
                nonlocal clicked
                if not clicked and image[240, 320, 2] > 200 and image[240, 320, 1] < 30:
                    callback(cv2.EVENT_LBUTTONDOWN, 320, 240, 0, None)
                    clicked = True

            args = SimpleNamespace(video=video, backend="apriltag3", camera_only=False,
                                   tag_only=False, tag_id=0, headless=False, record=None, log_csv=log)
            with patch.object(robot, "TelemetrixUnoR4WiFiAio", side_effect=AssertionError("Motor connection")), \
                 patch.object(robot.cv2, "namedWindow"), patch.object(robot.cv2, "setMouseCallback", side_effect=register), \
                 patch.object(robot.cv2, "imshow", side_effect=show), patch.object(robot.cv2, "waitKey", return_value=-1), \
                 patch.object(robot.cv2, "destroyAllWindows"):
                await robot.main(args)
            with log.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(clicked)
            self.assertTrue(any(row["mission"] == "color" and row["detected_key"] == "color" for row in rows))
            self.assertTrue(any(row["mission"] == "tag" and row["detected_key"] == "tag:0" for row in rows))
            self.assertTrue(any(row["phase"] == "DONE" for row in rows))

    async def test_replay_logs_and_records_without_any_motor_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
            self.assertTrue(writer.isOpened())
            for i in range(24):
                frame = marker_frame(1 if 12 <= i <= 15 else 0, x=300+i)
                if 6 <= i <= 9:
                    frame[:] = 255  # brief true detection loss
                writer.write(frame)
            writer.release()
            args = SimpleNamespace(video=video, backend="apriltag3", camera_only=False,
                                   tag_only=True, tag_id=0, headless=True,
                                   record=root/"recorded.avi", log_csv=root/"metrics.csv")
            with patch.object(robot, "TelemetrixUnoR4WiFiAio", side_effect=AssertionError("Replay connected to motors")):
                await robot.main(args)
            with args.log_csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 5)
            self.assertTrue(any(row["detected_key"] == "tag:0" for row in rows))
            self.assertTrue(any("1" in row["seen_ids"] for row in rows))
            self.assertFalse(any(row["detected_key"] == "tag:1" for row in rows))
            self.assertTrue(any(row["accepted_packet"] == "True" for row in rows))
            self.assertGreater(args.record.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
