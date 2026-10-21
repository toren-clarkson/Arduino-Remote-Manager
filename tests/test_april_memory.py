"""Hardware-free checks for remembered positions and directional reacquisition."""

import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio import telemetrix_uno_r4_wifi_aio as lib


spec = importlib.util.spec_from_file_location("april", Path(__file__).resolve().parents[1] / "demos/apriltag/simple.py")
april = importlib.util.module_from_spec(spec)
with patch.object(lib, "TelemetrixUnoR4WiFiAio", return_value=MagicMock()):
    spec.loader.exec_module(april)


class MemoryTests(unittest.TestCase):
    def test_remembers_left_and_right_then_expires(self):
        for x, direction in ((10, -1), (90, 1)):
            memory = april.TargetMemory()
            memory.observe((x, 50, 10), 100, 1)
            self.assertEqual(memory.search_direction(2), direction)
            self.assertEqual(memory.search_direction(1 + april.TARGET_MEMORY_SECONDS + .1), 1)

    def test_blur_motion_fades_without_refreshing_sighting(self):
        memory = april.TargetMemory()
        memory.observe((20, 50, 10), 100, 1)
        early = memory.remembered_speeds(1.1)
        late = memory.remembered_speeds(1.4)
        self.assertGreater(early[1], early[0])  # last target was to the left
        self.assertGreater(sum(early), sum(late))
        self.assertEqual(memory.remembered_speeds(1 + april.TARGET_HOLD_SECONDS + .01), (0, 0))
        self.assertEqual(memory.last_seen, 1)

    def test_center_does_not_flip_previous_search_side(self):
        memory = april.TargetMemory()
        memory.observe((10, 50, 10), 100, 1)
        memory.observe((51, 50, 10), 100, 1.1)
        self.assertEqual(memory.search_direction(1.2), -1)

    def test_families_and_reset_do_not_share_memory(self):
        color, tag = april.TargetMemory(), april.TargetMemory()
        color.observe((10, 50, 1000), 100, 1)
        self.assertEqual(tag.remembered_speeds(1.1), (0, 0))
        self.assertEqual(tag.search_direction(1.1), 1)
        color.reset()
        self.assertEqual(color.remembered_speeds(1.1), (0, 0))


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_color_and_tag_recover_toward_last_seen_left(self):
        for test_tag in (False, True):
            with self.subTest(tag=test_tag):
                await self.exercise(test_tag)

    async def test_frozen_detection_frame_expires_and_reacquires(self):
        for test_tag in (False, True):
            with self.subTest(tag=test_tag):
                await self.exercise(test_tag, frozen=True)

    async def exercise(self, test_tag, frozen=False):
        tick = 0
        _, jpeg = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
        cached_jpeg = jpeg.tobytes()

        class Camera:
            phone_url = "simulation"
            is_streaming = True
            stop = MagicMock()

            @property
            def jpeg(self):
                if frozen and tick >= (7 if test_tag else 4):
                    return cached_jpeg
                return bytes(bytearray(jpeg.tobytes()))

        tracker = april.ColorTracker()
        tracker.selected_hsv = (60, 255, 255)
        control = MagicMock(stop=AsyncMock(), ramp_stop=AsyncMock(), is_stopped=True)
        calls = []

        def key(_):
            nonlocal tick
            tick += 1
            return 27 if tick >= 30 else -1

        def color(*_):
            if test_tag:
                return (20, 50, april.COLOR_STOP_RADIUS + 1)
            return (20, 50, 10) if tick <= 4 else None

        def tag(*_):
            return (20, 50, 10) if tick <= 7 else None

        async def move(left, right):
            calls.append((tick, left, right))

        with patch.object(april.phonesense, "start", return_value=Camera()), \
             patch.object(april, "ColorTracker", return_value=tracker), \
             patch.object(april, "motor_controller", control), \
             patch.object(april, "setup_board", new_callable=AsyncMock), \
             patch.object(april, "move_motors", side_effect=move), \
             patch.object(april, "find_color", side_effect=color), \
             patch.object(april, "find_april_tag", side_effect=tag), \
             patch.object(april, "time", SimpleNamespace(monotonic=lambda: tick * .1)), \
             patch.object(april.asyncio, "sleep", new_callable=AsyncMock), \
             patch.object(april.cv2, "namedWindow"), patch.object(april.cv2, "setMouseCallback"), \
             patch.object(april.cv2, "imshow"), patch.object(april.cv2, "destroyAllWindows"), \
             patch.object(april.cv2, "waitKey", side_effect=key):
            await april.main()
        lost = 8 if test_tag else 5
        bridging = [(l, r) for t, l, r in calls if lost < t <= lost + 2]
        self.assertTrue(any(r > 0 and r > l for l, r in bridging), calls)
        self.assertTrue(any(l < 0 and r > 0 for t, l, r in calls if t > lost + 8), calls)
        control.stop.assert_awaited_once()  # normal exit; no connection failure simulated


if __name__ == "__main__":
    unittest.main()
