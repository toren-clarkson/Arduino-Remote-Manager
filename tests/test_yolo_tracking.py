"""Hardware-free checks for YOLO selection, freshness, and asynchronous control."""

import asyncio
import importlib.util
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np


spec = importlib.util.spec_from_file_location(
    "yolo_tracking", Path(__file__).resolve().parents[1] / "demos/yolo_tracking/track_yolo.py",
)
yolo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo)


class DetectionTests(unittest.TestCase):
    def test_class_filter_and_largest_box(self):
        small = yolo.Detection(0, 0, 10, 10, 0.9, "ball")
        large = yolo.Detection(0, 0, 20, 20, 0.8, "ball")
        other = yolo.Detection(0, 0, 40, 40, 0.99, "person")
        with patch.object(yolo, "TARGET_CLASS", "BALL"):
            self.assertEqual(yolo.choose_target([small, large, other]), large)
            self.assertIsNone(yolo.choose_target([other]))

    def test_steering_direction(self):
        left, right = yolo.steering_speeds(yolo.Detection(70, 0, 90, 10, 1, "ball"), 100)
        self.assertGreater(left, right)
        self.assertGreaterEqual(right, 0)
        self.assertLessEqual(left, 1)

    def test_old_frames_and_detection_gaps_expire(self):
        self.assertTrue(yolo.target_is_fresh(10.2, 10, 10.1, True))
        self.assertFalse(yolo.target_is_fresh(10.5, 10, 10.1, True))
        self.assertFalse(yolo.target_is_fresh(11, 10, 10.9, False))

    def test_yolo_outputs_are_filtered(self):
        boxes = MagicMock()
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 20, 20], [0, 0, 10, 10]])
        boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8, 0.2])
        boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 0])
        model = MagicMock()
        model.predict.return_value = [MagicMock(boxes=boxes, names={0: "ball"})]
        detections = yolo.detect(model, np.zeros((50, 50, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "ball")


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_inference_does_not_block_motor_updates(self):
        _, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
        polls = 0

        class Camera:
            phone_urls = ["https://test/phone"]
            stop = MagicMock()

            @property
            def jpeg(self):
                nonlocal polls
                polls += 1
                return bytes(bytearray(jpeg.tobytes()))

            @property
            def is_streaming(self):
                return polls < 35

        camera = Camera()
        board = MagicMock(sock=None)
        for name in ["start_aio", "disable_all_reporting", "set_pin_mode_digital_output",
                     "set_pin_mode_analog_output", "shutdown"]:
            setattr(board, name, AsyncMock())
        control = MagicMock(stop=AsyncMock(), set_speed=AsyncMock())
        inference_running = False
        updates_during_inference = 0

        async def record_update(*args):
            nonlocal updates_during_inference
            if inference_running:
                updates_during_inference += 1

        control.set_speed.side_effect = record_update

        def slow_detect(*args):
            nonlocal inference_running
            self.assertFalse(inference_running)  # never queue concurrent model calls
            inference_running = True
            try:
                time.sleep(0.12)
                return [yolo.Detection(20, 10, 40, 30, 0.9, "ball")]
            finally:
                inference_running = False

        with patch.object(yolo.phonesense, "start", return_value=camera), \
             patch.object(yolo, "TelemetrixUnoR4WiFiAio", return_value=board), \
             patch.object(yolo, "SmoothMotorController", return_value=control), \
             patch.object(yolo, "detect", side_effect=slow_detect), \
             patch.object(yolo.cv2, "namedWindow"), patch.object(yolo.cv2, "imshow"), \
             patch.object(yolo.cv2, "destroyAllWindows"), \
             patch.object(yolo.cv2, "waitKey", side_effect=lambda _: 27 if polls >= 55 else -1):
            await yolo.main(MagicMock())
        self.assertGreater(updates_during_inference, 0)
        self.assertTrue(any(call.args[1] > 0 for call in control.set_speed.call_args_list))
        self.assertGreaterEqual(control.stop.await_count, 3)  # setup, stream loss, exit
        camera.stop.assert_called_once()
        board.shutdown.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
