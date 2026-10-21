import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from training.download_dataset import safe_members
from training.extract_frames import extract


class TrainingToolTests(unittest.TestCase):
    def test_samples_video_without_overwriting_existing_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root/"sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
            self.assertTrue(writer.isOpened())
            for _ in range(20):
                writer.write(np.zeros((60, 80, 3), np.uint8))
            writer.release()
            self.assertEqual(extract(video, root/"images", 2), 4)
            with self.assertRaises(FileExistsError):
                extract(video, root/"images", 2)

    def test_rejects_dataset_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            for filename in ("../escape.txt", str(Path(folder).parent/"escape.txt")):
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as archive:
                    archive.writestr(filename, "bad")
                stream.seek(0)
                with zipfile.ZipFile(stream) as archive:
                    with self.assertRaises(ValueError):
                        list(safe_members(archive, Path(folder)/"dataset"))


if __name__ == "__main__":
    unittest.main()
