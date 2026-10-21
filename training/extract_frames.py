"""Sample unlabeled images from a video for annotation in Roboflow."""

import argparse
import math
from pathlib import Path

import cv2


def extract(video, output, sample_fps=2.0):
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    saved = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame-rate metadata")
        output.mkdir(parents=True, exist_ok=True)
        frame_index, next_sample = 0, 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index/fps >= next_sample:
                path = output/f"{video.stem}_{frame_index:08d}.jpg"
                if path.exists():
                    raise FileExistsError(path)
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Cannot write {path}")
                saved += 1
                next_sample += 1/sample_fps
            frame_index += 1
    finally:
        capture.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2, help="Images to sample per second")
    args = parser.parse_args()
    print("Saved", extract(args.video, args.output, args.fps), "unlabeled images to", args.output)


if __name__ == "__main__":
    main()
