"""Generate a tag36h11 marker with a white border in assets/tags/."""

import argparse
from pathlib import Path

import cv2

from robotics_lab.paths import TAGS

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag_id", type=int, nargs="?", default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if not 0 <= args.tag_id < len(dictionary.bytesList):
        parser.error(f"tag_id must be in [0, {len(dictionary.bytesList)-1}]")
    output = args.output or TAGS/f"apriltag_36h11_id{args.tag_id}.png"
    if output.exists():
        parser.error(f"File already exists: {output}. Use a new --output path.")
    tag = cv2.aruco.generateImageMarker(dictionary, args.tag_id, 600)
    tag = cv2.copyMakeBorder(tag, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), tag):
        raise RuntimeError(f"Could not write {output}")
    print("Saved", output)

if __name__ == "__main__":
    main()
