"""Simple terminal entry point. Launch one standalone program with console settings."""

import subprocess
import sys
import argparse

from robotics_lab.console import choose
from robotics_lab.paths import PROJECT_ROOT

PROGRAMS = [
    ('Adaptive Red-Ball -> AprilTag (learned robot/surface profiles)', 'adaptive-homing.py'),
    ('Stop-and-stare Red-Ball -> AprilTag', 'stop-and-stare.py'),
    ('YOLO object following', 'default-yolo-tracking.py'),
    ('Robust color -> AprilTag', 'robust-april-tag.py'),
    ('Simple color -> AprilTag', 'simple-april-tag.py'),
    ('OpenCV color following', 'default-opencv-tracking.py'),
    ('Motor experiment template', 'dual_motor_template.py'),
    ('Wi-Fi LED test', 'test_wifi_led.py'),
    ('Adaptive OpenCV HSV color -> AprilTag (click color, learned profiles)', 'adaptive-homing.py', '--color'),
]


if __name__ == '__main__':
    try:
        argparse.ArgumentParser(description=__doc__).parse_args()
        index = choose('Robot programs', [entry[0] for entry in PROGRAMS])
        script = PROJECT_ROOT / PROGRAMS[index][1]
        raise SystemExit(subprocess.call([sys.executable, str(script), '--setup', *PROGRAMS[index][2:]], cwd=PROJECT_ROOT))
    except KeyboardInterrupt:
        print('\nCancelled.')
