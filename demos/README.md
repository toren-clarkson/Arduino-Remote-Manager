# Earlier experiments

These grouped demos preserve the earlier work. Their folder READMEs describe how to run each one. They are useful comparisons, but they do not all implement the professor's fully separated stop-and-stare movement schedule.

- [color_tracking](color_tracking/README.md): continuous HSV color following.
- [yolo_tracking](yolo_tracking/README.md): continuous YOLO detection and steering.
- [apriltag](apriltag/README.md): earlier color-to-tag sequences with timed searches and memory/filtering.
- [motors](motors/README.md): LED connectivity and a camera/motor template.

For all movement to be discrete, use [the main mission](../missions/stop_and_stare/README.md). Commands in these READMEs run from the project root after `pip install -e .`. Close other board clients first and verify each demo's IP and pin settings.
