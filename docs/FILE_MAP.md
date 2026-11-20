# Restored standalone files

Mission programs are back in the project root; these are actual implementations,
not launch wrappers around a dashboard.

| Previous organized location | Current file |
|---|---|
| `demos/color_tracking/track_color.py` | `default-opencv-tracking.py` |
| `demos/yolo_tracking/track_yolo.py` | `default-yolo-tracking.py` |
| `demos/apriltag/simple.py` | `simple-april-tag.py` |
| `demos/apriltag/robust.py` | `robust-april-tag.py` |
| `demos/apriltag/DESIGN.md` | `APRILTAG_DESIGN.md` |
| `demos/motors/dual_motor.py` | `dual_motor_template.py` |
| `demos/motors/blink_led.py` | `test_wifi_led.py` |
| `missions/stop_and_stare/` framework | `stop-and-stare.py` (simplified Red-Ball -> tag mission) |

`training/`, `tools/`, `robotics_lab/`, model weights, datasets and recorded results
remain in their supporting folders. Dashboard records were archived under
`runs/archived_webapp_20261030/`; the dashboard code was removed.

New standalone programs: `run.py` provides a console menu;
`adaptive-homing.py` implements continuous YOLO -> tag homing using learned
robot/surface response profiles. See [the guide](ADAPTIVE_HOMING.md).
