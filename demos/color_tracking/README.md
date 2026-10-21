# OpenCV color-following demo

`track_color.py` is the former `default-opencv-tracking.py`.

```powershell
.\.venv\Scripts\python.exe -m demos.color_tracking.track_color
```

Set `transport_address`, motor pins and tuning values near the top. The script connects to Telemetrix on the board's event loop, starts PhoneSense, and displays phone frames. Click a color to select its HSV value; it follows the largest matching contour. Left/right offset changes wheel demands. Esc quits.

Implemented: circular hue thresholds, minimum contour area, proportional steering, last-sighting grace interval, shared PWM acceleration/deceleration, stale-stream stop and cleanup. There is no target-distance completion, automatic search spin, or AprilTag return. This demo uses continuous visual steering rather than the new stop-and-stare schedule. `MIN_PWM`, `BASE_SPEED`, `STEERING_GAIN`, `TARGET_HOLD_SECONDS`, and ramp rates need tuning for the robot.

Shared code is in `robotics_lab/motor_control.py`. For the fully discrete object-to-tag mission, see `missions/stop_and_stare/README.md`.
