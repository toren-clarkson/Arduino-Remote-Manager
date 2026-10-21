# Continuous YOLO-following demo

`track_yolo.py` is the former `default-yolo-tracking.py`. This older demo uses YOLO detections to steer continuously; the new mission adds ByteTrack IDs, discrete movement and an AprilTag leg.

```powershell
.\.venv\Scripts\python.exe -m demos.yolo_tracking.track_yolo --camera-only --model models/yolo26n.pt
```

For custom weights, use `--model models/custom_object.pt`. Edit `TARGET_CLASS`, `ARDUINO_IP`, motor pins and `PHONE_IP` near the top. `TARGET_CLASS=None` follows the largest detection of any class. Remove `--camera-only` to connect to motors.

Implemented: local Ultralytics YOLO inference in one worker thread, largest matching detection selection, scaled bounding-box preview, proportional steering, brief loss tolerance, source-result age limit, camera watchdog and shared smooth PWM ramps. Only the latest camera frame is submitted, so there is no inference queue. The script does not stop at a proximity threshold and does not transition to a tag. Esc/Q quits.

Roboflow preparation/training is described in `training/README.md`. For the requested stop-and-stare mission, use `missions/stop_and_stare/README.md` instead of modifying this comparison demo.
