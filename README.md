# VIP 295 robot vision project

## Local webapp

Manage datasets, model libraries, training, AprilTags, configurable missions,
job logs and project files from the [Main control room](webapp/README.md):

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m webapp
```

Open **http://127.0.0.1:5000**. The app starts without connecting the robot;
mission preview is the default. Read the [plan](webapp/PLAN.md) and
[extension guide](webapp/EXTENDING.md) for providers, trackers and custom jobs.

The main program is the [stop-and-stare object -> AprilTag mission](missions/stop_and_stare/README.md), using YOLO or OpenCV color detection. It observes only while stopped, makes one bounded turn or forward step, stops and settles, then observes again. Optional PID steering and Kalman position memory are configured in the webapp.

## Start here

From this folder (`Main`), select `.venv\Scripts\python.exe` in VS Code and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --target-class bottle --tag-id 0
```

This starts a **preview-only** mission with the included pretrained `models/yolo26n.pt`; it does not connect to the Arduino. Open the PhoneSense PC URL printed in the terminal on the phone, Connect, then Start camera. The default pretrained model supports COCO classes such as `bottle`, `cup`, and `sports ball`.

To run the robot, first verify the IP/pins and timing in [config.py](missions/stop_and_stare/config.py), then use:

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --drive --target-class bottle --tag-id 0 --arduino-ip 10.13.96.68
```

The current IP is a configurable starting value, not auto-discovery. Arduino firmware remains Telemetrix Wi-Fi, TCP port 31336. PhoneSense runs on the PC, normally HTTPS port 8080. Stop other programs using the board first. Esc/Q and Ctrl+C exit and request zero PWM.

For a custom object, use [training/README.md](training/README.md) to collect/label images in Roboflow and train YOLO. Videos are optional image sources; a trained `.pt` detector is what the runtime needs. The object-tracker association does not require a separately labeled training video.

## Project layout

| Folder | Purpose and instructions |
|---|---|
| [missions/stop_and_stare/](missions/stop_and_stare/README.md) | Discrete YOLO/HSV object -> AprilTag missions and control components |
| [training/](training/README.md) | Video-frame extraction, Roboflow download/export, YOLO training |
| [demos/color_tracking/](demos/color_tracking/README.md) | Original continuous OpenCV color following |
| [demos/yolo_tracking/](demos/yolo_tracking/README.md) | Original continuous YOLO detection/following |
| [demos/apriltag/](demos/apriltag/README.md) | Earlier simple and filtered color -> tag demos |
| [demos/motors/](demos/motors/README.md) | Remote LED check and dual-motor template |
| [robotics_lab/](robotics_lab/README.md) | Shared PWM ramps, older image-navigation controller, paths |
| [tools/](tools/README.md) | Printable AprilTag generation |
| [models/](models/README.md) | Pretrained and custom `.pt` weights |
| [assets/tags/](assets/tags/README.md) | Printable markers |
| [datasets/](datasets/README.md) | Local images and Roboflow exports, ignored by Git |
| [runs/](runs/README.md) | Session logs, stationary snapshots and training outputs |
| [tests/](tests/README.md) | Hardware-free automated tests |
| [docs/FILE_MAP.md](docs/FILE_MAP.md) | Old filenames mapped to their new locations |

File names now use Python module naming. The old root-level scripts were moved, not discarded. Run examples with the documented `python -m ...` commands from the root. The editable installation also enables shared imports when running a demo directly in VS Code.

## Hardware and camera setup

The main mission's current left motor tuple is `(IN1=7, IN2=4, PWM=5)` and right tuple is `(IN1=9, IN2=8, PWM=6)`. Some older templates retain different directions/IPs; check their clearly marked settings rather than assuming every experiment has the same configuration.

Upload the Telemetrix Uno R4 Wi-Fi sketch as before and join the board and phone to a network reachable from the PC. Read the actual board address in Serial Monitor at 115200 baud. If the sketch waits at `while (!Serial)` before Wi-Fi setup, opening Serial Monitor lets it proceed. No new firmware or encoder code was uploaded by this cleanup.

On a PC with campus Wi-Fi, hotspot, WSL or VPN interfaces, choose the printed PhoneSense URL sharing the phone's network. A hotspot phone may need `https://192.168.137.1:8080/phone`; the correct address depends on your current interfaces. The server must remain running. Accept the local certificate warning and allow camera access. If the page cannot open even on the PC, first inspect the script's terminal output; the server may never have started or the script may have exited. If local access works but the phone times out, check shared network access and the intended Python/firewall rule.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Tests do not move physical motors. They exercise motion ramps, target confirmation, tag detectors, replay, and the stop-and-stare phase boundaries. The new mission's physical hold durations, settling time, balance and stopping distance require calibration on your robot. Training helpers do not train a custom model until you supply labeled data and invoke training.

## TODO

Implemented from the original list (October 2026):

- [x] **Visual steering PID:** optional variable-interval PID controls each turn's hold duration, with bounds, integral limiting, derivative smoothing and resets on target loss/change. Existing PWM ramps still handle acceleration and braking. Select `steering_controller = pid`; this is image-error steering, not measured motor-speed regulation.
- [x] **OpenCV in the webapp:** choose `object_provider = hsv` for object-only or object-to-AprilTag missions, without model weights. Leave `color_h = -1`, start a preview, then use **Mission monitor → Freeze frame & pick color** (or click the live video first). Click the target in the frozen image. The robot waits stopped until a color is selected and applies changes between observations.
- [x] **Field help:** question-mark help beside mission, job, upload, profile, model and editor inputs. Hover, focus with Tab, or tap to read the explanation. New schema fields can supply their own help text.
- [x] **Timestamped short-term estimation:** optional `position_estimator = kalman` smooths confirmed horizontal positions and predicts search guidance with age/uncertainty limits. Local camera-receipt timestamps drive the filter. Movement increases uncertainty and resets learned image velocity; predictions alone cannot authorize forward motion or arrival.
- [x] **Replaceable components:** trusted registries select object perception, steering, position estimation and motor execution independently. YOLO and HSV share the same mission planner. Existing timed steering, raw memory and ramped movement remain the defaults. See the [component contracts and tuning guide](missions/stop_and_stare/CONTROL.md) and [webapp extension guide](webapp/EXTENDING.md).

Future implementation:

- [ ] **Closed-loop wheel-speed PID:** add wheel encoders, define pin/tick/direction calibration and timestamped Telemetrix feedback, then implement per-wheel speed controllers and tests for stalled wheels, lost feedback and integral windup. Requires hardware measurements and compatible Arduino firmware; PWM demand is currently not a speed measurement.
- [ ] **Calibrated motion-aware prediction:** record actual yaw/translation versus motor demand, including ramp-up/down, battery level and load; add encoder/IMU or calibrated camera-motion input to the estimator. Do not assume a timed pulse equals a known angle. The present filter increases uncertainty after motion instead of inventing that relationship.
- [ ] **Capture-time synchronization:** extend the camera transport to include capture timestamps and sequence IDs with clock-offset estimation. Fresh local arrivals cannot fully measure phone/network buffering; validate prediction against actual capture latency.
- [ ] **Learning/adaptive tracking:** collect synchronized real-robot sessions with detections, commands and measured motion; build a replay benchmark for blur, dropouts, identical objects and overshoot. Train/evaluate an adaptive motion or uncertainty model offline before offering bounded opt-in adaptation. The current Kalman filter is a fixed mathematical estimator, not a learned model.
- [ ] **More mission/provider plugins:** add task-specific input schemas for new detectors, trackers and motion drivers; support workflow composition beyond the three existing sequences. Continuous-video trackers need an explicit timing/odometry design rather than bypassing stationary observation rules.
- [ ] **Physical tuning and fault validation:** calibrate HSV lighting tolerances, stop-area thresholds, PID gains, filter noise, friction PWM and settling on the actual robot. Verify balance and stopping distance; add a board-side motor-command timeout for host/network failure.

Hardware-free checks cover color selection, real color-to-tag perception, PID bounds,
filter uncertainty, component substitution and stop-and-stare boundaries. Browser
layout and physical robot behavior still need hands-on verification.
