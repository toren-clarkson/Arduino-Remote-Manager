# Stop-and-stare: YOLO / OpenCV object -> AprilTag

You can configure and run this mission through the [local webapp](../../webapp/README.md).
It also offers object-only and tag-only variants, ByteTrack / BoT-SORT / detection-only
selection, saved profiles, job logs and stationary image previews. The default
remains preview; the webapp connects motors only with explicit driving enabled.
Detector margin, decimation and threads are configurable; tag-only skips loading YOLO.

**New options:** `object_provider = hsv` enables OpenCV color detection with a
browser camera picker; `steering_controller = pid` enables bounded visual PID turns;
`position_estimator = kalman` enables timestamped horizontal memory. See
[CONTROL.md](CONTROL.md) for setup, units, calibration, limitations and component
interfaces. These options retain stop-and-stare; YOLO/timed/raw remain the defaults.

Web missions open the Mission monitor page with clickable PhoneSense URLs, a live
display-only camera feed, a separate stationary detector image, terminal output,
phase/leg, requested movement history and Stop. Raw JPEGs are published by a
background latest-frame writer during motion; the detector still runs only while
stopped. The monitor's requested commands are not measured wheel feedback. Source
implementation: [telemetry.py](telemetry.py); full usage: [webapp guide](../../webapp/README.md).

For command-line use, `--config path/to/config.json` loads Config fields from JSON;
model/class/IP command-line flags are ignored when this option is supplied.
`--stop-file path/to/marker` requests cooperative stopping when that marker exists.
`--headless` now supports either live PhoneSense or video replay without an OpenCV
window. The webapp uses these same runner hooks; driving is never permitted with replay.

This is the main mission for the motion-blur problem. The robot finds the configured object class, aligns to it, approaches until its image bounding box is large enough, then finds and approaches a specified AprilTag. **Every turn and forward movement is a separate timed step followed by a complete commanded stop and a stationary observation period.**

## 1. Install and configure

Run commands from the project root, `Main`, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Select `.venv\Scripts\python.exe` as the VS Code Python interpreter. The editable installation also lets VS Code run `run.py` directly; module commands below are the most explicit way to choose an entry point.

In [config.py](config.py), check:

- `arduino_ip`: the board's current Wi-Fi address, as shown by Serial Monitor. It is not auto-discovered.
- `phone_ip`: optionally the PC interface shared with the phone. Leave `None` to print all available URLs.
- `left_pins` / `right_pins`: `(IN1, IN2, PWM)`, currently `(7, 4, 5)` and `(9, 8, 6)`.
- `weights`, `target_class`, `tag_id`: the model and targets for this mission.

Keep the Telemetrix Wi-Fi firmware on the Uno R4. No custom UDP firmware is required. The phone browser connects to the **PC's HTTPS PhoneSense server**, normally port 8080. Python connects to the **Arduino's Telemetrix TCP server**, port 31336. Stop other scripts connected to the board before driving this mission.

## 2. Try a pretrained object first

The existing `models/yolo26n.pt` recognizes COCO classes. The default target is `bottle`; alternatives include `cup` and `sports ball`. Put the target in view, print the tag in `assets/tags/apriltag_36h11_id0.png`, and first run without motors:

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --target-class bottle --tag-id 0 --save-frames
```

This defaults to **preview only** and does not open an Arduino connection. It warms YOLO/ByteTrack before starting the camera/board, then waits stationary for your first camera frame. It simulates the movement schedule so you can check detections, size readings, and decisions. Open the printed PC URL on the phone, accept the local certificate warning, then Connect and Start camera. The preview deliberately freezes on the last stationary image during simulated motion and settling.

For actual robot movement:

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --drive --target-class bottle --tag-id 0 --arduino-ip 10.13.96.68
```

Replace the address with the board's current address. Esc/Q or Ctrl+C requests a stop and exits. Start physical calibration with wheels raised, then use short floor tests. PWM is not measured speed, and the robot has no encoder feedback to confirm a physical stop.

## 3. Use your own Roboflow-trained object

You train a **detector on labeled images**, not usually a tracker on a special video. Photos work; video is an optional way to collect images. Roboflow supplies labeling, dataset versions and export. Ultralytics trains YOLO locally or in a GPU notebook. ByteTrack associates detections during a stationary observation burst; it does not need a separate per-object training step.

Follow [../../training/README.md](../../training/README.md). After producing `models/custom_object.pt`, run:

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --weights models/custom_object.pt --target-class my_object --tag-id 0
```

Use the exact class name printed by the model. Add `--drive` when ready. A Roboflow project URL or `data.yaml` is not a trained model; `--weights` must point to a `.pt` detection model. No Roboflow key is needed during robot operation because inference is local.

## 4. What is implemented

Each decision cycle is strictly ordered:

```text
ZERO PWM -> SETTLE -> DISCARD NEW FRAMES -> OBSERVE 3 FRESH FRAMES
         -> CONFIRM TARGET -> PLAN ONE STEP
         -> RAMP UP -> TIMED HOLD -> RAMP DOWN TO ZERO -> repeat
```

During `RAMP_UP`, `MOVE`, `BRAKE`, `SETTLE`, and `FLUSH`, the program drains arriving JPEGs but does not decode them for detection, update YOLO/ByteTrack, or use them to change direction. Only `OBSERVE` runs inference. An active pulse is never corrected using moving-camera video. Keyboard input and a stream-health check remain active throughout a pulse.

After settling, it discards two additional fresh arrivals, then gathers three new frames. At least two consistent sightings of the same class/track ID (or the same tag ID) are required. Repeated cached JPEGs cannot count as new evidence. Stale inference is rejected and retried while stationary; consistently slow or missing frames stop the program rather than being treated as an empty scene.

### Object leg

- Uses **Ultralytics YOLO + ByteTrack**, filtered to `target_class`.
- Keeps a track ID across the stationary burst; resets the tracker before the next burst because a camera move changes the scene.
- If the object is off-center, makes one in-place turn toward it, then observes again.
- If centered but small, makes one forward step; forward holds shorten as the apparent object size grows.
- If centered and large enough in confirmed frames, changes to the AprilTag leg while stopped.

After a camera move, this is class-based reacquisition using last-position preference, not guaranteed identity recognition between identical objects. Use a distinctive custom class or one target object in the scene if following one particular physical object matters.

### AprilTag leg

- Uses AprilTag 3 through `pupil-apriltags`, family `tag36h11`.
- Accepts only `tag_id`, zero corrected bits, and decision margin at least 20 as an initial tuning choice.
- Uses the same separate align/forward/stop/observe steps.
- Finishes only after a stationary, centered, confirmed size threshold. Object detections cannot affect this leg.

### Searching

One empty observation burst causes another stationary look. After two confirmed empty bursts, the robot executes one search-turn pulse, stops, settles, and looks again. It searches toward the last-seen side for up to eight seconds, then falls back to the default rightward direction. It never drives forward when it has no confirmed target.

## 5. Equivalent of the OpenCV radius threshold

YOLO supplies a bounding box rather than a colored circle. This mission uses:

```text
object area fraction = (box width * box height) / (image width * image height)
```

`object_stop_area = 0.12` means the box occupies at least **12% of the image area**. The preview displays this percentage. The smallest size among the confirming sightings is used so one oversized box cannot end the leg.

For the tag:

```text
tag width fraction = average of four tag edge lengths / image width
```

`tag_stop_width = 0.24` means the tag side occupies about **24% of the image width**. Both thresholds also require horizontal alignment. Increase a threshold to approach closer; decrease it to stop farther away. These are visual size proxies, not meters. Object shape, viewpoint, occlusion and camera zoom affect them. For a literal circle-equivalent radius, `sqrt(box_area_pixels / pi)` is the radius of a circle with the same area, but the normalized area fraction avoids dependence on resolution.

## 6. Movement timing and settling

All calibration is in [config.py](config.py):

| Setting | Meaning |
|---|---|
| `forward_min_seconds`, `forward_max_seconds` | Full-speed hold for a forward step; shorter near target |
| `turn_min_seconds`, `turn_max_seconds` | Turn hold, scaled by horizontal error |
| `search_seconds` | One search-turn hold |
| `forward_speed`, `turn_speed`, `search_speed` | Normalized motor demands |
| `acceleration`, `deceleration`, `min_pwm` | PWM ramp rates and running floor |
| `settle_seconds` | Wait after commanded PWM reaches zero |
| `discard_frames` | Additional new arrivals discarded after settling |
| `observe_frames`, `confirmations` | Stationary evidence required per decision |
| `center_tolerance` | Horizontal alignment tolerance on the normalized -1..1 scale |

**Hold seconds exclude ramp-up and ramp-down.** Total movement time is approximately `target_PWM / acceleration + hold_seconds + target_PWM / deceleration`. At the defaults, a 0.18 demand targets about 144 PWM, so the ramps add roughly 0.84 seconds. Actual wheel movement begins when PWM overcomes friction, and depends on load and battery voltage. Shorten the holds, reduce speed, or adjust ramp rates based on physical tests. Avoid changing several values at once.

Timed turns cannot guarantee "10 degrees" without encoders, an IMU, or measured calibration. To calibrate, run a small pulse, measure the actual rotation, and adjust timing. If the robot is still wobbling when observation begins, increase `settle_seconds`. If arriving frames still show the earlier motion, increase settling and discarded-frame count.

The current PhoneSense JPEG API does not expose the phone's capture timestamp. Fresh arrival after settling helps clear old frames, but cannot mathematically prove that a delayed image was captured after the stop. Logs measure local receipt age, not full camera/network latency. The host stop/stream watchdog also does not replace a firmware-side motor command timeout on a laptop crash; Arduino firmware is unchanged.

## 7. Logs, saved images, and replay

Every run creates a folder under `runs/stop_and_stare/` with `events.csv`. It records phases, stationary frames, actual frame sequence used, inference age/duration, sightings, track IDs, thresholds' measured sizes, decisions and requested movement holds. `--save-frames` also saves the accepted **raw stationary JPEGs** in `observations/`; annotate these images in Roboflow if using them for training.

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --video path\to\clip.avi --target-class bottle --headless --max-batches 2
```

Replay cannot be combined with `--drive`. It simulates movement intervals and discards frames during those intervals, allowing detector/decision checks. It is not a simulation of robot physics and cannot prove the physical mission would complete. `--max-batches` limits debugging runs; `0` means no limit.

## 8. Files and verification

| File | Responsibility |
|---|---|
| `run.py` | Strict phase sequencing, preview, logging, fresh-frame gate |
| `config.py` | Hardware, model, thresholds and timing |
| `controller.py` | Pure target confirmation and next-step planning |
| `perception.py` | YOLO/ByteTrack and AprilTag detection, stationary-only |
| `motion.py` | Bounded motor pulses, ramps, stop and cleanup |
| `camera.py` | PhoneSense/video latest-frame sources |

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_stop_and_stare -v
```

Tests check the object-to-tag sequence, alignment before arrival, missing-target direction, wrong targets, confirmation/outlier rules, movement ramp/hold/brake order, camera-failure stop, fresh-frame gating and the absence of inference outside `OBSERVE`. Real YOLO/ByteTrack and AprilTag detector smoke checks complement these tests. Physical timing, balance and stopping distance still require your robot.

References: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [Ultralytics training](https://docs.ultralytics.com/modes/train/), [Roboflow dataset export](https://docs.roboflow.com/datasets/exporting-data), [AprilTag detector guidance](https://github.com/AprilRobotics/apriltag#tuning-the-detector-parameters).
