# Robust AprilTag approach for this robot

`demos/apriltag/robust.py` is a separate implementation. It keeps the existing PhoneSense -> laptop -> Telemetrix Uno R4 Wi-Fi arrangement and the color -> tag sequence. `demos/apriltag/simple.py` remains available for comparison.

## Research and engineering choice

For this robot, the first useful improvement is a controller that treats observations as noisy, intermittent evidence. The implementation separates detection, target estimation, navigation state, and motor output. The attached research notes informed the design; the following primary references were checked independently:

- [AprilRobotics AprilTag 3](https://github.com/AprilRobotics/apriltag#tuning-the-detector-parameters): reducing quad decimation improves detection range at a processing cost; sharper decoding can be tuned with example images. The new default uses full-resolution quad detection (`quad_decimate=1.0`), edge refinement, and the existing `tag36h11` family so printed tags remain usable.
- [Pupil Labs AprilTag 3 bindings](https://pupil-apriltags.readthedocs.io/en/stable/): provides the native detector through Python, including detector configuration. A Windows/Python 3.13 wheel was installed and tested on this machine. The `--backend opencv` option permits a comparison with the existing OpenCV installation.
- [PhotonVision camera tuning](https://docs.photonvision.org/en/latest/docs/pipelines/input.html#apriltags-and-motion-blur): exposure, resolution, lighting, and movement affect tag readability. Short exposure with sufficient light and deliberate stationary observation intervals remain necessary when moving images are unreadable. Filtering can bridge a brief gap; it cannot reconstruct missing image detail.
- [OpenCV marker detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html): marker identity comes from its code, and calibrated geometry is required for pose estimation. This implementation uses normalized image position and apparent size; those values are not robot pose or distance in meters.

These references do not establish universally best speed limits or timing values. The settings below are conservative engineering starting points for experiments on your actual robot. A filter with bounded image-velocity prediction is used here; wheel encoders, an IMU, and calibrated camera pose would be needed to support stronger motion estimates through longer gaps.

## What changes

1. **Confirm the target.** Require three matching observations spanning at least 0.12 seconds. Misses, mismatched IDs, long gaps, or inconsistent candidate positions restart confirmation. `TARGET_TAG_ID=0` matches the project's generated tag. Set another ID as needed. With `None`, the first consistently confirmed tag is locked for that mission, preventing switching between visible IDs.
2. **Filter position and limit prediction.** A time-based exponential filter smooths center/size. A filtered image velocity predicts horizontal position for at most 0.30 seconds and at most 0.15 normalized-image units. Position/size jumps are rejected while tracking. Prediction never supplies arrival evidence.
3. **Control movement continuously.** Deadband, limited turn demand, reduced forward demand during large steering corrections, and slowing with apparent target size precede the existing PWM ramps. PWM values are not calibrated wheel velocities; `MIN_PWM` still needs tuning to the motor driver and load.
4. **Handle loss in stages.** A recently confirmed target enters `COAST` during a short gap, with reduced, fading movement. Near the target, the forward component is zero during this gap. After 0.30 seconds the controller requests zero and settles; after 0.65 seconds it begins a turn-and-pause search toward the last-seen side. Direction memory lasts four seconds. A lost target must be confirmed again before approach resumes.
5. **Confirm arrival.** Three real, centered detections spanning 0.15 seconds must meet the size threshold. Large uncorroborated size jumps are rejected. Stop commands ramp down, and the color-to-tag transition waits for commanded motor PWM to reach zero, then pauses. No encoder feedback is available to prove a physical stop.
6. **Keep perception out of the motor loop.** One worker performs detection; the main loop continues motor updates at 20 Hz. Only the latest available frame is submitted, with no accumulated frame queue. Results older than 0.35 seconds from local receipt are rejected. Camera/perception watchdogs request an immediate stop on stale input, and communication writes are bounded by timeouts.

The host watchdog cannot protect against every laptop/process or network failure. The existing Arduino firmware was not modified; a firmware-side command timeout would be a separate improvement. Immediate stops on stale data/errors remain distinct from ordinary gradual speed changes.

## Run and tune

The additional dependency is already installed here. For another environment:

```powershell
.\.venv\Scripts\python.exe -m pip install pupil-apriltags
```

Start with tag-only perception:

```powershell
.\.venv\Scripts\python.exe -m demos.apriltag.robust --camera-only --tag-only --tag-id 0
```

For the full sequence, verify `ARDUINO_IP`, motor pins, and target ID near the top, then run:

```powershell
.\.venv\Scripts\python.exe -m demos.apriltag.robust
```

Connect the phone using one of the printed PC URLs. Click the colored object in the preview. The controller approaches the color, settles, then searches for the configured tag and approaches it. Esc/Q stops and exits. `--tag-only` starts directly with tag homing. `--camera-only` never connects to the board and shows the navigation decisions without commanding hardware.

| Setting near the top of the script | Meaning |
|---|---|
| `NAVIGATION.acquire_frames`, `acquire_seconds` | Confirmation required before approach/reacquisition |
| `grace_seconds`, `search_after` | Bounded loss bridge and delay before searching |
| `search_turn`, `search_pause` | Turn-command duration and full stationary pause after ramp-down |
| `base_speed`, `max_turn`, `steering_gain` | Requested movement and steering limits |
| `filter_seconds`, `max_prediction_shift` | Smoothing time constant and prediction bound |
| `TAG_STOP_FRACTION`, `COLOR_STOP_FRACTION` | Apparent tag side or color radius divided by image width |
| `TAG_MIN_DECISION_MARGIN`, `TAG_MAX_HAMMING` | AprilTag 3 decoding acceptance thresholds |
| `PWM_ACCELERATION`, `PWM_DECELERATION`, `MIN_PWM` | Motor response; increasing smoothness increases stopping distance |

The 1.1-second search command includes acceleration. If shortened too far, PWM may never reach the level needed to start the motors. `search_pause` begins after commanded PWM is zero; allow additional settling time if the camera continues shaking. Decision margin 20 and Hamming 0 are trial thresholds, not universal confidence levels. OpenCV does not supply equivalent margin/Hamming diagnostics here, so those CSV fields are empty for that backend.

## Record and replay the same evidence

```powershell
.\.venv\Scripts\python.exe -m demos.apriltag.robust --tag-only --record moving-test.avi --log-csv moving-test.csv
.\.venv\Scripts\python.exe -m demos.apriltag.robust --video moving-test.avi --tag-only --log-csv replay-test.csv
.\.venv\Scripts\python.exe -m demos.apriltag.robust --video moving-test.avi --tag-only --backend opencv --headless
```

Replay always disables the Arduino connection, even without `--camera-only`. Output filenames must be new; existing files are not overwritten. Video contains unannotated **processed** frames at a nominal 20 FPS, so it is useful for comparing detector settings but does not preserve variable real-time frame intervals. The companion CSV preserves local receive/result times and reports processing duration, a Laplacian sharpness diagnostic, detected/rejected IDs, decision margin/Hamming when available, navigation phase, filtered position and requested wheel speeds. Phase/speed fields sample the latest control output at result delivery.

PhoneSense's current in-process JPEG API does not expose the phone capture time. `receive_to_result_ms` excludes phone exposure, encoding, and network transport; it is not end-to-end latency. Sharpness is diagnostic only: no arbitrary sharpness threshold rejects otherwise valid detections.

Compare recordings with the robot stationary and moving at several low speeds, using the same tag and lighting. Tune detector quality thresholds from these recordings, then tune steering and search timing. Start physical changes with wheels raised, followed by slow floor tests with clearance for stopping distance. Pose accuracy, real blur tolerance, balance and physical stopping distance remain unverified until tested with the actual robot.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_robust_april_tag -v
```

Checks cover acquisition/reacquisition, wrong IDs, inconsistent candidates, filtering/outliers, bounded prediction, close-range loss, stable arrival, stale input, and timed search pauses. Generated tags are decoded by both actual detector backends. Recorded synthetic clips exercise the full color-to-tag sequence, missing frames, wrong IDs, CSV output and video recording, with an assertion preventing Arduino connection.
