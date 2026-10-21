# Perception, steering and position memory

These components retain stop-and-stare: stop, settle, flush delayed frames, confirm
fresh stationary detections, plan one bounded pulse, ramp to zero, repeat. Defaults
remain YOLO, timed steering, raw memory and ramped motor execution. No new packages
are required. The optional controllers need physical calibration before driving.

## OpenCV color missions

1. Run `python -m webapp` from Main and open **Missions**.
2. Choose **Object approach** or **Object → AprilTag** and set **object provider**
   to `hsv`. YOLO weights, confidence and association settings are hidden.
3. Give **target class** a descriptive name, such as `red ball`. With HSV this is
   a log label, not a trained class. Set the tag ID if using the second leg.
4. Leave **color h** at `-1` for interactive selection. Start with driving unchecked.
5. Connect the phone using the monitor's PhoneSense link and start its camera.
   `WAIT_COLOR` issues no movement until a color is set.
6. Choose **Freeze frame & pick color**, or click the live camera to freeze a frame.
   Click a well-lit patch inside the target in that frozen image. Avoid highlights,
   edges and shadows. The exact displayed JPEG is sampled even if new video arrives.
7. Inspect the annotated stationary image. Tune tolerances in a new mission/profile
   if necessary. Pick again during the object leg to change color. Changes apply
   while stopped and invalidate an observation burst using the previous color.

The picker samples a small neighborhood, averaging hue circularly and taking median
saturation/brightness. Pinned images expire after two minutes; freeze again if
needed. **Active HSV** acknowledges the worker's applied selection. Selections also
appear in runtime telemetry and terminal output; they do not overwrite saved
profiles. Copy H/S/V into a profile to reuse them. An already-running pulse finishes
before a queued selection applies; use **Stop job** to stop the mission immediately.

OpenCV uses H=0..179 and S/V=0..255. Hue tolerance wraps around red at 0/179.
The provider applies saturation/value tolerances, morphological opening and a
minimum contour-area fraction. It selects the largest blob initially, then prefers
the remembered horizontal position. Identical same-color objects are ambiguous.

Approach size is the blob's **bounding-box area / image area**, matching YOLO's
`object_stop_area`. `color_min_area` filters small contour speckles; it is not the
arrival threshold. AprilTag perception is unchanged. HSV loads no neural weights.

For CLI use, save a JSON config with an explicit color (the interactive picker is
in the webapp):

```json
{
  "mission_mode": "object_then_tag",
  "object_provider": "hsv",
  "target_class": "red ball",
  "color_h": 0,
  "color_s": 220,
  "color_v": 220,
  "tag_id": 0
}
```

```powershell
python -m missions.stop_and_stare.run --config color-mission.json
```

## Optional visual PID steering

Set `steering_controller = pid`. Horizontal error is -1 at the left edge, 0 in the
center, +1 at the right edge. PID uses actual intervals between confirmed camera
receipt measurements rather than assuming fixed FPS.

| Setting | Units / purpose |
|---|---|
| `pid_kp` | seconds of hold per unit image error |
| `pid_ki` | integral contribution to hold; start at zero |
| `pid_kd` | derivative contribution using a smoothed error derivative |
| `pid_integral_limit` | absolute accumulated error-seconds limit |
| `turn_min_seconds`, `turn_max_seconds` | lower/upper turn-hold bounds |
| `turn_speed` | motor demand during alignment turns |

Integral accumulation is suppressed when it pushes saturated output further.
History resets on sign reversal, target loss, centering, long gaps, camera
reconnection, new color selection and mission-leg change. Direction follows fresh
image evidence; integral/derivative terms cannot turn against it. PID never exceeds
the configured maximum hold. Ramp-up/down still add movement time outside the hold.

Start with Kp only and Ki=0, then calibrate short physical turns. Add a little Kd
only if repeatable overshoot persists. Change one value at a time. The supplied
gains are starting points, not values validated on your robot. This is visual
steering; wheel-speed PID needs encoders and a separate firmware/feedback loop.
Forward movement remains a size-based timed pulse.

## Optional Kalman position memory

Set `position_estimator = kalman`. State is horizontal position and velocity, with
a variable-time constant-velocity transition and covariance. Confirmed bursts
correct it using the latest contributing packet's local receipt timestamp. A
Joseph covariance update preserves numerical stability. Arrival uses original
conservative confirmed size; only horizontal position is filtered.

`filter_process_noise` allows unmodeled image motion; `filter_measurement_noise`
sets trust in positions. `filter_motion_noise` adds uncertainty per second of
completed robot movement, including ramps. Each pulse resets learned image velocity
so robot rotation is not mistaken for continuing target motion. The filter does
not yet map motor commands to calibrated camera displacement.

Predictions guide selection and search only within `position_memory_seconds` and
below `filter_max_uncertainty` standard deviation. The last confirmed side remains
a fallback within its existing memory interval. Camera gaps reset estimation and
require a fresh stationary burst. Missing detections never become synthetic
confirmed targets: no predicted forward approach or arrival. Fresh off-center
evidence cannot be hidden by filter lag.

This is mathematical estimation, not machine learning. Learnable motion prediction,
capture timestamps and encoder/IMU fusion remain in the root README TODO.

## Component contracts

Factories in [components.py](components.py) are trusted code loaded by both server
and worker. Config validates names against these registries and the UI renders
their options automatically. Browsers cannot submit import paths or Python code.

| Registry | Required interface |
|---|---|
| `OBJECT_PROVIDERS` | Construct with Config; `ready` bool property, `warmup()`, `begin_burst(preferred_x=None)`, `detect(bgr_frame)` returning `(Sighting or None, annotated_frame)` |
| `STEERING` | Construct with Config; `reset()`, `duration(error, receipt_time)` returning bounded positive hold seconds |
| `ESTIMATORS` | Construct with Config; `reset()`, `update(x, receipt_time)` returning filtered x, `predict(now)` returning x or None, `moved(step, elapsed, now)` |
| `MOTION` | Construct with Config and `drive=False`; async `open()`, `pulse(step, hook)`, `stop()`, `close()`; expose `drive` and `control` (None or an object with `is_stopped`) |

`Perception` decodes JPEGs and delegates the object leg; AprilTags stay separate.
`Runner` also accepts source, perception and driver objects directly for tests.
Keep factories lazy: menus must not load models, open cameras or connect hardware.

Motion plugins must poll the async hook frequently, request zero PWM in `finally`
on exceptions, and return from `pulse` only after requesting zero. Preserve stop
markers, camera-health checks and stationary observation. Add hardware-free adapter
tests before enabling driving. Browser values must never become shell commands or
arbitrary tracker YAML paths.

New Config fields need validation and descriptions in `webapp/help.py`; they then
appear under additional settings. Providers needing different inputs also need
matching Jobs validation/form metadata. Registries do not make arbitrary artifacts
compatible automatically. See [the extension guide](../../webapp/EXTENDING.md).

## References and checks

The provider follows [OpenCV's HSV thresholding approach](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html),
with explicit circular hue handling. Prediction/correction use standard
state/covariance equations documented by [OpenCV's Kalman reference](https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html).
Integral limits and resets follow robot-controller practice described by
[WPILib's PID documentation](https://docs.wpilib.org/en/stable/docs/software/advanced-controls/controllers/pidcontroller.html).
Our variable-interval controller does not use WPILib's fixed-period class.

`tests/test_mission_components.py` covers real color-to-AprilTag perception, frozen
frame sampling despite newer video, validation, bounded storage, no movement before
selection, color-change burst invalidation, PID limits, Kalman uncertainty and
component substitution. Tests connect no physical motors. Physical stopping,
gain/noise calibration and balance require robot trials.
