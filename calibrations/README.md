# Robot body profiles

This folder holds lightweight JSON models of your robot's response to motor pulses.
They sit between a confirmed vision observation and Telemetrix motor execution.
They are separate from YOLO detection weights and require no neural-network training.
Personal JSON profiles are ignored by Git; use a distinct filename for each robot
or changed setup. The webapp offers existing profiles in the body-model input.

## Learn a body response

1. Restart `python -m webapp` after updating the code. Open **Missions**.
2. Verify the Arduino IP and the two `(IN1, IN2, PWM)` pin groups. Software cannot
   discover which physical pins you wired. You do not need to identify motor polarity.
3. Attach the phone camera rigidly to the robot. Place a **stationary** tag36h11
   marker in view, reasonably centered, with detected edge width 8–50% of the image.
   First use **Continuous detector preview** in a tag-only mission to check detection.
4. In **Robot body response**, choose a new `body_model` path, such as
   `calibrations/my-robot-v1.json`. Keep `body_control = off` while calibrating.
5. Select **Learn robot body using a fixed AprilTag**. This selects the tag-only
   behavior. Choose its tag ID, use live PhoneSense rather than replay, and explicitly
   enable physical driving. Use clear floor space and keep Stop within reach.
6. Start the mission and connect the phone. Calibration waits for a confirmed tag,
   then performs **eight single-motor probes**: positive/negative commands to each
   motor, repeated twice. Each probe ramps, stops, settles and confirms new frames.
7. Inspect the terminal and progress on Mission monitor. Losing the tag, excessive
   displacement, inconsistent response, a stalled motor, camera loss or a stop request
   aborts. Calibration never searches blindly or saves a partial fitted model.

`body_probe_speed` defaults to 0.12 motor demand; `body_probe_seconds` defaults to
0.06 seconds of hold. **Ramps add motion time.** These are bounded starting values,
not a guarantee of a safe displacement on unknown hardware. Minimum PWM must already
be sufficient to move the robot. If a probe produces no reliable movement, calibration
fails instead of increasing motor power automatically. It never overwrites a profile.

The fitted model is four numbers: a local 2×2 image-response matrix. Input is signed
requested PWM-seconds per motor, measured across ramps/hold/braking; output is change
in horizontal target position and log apparent linear size. Reversed motor polarity
changes a learned column's signs, and unequal response changes its magnitude.
Fit checks reject insufficient excitation, weak responses, singular matrices and
large residual errors. This learns a local response, not a complete physical body.

## Use it for a mission

Select `body_control = learned` and the saved `body_model`, then choose any of the
existing object/tag mission variants. Preview first. The middleware predicts a small
set of bounded motor pulses, including independent left/right hold times, and chooses
one toward the fresh target. A faster wheel can brake earlier. This compensates for
observed polarity and relative response without changing detection models or pin wiring.
It refuses a straight approach with excessive predicted yaw or no predicted progress.

All moves still finish at commanded zero PWM before another stationary decision.
Learned commands may have zero hold while still moving during ramps. The configured
maximum demand/hold remains a limit, and demand cannot exceed the calibration range.
Very unequal motors, minimum-PWM constraints or nonlinear response may leave no valid
pulse; control stops with an error instead of inventing a correction.

The profile records pins, minimum PWM and ramp rates. Loading it with different
values is rejected before hardware connection. Recalibrate after moving the camera,
swapping wheels/motors, changing load, floor, battery conditions or significant target
distance; these conditions are not automatically identifiable from the profile file.
The calibration tag's physical size is not needed, and the model does not report meters.
An object's box area is converted to a linear-size proxy before using the model.

`body_learning = frozen` is the default. `tag_online` allows small recursive
least-squares updates from consecutive fresh, confirmed before/after sightings of
the **same fixed AprilTag**. It rejects large prediction errors and caps updates;
camera gaps discard pending pairs. Object detections never train the model online,
because motion of the tracked object could be mistaken for robot response. Online
updates are in memory for that run; the original profile stays unchanged.

## What still needs hardware or further work

This does not learn safe motor power without test motions, measure true wheel speed,
infer pin wiring, estimate full inertia, detect obstacles or guarantee balance.
Wheel encoders/IMU feedback are needed for measured-speed PID, better yaw/distance
prediction and transient dynamics. Network and image receipt times are not exposure
timestamps. Keep the marker fixed throughout calibration and online adaptation;
camera-only observations cannot distinguish robot movement from target movement.
Stop/connection checks on the PC do not replace a firmware motor-command watchdog.

## Existing methods and implementation

Visual-response identification follows the image-Jacobian idea used in
[uncalibrated visual servoing research](https://www.cs.columbia.edu/robotics/projects/visual_control/bill-peginhole.html).
[ViSP](https://github.com/lagadic/visp) and
[Machine Vision Toolbox](https://petercorke.github.io/machinevision-toolbox-python/stubs/camera/CentralCamera/machinevisiontoolbox.Camera.CentralCamera.visjac_p.html)
provide broader visual-servoing tools; integrating them still requires a robot-specific
motion interface and observations. The optional online update uses the standard
[recursive least-squares estimation formulation](https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html).
The implementation uses the project's existing NumPy/OpenCV stack, with no new model,
cloud service or heavyweight dependency.

Source: [body_model.py](../robotics_lab/body_model.py),
[calibration.py](../missions/stop_and_stare/calibration.py), and
[motion.py](../missions/stop_and_stare/motion.py).
Hardware-free tests identify a simulated reversed/unequal robot, verify independent
wheel braking and motor integrals, reject bad fits, and stop on lost calibration tags.
No physical body calibration has been performed by these tests.
