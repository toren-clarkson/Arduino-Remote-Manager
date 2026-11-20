# Adaptive Red-Ball -> AprilTag homing

Run from the Main project using its virtual environment:

```powershell
.\.venv\Scripts\python.exe run.py
```

Choose **Adaptive Red-Ball -> AprilTag**. Alternatively run
`python adaptive-homing.py` directly; with no arguments it opens the same setup.
The mission remains one root Python program. There is no dashboard or server
between the camera and the controller besides PhoneSense itself.

## OpenCV HSV color instead of YOLO

Choose **9. Adaptive OpenCV HSV color -> AprilTag** in `python -m run`, or:

```powershell
python adaptive-homing.py --color --setup
```

This uses the same adaptive body model, prediction, motor ramps and AprilTag
homing, with an OpenCV HSV mask replacing YOLO/ByteTrack for the object stage.
No YOLO weights are read, no YOLO warmup occurs, and no neural network is loaded.
The color setup skips all YOLO weights/class/tracker questions.

1. Choose **Camera preview**. Connect/start the phone camera as usual. Click
   directly on the ball/object in **Adaptive robot - live**, not the detector
   result window. The console prints the selected HSV and the detector draws
   its matching region. Click an interior colored pixel, away from highlights,
   shadows and the status text. Esc/Q exits.
2. Run again and choose **Learn robot/surface profile**. Prefer a name such as
   `robot1-tile-color`. Click the stationary ball to choose HSV, then press Enter
   at the calibration prompt. Until selection and the prompt finish, motors
   remain stopped. Complete the wheel probes and the AprilTag calibration.
3. Run again and choose **Drive** with that profile. Its saved HSV is restored
   automatically. It approaches the color continuously, then homes to the tag.

The HSV is locked for each run so a stray click cannot change the goal during
calibration or driving. Rerun Preview to try another color; rerun Learn to save
another color/target definition. An existing YOLO object model cannot be reused
as an HSV object calibration. Selecting a different HSV or tolerance combination
requires matching calibration before driving. The tag model can remain valid
when the tag and robot/camera setup are the same.

Setup offers hue, saturation and brightness tolerances, defaulting to 10, 80 and
80 respectively in OpenCV's HSV units (H 0..179; S/V 0..255). Red wraps correctly
across H=179 and H=0. A small morphological opening/closing suppresses isolated
mask noise; contours smaller than 100 pixels are rejected. The largest matching
region is selected, so keep other objects of the same color out of the scene.
Color segmentation does not distinguish two same-colored objects by identity.

Arrival remains **bounding-box area / image area** (`0.12` by default), now for
the color contour's box. It is not the circle radius used by some older demos.
The adaptive feature uses its square root, just as with YOLO box extent.
The HSV and segmentation tolerances are part of the profile target signature;
color and YOLO console preferences are stored separately.

Optional direct commands (headless replay requires an explicit or saved HSV):

```powershell
python adaptive-homing.py --color --mode preview
python adaptive-homing.py --color --mode calibrate --profile robot1-tile-color
python adaptive-homing.py --color --mode drive --profile robot1-tile-color
python adaptive-homing.py --color --mode preview --hsv 0 255 255 --video path\to\clip.avi --headless
```

## First run

1. Upload the [AP firmware](../Telemetrix4UnoR4WIFI/README.md) and connect both phone
   and PC to `UNO-R4-Robot`, password `Robot295!`. Keep the no-Internet connection.
2. Choose **Camera preview** first. Choose local YOLO weights (normally
   `models/red-ball-v1.pt`) and the actual class inside them (`Red-Ball`), then the
   printed AprilTag 36h11 ID (normally `0`). A YOLO class and an AprilTag ID are
   separate settings. No training video or new neural-network training is needed.
3. Open the **PC** HTTPS PhoneSense URL printed in the terminal on the phone;
   allow the camera and choose Connect/Start. Fix the phone rigidly to the robot.
4. Run again, choose **Learn robot/surface profile**, and create a profile such as
   `robot1-tile` or `robot1-carpet`. Enter the motor pin triples and a conservative
   maximum PWM. The provided starting pins are left `7,4,5`, right `9,8,6`.
   They mean IN1,IN2,enable, and the two enable pins must support PWM.
5. The console prompts for the ball first. Place a **stationary** ball in view,
   roughly centered and small enough to leave room for turning and approach.
   Put the robot on the actual surface, with clear space around it. Do not hold
   the target in your hand or reposition it between probes. Press Enter to begin.
6. The calibration tries increasing PWM levels on each wheel and reverses each
   probe. It measures before/after images **while stationary**, identifies both
   wheel directions and response gains, and records the first level with visible
   movement. Keep the target in view throughout. A lost target, insufficient
   movement, inconsistent response or disconnected camera aborts calibration.
7. Repeat for the stationary printed AprilTag when prompted. The ball and tag
   have separate local models; their size measurements are different.
8. Run again and choose **Drive using a learned profile**, select that profile
   and the same detector/class/tag. The robot searches for and approaches the
   ball, stops briefly at the stage transition, then searches for and approaches
   the tag. Esc/Q in a camera window or Ctrl+C stops.

Calibration moves the wheels in **both** directions. The maximum PWM and probe
duration are physical movement limits, not a learned guarantee of clearance.
The probe duration includes a hold plus the time allowed for acceleration; it
can take longer than the displayed hold alone. The calibration stops PWM
immediately after each probe, then waits for settling before measurements.

## Settings and profiles

The terminal asks for a robot/surface profile, motor pins, PWM limit, calibration
pulse and settling times, acceleration/deceleration, estimated camera delay,
weights, target class, AprilTag ID, arrival sizes, inference size/confidence and
ByteTrack / BoT-SORT / detector-only tracking. Enter keeps the displayed value.

The ball arrival value is **bounding-box area / image area** (default `0.12`).
The tag arrival value is **mean tag edge / image width** (default `0.24`).
These are apparent image sizes, not physical distances. Three distinct fresh,
centered observations must meet the chosen threshold to complete a stage.

Body profiles are small human-readable JSON files under
`calibrations/adaptive/<name>.json`, including:

- Robot/surface/camera notes, pins, maximum PWM and ramp settings.
- First observed moving PWM for each wheel (a coarse friction floor).
- Separate 2x2 response matrices and RLS estimator state for object and tag.
- YOLO weight checksum/class and tag-family/ID signatures, preventing accidental
  use of calibration from a different detector or target ID.
- In color mode, the selected HSV and an HSV/tolerance target signature instead
  of a YOLO weight signature.

Create a different profile when changing the robot geometry, wheel placement,
camera mounting/orientation/zoom, target physical size, load, or surface. Editing
pins or robot/surface notes in setup invalidates the old calibration. Other
physical changes cannot be detected automatically; select/create the appropriate
profile yourself. Weight/class/tag changes require the matching leg to be
recalibrated. The online estimator can accommodate modest response changes; a
large change, heavy slip or stall needs fresh calibration.

Successful calibration saves each completed leg. Online model updates are saved
on successful mission completion and an ordinary Ctrl+C/window quit; a fault
does not overwrite the saved profile. Keep a copy of a known-good JSON if you
want to compare sessions. Historical body-model JSON files from the removed
webapp are incompatible and are not loaded.

Console preferences are stored separately under `calibrations/console/`.
Other root programs support `--setup`, or choose them through `run.py`, to set
their applicable models/classes/tag IDs, motor pins and movement settings.
They keep their own existing control logic; learned profiles are used by the
adaptive mission. CLI scripts remain available without interactive setup.

## What the model learns

```text
PhoneSense latest JPEG -> local YOLO + ByteTrack / OpenCV AprilTag
                      -> target x and log(apparent size)
                      -> learned image-response model + bounded prediction
                      -> damped inverse controller -> PWM ramps -> Telemetrix
                                              ^                |
                                              |___ feedback ___|
```

The feature vector is `[horizontal offset, log(target extent)]`. The offset is
on a -1..1 scale. YOLO extent is the square root of box area fraction; tag extent
is its edge fraction. The model estimates:

```text
change in image features ~= J * integral([signed left PWM, signed right PWM]/255)
```

Calibration fits J with independent wheel probes. While approaching, recursive
least squares (RLS, forgetting factor 0.995) slowly updates J using consecutive
observations of the same tracked target and the **applied host PWM**, including
ramps. The damped inverse maps desired centering/approach into wheel demands.
It can discover reversed wiring and unequal wheel effects without direction
checkboxes. It does not know robot position in meters, wheel speed, yaw angle,
obstacles, or the motor's electrical/mechanical state.

The controller reduces approach when off-center and estimates target motion
between observations from the recent command history. Prediction is capped at
0.15 per feature and can bridge at most 0.40 seconds since the last real sighting.
It fades after 0.25 seconds or a missed detection. Predictions cannot confirm
arrival or train the body model. Outliers, identity changes and long observation
gaps are gated out of learning. There is no exploration/probing during a mission;
learning from ordinary motion needs enough varied commands to identify changes.

When the prediction expires the robot requests zero PWM through its normal ramp
and waits briefly for reacquisition. After 0.9 seconds without a sighting, it
searches toward the last-seen side with short turn/pause periods. The learned
matrix generates the turn; a nominal "left wheel forward" convention is not
assumed. These pauses are fallback search behavior; fresh tracking and homing
run **continuously**, processing moving-camera frames.

## Latency and limits

- One inference task processes the latest available image. New arrivals replace
  pending camera data; there is no growing inference queue. Native live preview
  and motor updates continue while YOLO runs in a worker thread.
- Source timestamps are **PC receipt times**, not phone exposure timestamps.
  The profile's camera-delay estimate shifts the command history used for
  learning. It is a bounded approximation, not an automatic latency measurement.
- Detections older than 0.55 seconds cannot authorize tracking or arrival.
  Camera interruption/freshness failure after 0.75 seconds stops the mission.
  Lower inference image size or use smaller local weights if your laptop cannot
  provide sufficiently fresh results. Roboflow is used for dataset/training
  exports; no cloud inference or Internet is needed during this mission.
- A learned model cannot recover a tag from severe blur indefinitely. Improve
  lighting/exposure, reduce maximum PWM, or return to stop-and-stare if the
  detector cannot provide fresh evidence. A motor whose minimum moving PWM is
  too fast limits how smoothly any continuous controller can drive it.
- This is an experimental local image-response controller, not a pretrained
  navigation policy, balance controller, physical dynamics model, obstacle
  avoidance system, or wheel-speed PID. No encoders/IMU are available here.
  The firmware still needs a heartbeat watchdog for a stalled PC before TCP
  detects disconnection. Host-side timeout handling requires the process to run.

Each run writes `runs/adaptive/<timestamp>.jsonl`: observations, source ages,
stage, requested/applied wheel effort, accepted RLS sample count and calibration
probes. The terminal prints one status update per second. Replay is always
preview and cannot open a physical motor connection:

```powershell
python adaptive-homing.py --mode preview --video path\to\clip.avi --headless
python adaptive-homing.py --mode preview --tag-only --tag-id 0
python adaptive-homing.py --mode calibrate --profile robot1-tile
python adaptive-homing.py --mode drive --profile robot1-tile --weights models/red-ball-v1.pt --target-class Red-Ball --tag-id 0
```

## Research and implementation choice

Image-based visual servoing is an established method of deriving robot movement
from visual feature error; see Hutchinson, Hager and Corke's
[visual servo control tutorial](https://faculty.cc.gatech.edu/~seth/ResPages/pdfs/HutHagCor96.pdf).
[ViSP](https://github.com/lagadic/visp) is an existing mature visual-servoing
library. Its platform provides control-law and tracking tools, rather than
weights already calibrated to this custom PWM/Telemetrix robot. For this small
Windows Python project I chose a NumPy implementation of the local model and
controller rather than adding that platform and a custom robot adapter.

RLS is an established, inexpensive online identification algorithm, not a new
neural-network architecture; the implementation follows the standard forgetting
factor recursion in [Linkoping University's adaptive signal processing lecture](https://isy.gitlab-pages.liu.se/rt/assets/TSRT78/le11.pdf).
Parameter identification needs informative inputs; convergence claims for ideal
regression do not establish stability of this noisy, delayed physical robot.
The code's gates/caps and local model are engineering choices, not a reproduction
of a published robot controller or a claim of globally stable navigation.

YOLO tracking uses the existing Ultralytics
[ByteTrack / BoT-SORT API](https://docs.ultralytics.com/modes/track/), maintaining
IDs with `persist=True`. The tracker maintains object identity; the separate
adaptive model learns the robot's visual response. No extra Python dependency
is introduced. Hardware-free tests cover identification, reversed/unequal wheels,
RLS gain changes, stale observations, dropouts, arrival, real tag detection and
a complete asynchronous ball -> tag mission. Physical calibration and real
Wi-Fi/camera validation must still be done on your robot.
