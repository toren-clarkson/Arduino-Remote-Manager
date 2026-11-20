# VIP 295 robot vision project

Run one Python file directly from this folder. Settings live at the top of each
script; PhoneSense prints its phone-camera URLs in that script's terminal.
The dashboard, job runner, browser streaming and plugin/control framework have
been removed. Your datasets, weights, printed tags and recorded runs are retained.

## Console setup and adaptive Red-Ball -> AprilTag

Run the simple terminal menu:

```powershell
.\.venv\Scripts\python.exe run.py
```

Choose a program, then select its applicable weights, object class, AprilTag ID,
motor pins and initial movement settings in the console. Enter keeps the shown
value. Settings are remembered for the next setup. Root scripts also accept
`--setup`; their existing direct commands continue to work.

The new [adaptive-homing.py](adaptive-homing.py) learns a small robot/surface
response model between detection and Telemetrix. It follows the selected YOLO
object continuously, then homes to the selected AprilTag. Start with **Camera
preview**, then **Learn robot/surface profile**, then **Drive using a learned
profile**. Calibration probes each wheel in both directions and measures the
effect on stationary target images; no neural-network training is required.
Choose separate profiles for different bodies, surfaces and camera mounts.

Fresh observations update the controller during movement. Brief blur is bridged
with bounded prediction; prolonged loss stops approach and triggers remembered-side
search. The stop-and-stare mission is still available. See the
[adaptive setup, calibration, algorithm and research guide](docs/ADAPTIVE_HOMING.md)
before calibrating or driving. Learned body profiles apply to the adaptive mission;
the older programs retain their own control behavior.

For adaptive homing using **OpenCV color tracking instead of YOLO**, choose
**9. Adaptive OpenCV HSV color -> AprilTag** in `python -m run`, or run:

```powershell
.\.venv\Scripts\python.exe adaptive-homing.py --color --setup
```

Choose Preview or Learn profile, connect the phone, and **click the ball in the
"Adaptive robot - live" window** to select its HSV. During calibration the robot
stays stopped until selection and the console prompt are complete. The selected
HSV is saved in the learned profile; Drive reuses it automatically. Use a new
color calibration when switching from YOLO. This option loads no YOLO model.
See the [HSV workflow](docs/ADAPTIVE_HOMING.md#opencv-hsv-color-instead-of-yolo).

## Robot Wi-Fi: connect directly to the UNO

First upload the updated [Telemetrix AP sketch](Telemetrix4UnoR4WIFI/Telemetrix4UnoR4WIFI.ino).
It creates **UNO-R4-Robot**, password **Robot295!**, with fixed Arduino address
**192.168.4.1** and Telemetrix TCP port **31336**. Connect both PC and phone to that
Wi-Fi and choose to stay connected without Internet. Serial Monitor is optional.

All Python clients now use that fixed address automatically. The phone-camera URL
is different: PhoneSense runs on the **PC's** DHCP address on the AP, which the scripts
select and print automatically. Open that printed HTTPS URL on the phone, Connect,
then Start camera. Do not open `192.168.4.1` as a camera page.
See the [network setup and troubleshooting guide](Telemetrix4UnoR4WIFI/README.md).

## Red-Ball -> AprilTag: stop and stare

[stop-and-stare.py](stop-and-stare.py) defaults to `models/red-ball-v1.pt`, class
`Red-Ball`, then tag36h11 ID `0`. First join the robot AP and check motor pins/PWM
settings at the top of the file. Preview does not connect to the Arduino:

```powershell
.\.venv\Scripts\python.exe stop-and-stare.py
```

For physical driving on the robot AP (no IP input needed):

```powershell
.\.venv\Scripts\python.exe stop-and-stare.py --drive
```

1. The script loads YOLO, starts PhoneSense and prints its HTTPS phone URLs.
2. Open the PC address reachable from your phone. Accept the local certificate
   prompt, allow camera access, then choose **Connect** and **Start camera**.
3. The robot stays stopped, waits `SETTLE_SECONDS`, discards buffered arrivals,
   and confirms the target across distinct new stationary frames.
4. It stores the confirmed position, size and a corresponding stationary image.
   An off-center ball triggers one turn; a centered, small ball triggers one
   forward pulse. Then it requests zero PWM and repeats the stationary cycle.
5. Once the ball is centered and its bounding box meets `OBJECT_STOP_AREA`, it
   switches to the selected AprilTag and approaches using `TAG_STOP_WIDTH`.
6. When the tag is confirmed centered and close enough, it stops and exits.

**No detection or target-memory updates happen during movement, braking or
settling.** The live camera window can still display those frames. The second
window shows the last stationary detection and stays unchanged during movement.
After stopping, the script gates out old frames before observing again.

One missed burst causes another stationary look. Two missed bursts permit one
search-turn pulse toward the last-seen side; the robot stops and looks again.
Memory expires after `TARGET_MEMORY_SECONDS`, then search defaults right. Stored
memory never authorizes blind forward driving or declares arrival without fresh
confirmation. Camera interruption, a detection processing timeout or a ramp timeout
stops the run. Because the robot stays stopped throughout detection, processing
that takes over one second produces a warning and can still use that settled
image. The independent camera watchdog stays active; frozen/disconnected video
still stops the mission. Each detection has an eight-second processing limit.

YOLO defaults to a 320-pixel inference size for this mission. Console setup now
offers image size, confidence and the stationary inference timeout. For a direct
run these can be set with `--image-size 320 --inference-timeout 8`. Slow inference
adds stationary waiting time rather than allowing motion during detection.

### Tune the movement in the same file

| Setting | Meaning |
|---|---|
| `LEFT_MOTOR`, `RIGHT_MOTOR` | `(IN1, IN2, PWM)` wiring; the fixed AP endpoint is shared in `robotics_lab/network.py` |
| `MIN_PWM` | Motor friction floor; initial value matches the robust AprilTag script |
| `FORWARD_SPEED`, `TURN_SPEED` | Normalized motor demands, not measured velocity |
| `FORWARD_HOLD`, `TURN_HOLD`, `SEARCH_HOLD` | Requested hold seconds per movement pulse |
| `ACCELERATION`, `DECELERATION` | PWM ramp rates; ramps add time to every movement |
| `SETTLE_SECONDS` | Stationary wait after commanded PWM reaches zero |
| `DISCARD_FRAMES` | Extra fresh arrivals discarded after settling |
| `OBSERVE_FRAMES`, `CONFIRMATIONS` | Stationary detections required for one decision |
| `OBJECT_STOP_AREA` | Ball box area / image area; 0.12 means 12% |
| `TAG_STOP_WIDTH` | Mean tag edge / image width; 0.24 means 24% |
| `CENTER_TOLERANCE` | Horizontal alignment tolerance on a -1..1 scale |
| `IMAGE_SIZE`, `CONFIDENCE`, `DEVICE` | YOLO inference size, confidence and processing device |
| `INFERENCE_TIMEOUT` | Maximum processing seconds per stationary image; camera freshness is checked separately |

Hold times exclude ramps. Total commanded movement can be much longer than the
hold; calibrate short pulses on your robot before unattended operation. Increasing
an arrival threshold approaches closer. These image sizes are not meters.

Other ways to run the same file:

```powershell
# Only approach AprilTag 0, without YOLO:
.\.venv\Scripts\python.exe stop-and-stare.py --tag-only

# Click an OpenCV color instead of using YOLO, then approach tag 0:
.\.venv\Scripts\python.exe stop-and-stare.py --color

# Different local detection weights/class:
.\.venv\Scripts\python.exe stop-and-stare.py --weights models/my_object.pt --target-class my_object

# Offline detector/decision replay, never connects the phone or Arduino:
.\.venv\Scripts\python.exe stop-and-stare.py --video path\to\clip.avi --headless --max-steps 5
```

Esc/Q or Ctrl+C exits and requests zero PWM. The script uses local frame-receipt
times, not unavailable phone-exposure timestamps. Flush/settle reduce buffered
motion images but cannot prove when a frame was captured. Timed motion has no
wheel-encoder feedback or firmware command watchdog yet.

## Other standalone programs

| File | Use |
|---|---|
| [robust-april-tag.py](robust-april-tag.py) | Existing filtered color -> AprilTag mission; retained behavior and settings |
| [simple-april-tag.py](simple-april-tag.py) | Earlier color -> tag approach with short-term memory and intermittent search |
| [default-opencv-tracking.py](default-opencv-tracking.py) | Click-to-follow OpenCV color tracker |
| [default-yolo-tracking.py](default-yolo-tracking.py) | Local YOLO object following |
| [dual_motor_template.py](dual_motor_template.py) | Direct motor/camera experiment template |
| [test_wifi_led.py](test_wifi_led.py) | Blink pin 13 over Telemetrix Wi-Fi; no motors |

Start with the native AprilTag preview that worked well for you:

```powershell
.\.venv\Scripts\python.exe robust-april-tag.py --camera-only --tag-only --backend opencv
```

For YOLO-only preview:

```powershell
.\.venv\Scripts\python.exe default-yolo-tracking.py --camera-only --model models/red-ball-v1.pt
```

The older color/simple/motor scripts retain their direct hardware defaults. Read
their pin settings before running; motor directions differ. All share the same AP address.
The stop-and-stare mission defaults to preview without `--drive`; adaptive homing
also defaults to preview and requires a calibrated profile for Drive mode.
`robust-april-tag.py` and `default-yolo-tracking.py` provide `--camera-only`.
None of the scripts opens hardware just because it is imported.

## Install and camera connection

Use this project's `.venv\Scripts\python.exe` in VS Code:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PhoneSense provides the phone-camera page on the PC's HTTPS port 8080 and starts
directly in the script. The network helper automatically advertises the PC address
on the robot AP, even with other Ethernet/VPN interfaces. Optional `PHONE_IP` /
`--phone-ip` overrides are unnecessary for normal use. The Arduino is always
`192.168.4.1:31336` after uploading this AP sketch; the PC/phone have separate DHCP
addresses. Run only one robot/camera script at a time and keep the terminal running.

## Supporting folders

- [training](training/README.md): frame extraction, Roboflow dataset download and YOLO training.
- [models](models/README.md), [datasets](datasets/README.md): your existing weights and data.
- [tools](tools/README.md), [assets/tags](assets/tags/README.md): printable AprilTags.
- [robotics_lab](robotics_lab/README.md): small shared PWM/navigation/path helpers.
- [Telemetrix4UnoR4WIFI](Telemetrix4UnoR4WIFI/README.md): Arduino AP firmware and Wi-Fi setup.
- [APRILTAG_DESIGN.md](APRILTAG_DESIGN.md): earlier robust tracker design and research.
- [tests](tests/README.md): hardware-free checks.
- [runs](runs/README.md): previous recordings/results and archived dashboard records.
- [calibrations](calibrations/README.md): adaptive robot/surface profiles, console preferences and historical data.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Tests use simulated boards or offline images/video, never physical motors.

## TODO

- [x] Add a standalone adaptive visual-servo mission with online body-response learning and per-robot/surface profiles.
- [x] Add console selection of mission, detector, class, tag, initial settings and robot profile.
- [ ] Validate adaptive calibration and continuous homing on the physical robot; record results for multiple surfaces.
- [ ] Estimate phone exposure/transport delay automatically; current compensation uses an explicit profile estimate.
- [ ] Try and tune a Kalman filter using recorded real-camera footage; adaptive homing currently uses bounded model prediction.
- [ ] Add stall/slip detection and richer direction/deadband dynamics if the local linear response model proves insufficient.
- [ ] Add a heartbeat watchdog for a stalled host or network failure before TCP detects a disconnect; normal disconnect cleanup now clears outputs.
- [ ] Add encoder feedback before implementing closed-loop wheel-speed PID or measured turning angles.
- [ ] Add capture timestamps / clock synchronization to distinguish camera buffering from fresh exposure.
- [ ] Revisit optional body-response learning as a separate small script after basic missions work reliably.
- [ ] Validate balance, friction PWM, stopping distance, settle time and arrival thresholds on the physical robot.
