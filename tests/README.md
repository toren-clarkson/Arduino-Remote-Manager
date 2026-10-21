# Automated verification

`test_mission_components.py` adds hardware-free coverage for real HSV-to-AprilTag
perception, exact frozen-frame color picking, safe color changes, PID timing/limits,
Kalman uncertainty and component substitution. Physical tuning and browser layout
require separate hands-on verification.

From the root after installing the editable project:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use simulated hardware; they never intentionally connect to the Arduino. Detector tests use generated tags and video clips. Model smoke checks use local weights separately from unit tests.

`test_webapp.py` covers the local workbench's path/CSRF checks, uploads,
revision conflicts, model metadata, file/trash recovery, profile validation,
archive/YAML validation, resource locking, cooperative mission stop and a real
marker-generation subprocess. It uses a temporary project. Frontend templates
can also be checked without a browser with `node tests/check_webapp_ui.cjs` while
the local server is running; this is not visual browser QA.

| Test group | Coverage |
|---|---|
| `test_stop_and_stare.py` | Object -> tag planning, alignment/proximity, stationary confirmation, directional search, ramp/hold/brake, camera-failure stop, post-settle frame gating, inference only during OBSERVE |
| `test_motor_control.py`, `test_timed_search.py` | PWM ramps, steady output, reversal, emergency stop and complete search pauses |
| `test_april_memory.py` | Earlier simple demo's last-position memory and frozen-frame expiry |
| `test_robust_april_tag.py` | Earlier filtered controller, actual AprilTag backends and recorded replay |
| `test_yolo_tracking.py` | Earlier continuous YOLO demo selection, age checks and asynchronous loop |
| `test_training_tools.py` | Frame sampling and dataset ZIP extraction path checks |

The physical Wi-Fi LED test is an executable demo in `demos/motors/blink_led.py`, not an automated test. Physical balancing, camera exposure, wheel velocity and calibrated turning angles are outside these hardware-free assertions.
