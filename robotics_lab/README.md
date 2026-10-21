# Shared robot code

These modules contain reusable components and do not run a robot on import.

| Module | Implemented behavior |
|---|---|
| `motor_control.py` | Signed PWM ramps, cached output commands, reversal through zero, immediate stop, gradual stop, and timed search scheduling |
| `april_navigation.py` | Earlier continuous image-based navigation with filtering, bounded prediction, acquisition/loss states and arrival confirmation; used by `demos.apriltag.robust` |
| `paths.py` | Checkout-relative paths for models, runs and printable tags |

Use `from robotics_lab.motor_control import SmoothMotorController`. After pin setup, request each motor's speed repeatedly on the board's event loop; calls advance the PWM ramp. `stop()` bypasses the ramp for exit/fault stops. `is_stopped` describes commanded PWM, not measured wheel velocity.

The fully discrete mission uses `missions.stop_and_stare.controller` for decisions and `motion.PulseDriver` for a complete ramp/hold/brake step. It does not use the continuous navigator to steer during movement.

Run relevant checks from the root:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_motor_control tests.test_timed_search tests.test_robust_april_tag -v
```
