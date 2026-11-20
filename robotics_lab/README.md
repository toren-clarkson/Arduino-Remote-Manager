# Shared robot helpers

The root mission files run directly. These small helpers are shared so motor ramps
and the older filtered navigator are maintained in one place; none opens hardware
on import.

| Module | Purpose |
|---|---|
| `motor_control.py` | PWM ramps, reversal through zero, immediate/gradual stopping and intermittent search scheduling |
| `april_navigation.py` | Filtered navigation used by `robust-april-tag.py` |
| `paths.py` | Paths to local model weights, recorded runs and printable tags |
| `network.py` | Fixed robot AP address/Telemetrix port and automatic PC PhoneSense address selection |
| `adaptive_control.py` | Local image-response identification, RLS updates, robot/surface profile storage and visual servo control |
| `console.py` | Simple console settings shared by the root programs |

All stop-and-stare mission settings, perception, memory and sequencing are in
[stop-and-stare.py](../stop-and-stare.py). There is no webapp, job runner, live-video
proxy or component registry. [adaptive-homing.py](../adaptive-homing.py) is the
standalone continuous learned-control mission; its
[guide](../docs/ADAPTIVE_HOMING.md) explains calibration, limits and research.
