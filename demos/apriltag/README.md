# Earlier color-to-AprilTag demos

Both scripts retain the color -> tag sequence. Check the board address, motor directions, tag ID and stop-size tuning in the chosen file before running it. A printable tag36h11 ID 0 is in `assets/tags/`.

## Simple memory-based version

```powershell
.\.venv\Scripts\python.exe -m demos.apriltag.simple
```

`simple.py` was `simple-april-tag.py`. Click a color; it searches with timed turns and pauses, approaches the color, then searches for and approaches a tag. Separate memories bridge short color/tag detection gaps and choose search direction from the last-seen side. `TARGET_HOLD_SECONDS`, `TARGET_MEMORY_SECONDS`, `MEMORY_FORWARD_SCALE`, `MEMORY_STEERING_SCALE`, `SEARCH_TURN_SECONDS`, and `SEARCH_PAUSE_SECONDS` tune this behavior. Radius/side-length pixel thresholds indicate arrival. Stale camera input clears memory and requests a stop. Esc quits.

## Filtered AprilTag 3 version

```powershell
.\.venv\Scripts\python.exe -m demos.apriltag.robust --camera-only --tag-only --tag-id 0
.\.venv\Scripts\python.exe -m demos.apriltag.robust --video path\to\recording.avi --tag-only --headless
```

`robust.py` was `robust-april-tag.py`. It separates a vision worker from the motor loop and uses confirmed acquisition, image-position filtering, bounded prediction, loss/reacquisition states, tag-ID lock, full-resolution AprilTag 3 decoding, quality thresholds and stable arrival confirmation. Remove `--camera-only` for motors. Without `--tag-only`, click a color to start the full sequence.

`--backend opencv` selects the alternative detector. `--record new.avi` saves raw processed frames; `--log-csv new.csv` saves timing/vision diagnostics. Output paths must be new. Replay always disables the board connection. Full research and tuning notes are in [DESIGN.md](DESIGN.md).

These earlier approaches still permit movement while observing during approach. For **every motion** to be followed by a stationary look, use [the new mission](../../missions/stop_and_stare/README.md).


## TODO
1. Stop Stare
2. As it approaches the tracking object/tag, the degree at which it steers should decrease