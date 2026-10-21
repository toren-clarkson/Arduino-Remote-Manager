# Generated outputs

The [local webapp](../webapp/README.md) Jobs page shows process logs,
configuration snapshots, previews and artifact downloads. Web training outputs
use `training/<job-id>/`; mission outputs retain the timestamp layout below.

- `stop_and_stare/<timestamp>/events.csv`: mission phases, stationary detections, decisions, timing and requested pulses.
- `stop_and_stare/<timestamp>/observations/*.jpg`: raw stationary images, only when `--save-frames` is selected.
- `training/custom_detector*/`: training curves, validation metrics, predictions and `weights/best.pt`.

Generated contents are ignored by Git. Each mission creates a new timestamped folder. Logs use local monotonic times and laptop frame receipt; they do not include an unavailable phone exposure timestamp. Keep a run's logs when comparing physical tuning changes.
