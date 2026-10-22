# Executed verification

Verified locally on this PC, using the project's `.venv`, without connecting an
Arduino or moving physical motors.

## Automated checks

Body learning and streaming update: 121 Python tests pass, including simulated
reversed/unequal motors, independent braking/command integration, eight-probe fitting,
lost-tag aborts, transport authentication and motor-free continuous preview. These
do not physically calibrate or drive the robot.

Actual continuous HSV preview job `0efbd60cde6543d3b38fab2d9ac571da` received a
25 FPS simulated phone via the real PhoneSense HTTPS WebSocket. Two concurrent
webapp streams measured **23.62 FPS raw** and **25.28 FPS annotated** over 75 frames
each, with **14.6 ms** reported inference at the sampled status. No motion was
requested (`drive=False`). Evidence: `.control/phonesense-preview-fps.json`.
An actual video-replay worker measured about 16.3 FPS on both channels; replay also
decodes/re-encodes its source frames. Evidence: `.control/live-preview-verification.json`.

The actual `models/red-ball-v1.pt` CPU preview at input size 416 initially measured
median **322.3 ms** inference over 24 frames (about 3 inferences/second), showing a
model-compute limit distinct from live transport. Evidence:
`.control/yolo-preview-verification.json`. A thread-selection experiment increased
the real worker's median to 465.55 ms despite a faster synthetic warmup, so automatic
thread tuning was removed. These measurements do not promise 25 FPS CPU YOLO.

Root README TODO implementation: **108 Python tests passed**, including 14 new
component/color-picker tests. Both frontend scripts passed `node --check`;
`pip check` reported no broken requirements. All ten page templates rendered
against the real server, including registry-driven select options and field help,
and their generated HTML passed nesting checks. The refactored real YOLO/ByteTrack
provider detected a person in `runs/verification/bus.jpg` after warmup.

Actual web worker job `c82a991c847542e3bd3aeecb084b1106` replayed generated video
with **drive=False**. It waited in WAIT_COLOR with no commands, accepted a color
sample from the frozen-frame API, acknowledged the selection through telemetry,
then used real HSV and AprilTag perception to produce `object_reached` and `done`.
Evidence: `.control/color-picker-verification.json`. No PhoneSense server or
Arduino was connected. Browser surfaces were unavailable, so this is API/template
verification, not visual layout review or physical PID/filter calibration.

Camera reconnect regression: six tests verify independent frame/inference budgets,
stopping PWM before reconnect waiting, discarding partial bursts without search
commands, fresh observation after recovery, Stop while waiting, and bounded
reconnect/inference failure. A real PhoneSense preview job
`0a54706d31b146ed8aded4e37a6531ba` received generated JPEGs over a loopback WebSocket;
the simulated phone disconnected during a pulse, the same HTTPS server remained
available in WAIT_RECONNECT, a second socket resumed with a generated AprilTag,
and the mission completed. `drive=False`; no Arduino connection. Evidence:
`.control/camera-reconnect-verification.json`. The initial physical phone/network
drop in the user's log has no transport reason recorded, so its cause is not
claimed resolved; this fix removes the false timeout/server shutdown and tolerates
brief gaps. The installed phone page still requires Connect/Start camera after
an unexpected socket close.

Windows monitor file-lock regression: the user's downloaded log showed a fatal
`PermissionError` replacing `preview.jpg` while the camera/Arduino had connected.
Five new tests reproduce ordinary Windows open-file replacement failure, verify
immutable snapshot publication while an older reader stays open, retention,
and mission completion despite a locked legacy preview or injected display-write
failure. Annotated images now encode/write on the background display thread.
Actual preview job `9cdfe294b09b420291b14deefcf698f2` completed under two concurrent
MJPEG viewers plus repeated status/observation/artifact downloads: 1,696 stream
chunks, 520 status reads and 210 observation reads, with no file-lock errors or
dropped-display warnings. Evidence: `.control/snapshot-stress-verification.json`.
No physical hardware was connected for verification.

Mission monitor regression: six focused tests pass for latest-frame/background
telemetry, bounded command snapshots, raw display during MOVE without inference,
old PhoneSense URLs outside the log tail, safe connection links, separate raw and
stationary image endpoints, and stream-slot release on disconnect/HEAD.
An actual YOLO→AprilTag **video preview** job (`1ab820319f3f4a269e4a21a00582bef7`)
reached `done` through the monitor API. Received multipart JPEGs, observed raw frames
during movement phases, read command history and terminal output, and fetched the
stationary detector image. Evidence is in `.control/mission-monitor-verification.json`.
The phone links in the user's older mission log were recovered without rerunning
that physical mission. No Arduino connection was made during these checks.

Roboflow import regression: the webapp's 23 API/storage tests pass, including
export metadata stripping in the generated snapshot, missing `../` split-path
fallbacks, preserved sibling paths/lists, containment and descriptive errors for
unsupported settings. The user's actual `datasets/red-ball/data.yaml` and selected
50-epoch / 416 / batch-8 CPU parameters passed job preparation and Ultralytics'
dataset validation. Its original YAML was unchanged; no full training was started.

Startup-lock regression: three additional tests pass for real subprocess lock
contention, reacquisition after an exception, and duplicate CLI rejection before
application initialization. A real fresh launch on port 5000 returned HTTP 200;
a second launch produced the friendly already-running message without a traceback.
The verification server was stopped afterward so the user can own the terminal.

- Python unit suite: **94 tests pass**: 23 webapp API/storage tests, six monitor
  tests, six camera-recovery tests, five snapshot I/O tests, three startup-lock tests
  and 51 perception/motion/replay/training-helper tests.
- `node --check webapp/static/app.js`: passes.
- `node tests/check_webapp_ui.cjs`: all ten page templates render against the
  live API using a minimal document adapter; generated HTML has balanced tags
  and no nested forms. This is a nonvisual rendering check.
- `pip check`: no broken requirements.

The API tests exercise page routes/catalogs, CSRF/origin/Host checks, path/secret
protection, upload traversal/overwrites, optimistic editor revisions/backups,
file CRUD/Trash/restore, model metadata/rename, profile validation, ZIP/YAML
validation, resource ownership, plugin schemas with different input/output
formats, cooperative-stop policy, tag-only/object-only planning, stop marker
polling and a real marker-generation subprocess in a temporary project.

## Actual background jobs through localhost HTTP

| Workflow | Evidence / result |
| --- | --- |
| Inspect local YOLO26n | `3305e76937df467e95740e21792dea98`; model classes/task cached in library |
| YOLO image inference | `c35d959f866242f186dc3d48f2299ee9`; bus fixture annotated and saved |
| Detect existing AprilTag | `5a356aec936e4a66a634ee9ed1ee1c34`; tag 0 accepted |
| Inject synthetic tag | `2b29f1e12e8b41c1a23f5df3e5076bbe`; tag 23 pasted into bus image |
| Copy and detect injection | `1d01b49ffc294509b04e0819bb907922`; copied image detected as tag 23 |
| Extract frames | `90bade7b806b4a4ba62c9dce8828198a`; real replay sampled into a dataset folder |
| Object → tag replay | `f4292ffcbd5f4bbd98ab4ede27508103`; actual YOLO + ByteTrack + Pupil AprilTag, reached `done`; preview only |
| Upload/import labeled ZIP | `c849903255b3450f8c05f63d6de5043f`; local bounded ZIP import succeeded |
| Train and copy weights | `9a563425851f4ef393ff794d3748b7f4`; one CPU epoch, checkpoints/metrics saved and best weights copied |
| Tag-only cooperative Stop | `3e4075f2ca68483a9dfc00d9c47a5236`; deliberately nonexistent model path, no YOLO needed; stop requested during a simulated pulse, cleanup completed, status stopped |

Open Jobs & logs to view these records. `.control/webapp-integration.json`,
`webapp-training-integration.json`, `webapp-stop-integration.json` and
`webapp-ui-check.json` retain local check metadata. Job artifacts/logs remain
under `.control/jobs/`; mission/training outputs use their normal `runs/` paths.

BoT-SORT and Detection only were also tested directly with three repeated bus
images: all produced the bus target, BoT-SORT retained ID 1 within the burst,
and Detection only returned no track IDs. ByteTrack ran in the full mission.

## Environment issue found and fixed

The first training attempt (`7f6d2877bb6c4f6f93f020c90c9c66f6`) completed a batch
but failed while saving metrics/checkpoints: Polars 2.0's CPU check returned
`unknown feature flag: 'sse3'`. This PC reports ARM64 while the installed Python
runtime is win-amd64. Installing `polars[rtcompat]==1.35.2` restored import/CSV
operations; the second actual training job succeeded. A platform-specific project
dependency preserves the tested version, and training now imports Polars before
starting expensive work. No CPU checks were bypassed and no library source was
patched. Upstream sources: [Polars installation](https://docs.pola.rs/user-guide/installation/),
[1.35.2 CPU-check implementation](https://github.com/pola-rs/polars/blob/py-1.35.2/py-polars/src/polars/_cpu_check.py).

The tiny training fixture reuses a single image for train/validation and rough
boxes. It tests plumbing only and has no meaningful model-quality result. Its
`models/webapp-smoke.pt` output was moved to recoverable Trash, so it is not
offered as a deployment model. QA-only dataset folders and generated image copies
were also moved to recoverable Trash to keep the active datasets clean. Restore
`datasets/webapp-training-smoke` from Files → Trash if you want the tiny fixture;
its README identifies it as an integration test. Job logs and results remain.
The failed job was retained for troubleshooting evidence.

## Limits

- Browser automation surfaces were unavailable (no browsers/apps exposed), so
  no screenshot, responsive-layout or real browser interaction review is claimed.
- Live phone streaming, physical driving/balance/stopping distance and Arduino
  stop acknowledgement were not tested. All executed missions used video preview.
- No private Roboflow key was available for an authenticated download. Local ZIP
  import was verified end to end; the credential gate is tested without a key.
- Detection training and native `.pt` inference were exercised. Other task labels,
  exported-model runtimes and future provider plugins require their own compatible
  models/dependencies and validation.
