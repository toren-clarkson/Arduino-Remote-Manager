# Main robotics control room

A localhost workbench for the **Main** repository. It manages files, model
artifacts, labeled datasets, training, AprilTags, mission profiles and background
jobs. The [implementation plan](PLAN.md) describes the page/workflow design;
[EXTENDING.md](EXTENDING.md) explains the capability registries.

## Start / stop

From Main in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m webapp
```

Open **http://127.0.0.1:5000** or **http://localhost:5000** on this PC. Keep that
terminal running. Ctrl+C stops the web server and requests active jobs to stop.
Use `--port 5001` if another application uses port 5000. Only one control-room
server can manage Main at a time, regardless of port. There is no debug reloader.

If startup says a control-room server is already managing Main, use that server
or stop it with Ctrl+C in its owning terminal before starting another. A leftover
`server.lock` file does not prevent startup: the operating-system lock is released
when its process exits. Duplicate launches report a short message, without reading
the byte locked by the existing Windows process.

This HTTP address is the management app. The phone camera still connects to the
separate **HTTPS PhoneSense URL** printed in a live mission's job log, typically
on port 8080. Starting the webapp alone does not start PhoneSense or connect the
Arduino. On another device, `localhost` refers to that device, not this PC.

## Pages

For the new **Learn robot body** option, saved profiles and learned control,
follow [calibrations/README.md](../calibrations/README.md). It uses a fixed AprilTag
and eight explicit user-started motor probes. No neural-network training is needed.

**Continuous detector preview** on Missions runs fresh detections continuously
without opening an Arduino connection or issuing movement requests. Choose tag-only
for AprilTags or an object provider for HSV/YOLO. It remains on the selected first
leg instead of performing an approach. `max_batches` becomes a preview frame limit;
0 continues until Stop. Driving and calibration are rejected in this mode.

Live raw and continuous annotated frames use authenticated loopback streaming between
worker and webapp, capped by `preview_fps` (1–30). They no longer wait for per-frame
disk publication. Immutable JPEG downloads remain as approximately one-second snapshots;
status snapshots are throttled separately. Older jobs retain a snapshot fallback.
The phone still connects only to its normal PhoneSense URL, not the internal IPC port.

Normal driving deliberately updates its detector image only during stationary bursts.
Movement, ramping, settling and frame flushing still consume time. A smooth live feed
does not mean 25 robot decisions per second. Camera FPS and detector milliseconds in
the monitor distinguish acquisition/display rate from model throughput. CPU YOLO may
need a smaller model or image size; faster video transport cannot make inference free.

Question-mark icons beside inputs explain their effect. Hover, focus with Tab,
or tap to read them. Job and mission schemas carry descriptions for extensions.

For an OpenCV color mission, choose `hsv` as **object provider**, leave `color_h = -1`,
and start a preview. Connect the phone on Mission monitor, choose **Freeze frame &
pick color** (or click the live video first), then click the target in the frozen
image. No color means no search movement. Changes apply while stopped and discard
old-color observations; selections do not overwrite profiles. See the
[control guide](../missions/stop_and_stare/CONTROL.md) for HSV tuning and optional
PID steering / Kalman filtering.

### Overview and files

Overview lists the actual visible top-level directories. Every directory links
to a Files page whose URL retains its path, e.g. `/files?path=missions`.

Files supports folder creation, text-file creation/editing, upload, download,
image/video preview, rename, Trash and restore. Text editing supports UTF-8
Python, Markdown, JSON, YAML, TOML, CSV, HTML/CSS/JS and other common text formats
up to 2 MB. If VS Code changed a file after you opened it in the app, saving is
rejected until you reload. Previous text contents are backed up under
`.control/history/`. These backups are for manual recovery; Trash has its own
restore controls. There is no permanent-delete button.

Top-level project folders, `.venv`, Git state, hidden files, control-room state,
certificate/key files, junctions and symlinks are protected. Source files remain
ordinary files that VS Code sees immediately. Source changes require restarting
the server to reload its own backend; running Python jobs use their initial
imports. Finish jobs before editing, renaming, uploading replacements or trashing
their inputs. Uploaded filenames must use ordinary letters/numbers, spaces,
underscores, hyphens and dots. Upload requests are limited to 2 GB / 500 files.

### Datasets and Roboflow

1. Upload footage or images into a collection such as `datasets/bottle-footage`.
2. Extract video frames at a chosen sample rate if needed. These are **unlabeled**.
3. Annotate objects in Roboflow and generate a dataset version. Export as YOLO
   detection data. For segmentation, pose or OBB, use the corresponding labels
   and a compatible model; these tasks have different annotation formats.
4. Upload the export ZIP, then run **Import labeled dataset ZIP**, or use the
   Roboflow import form with workspace, project and version IDs.
5. Inspect/edit the imported `data.yaml` and browse its images/labels in Files.

For direct Roboflow download, set the key before launching the server:

```powershell
$env:ROBOFLOW_API_KEY = 'YOUR_PRIVATE_KEY'
.\.venv\Scripts\python.exe -m webapp
```

The UI shows only whether a key exists; it never displays or stores the key.
It is inherited by download jobs. Alternatively, upload a downloaded ZIP with no
key configured. This is a **local training/inference workflow using Roboflow
datasets**, not a Roboflow hosted-inference subscription or cloud training UI.

ZIP imports allow images, labels, YAML and related text metadata, reject unsafe
paths/links, and cap extracted content at 4 GB / 50,000 entries. Training accepts
local image directories inside `datasets/` with train and val splits, optional
test, names, and optional pose metadata. Dataset download hooks, arbitrary code,
URLs and image-list text-file splits are deliberately unsupported by the built-in
training executor. Normalize exported YAML paths in Files if they refer to an
old machine. Roboflow download jobs use the repository's existing download helper;
local ZIP import uses the stricter bounded importer.

Roboflow's `roboflow:` provenance metadata is accepted and omitted only from the
job's normalized training YAML. The original dataset YAML stays unchanged. If an
export has `../train/images` beside a local `train/images` folder, the app uses
that local folder when the declared path is missing and no explicit `path` root
is set. The same applies to validation/test splits. Existing sibling-directory
paths keep their meaning; all final paths must remain inside `datasets/`.

### Models and training

Upload weights to Models. Edit display names, notes and provider without changing
the file, or use Rename file to change its filename. Inspect an Ultralytics model
to cache its task/classes in the library. Multiple files/versions are supported;
give outputs distinct names. Existing saved profiles keep their original weight
path when a file is renamed, so update affected profiles before relaunching.

The Ultralytics provider supports `.pt`, `.onnx`, `.engine` and `.tflite` inference
where the corresponding backend is installed and compatible with the machine.
Exported formats may require additional runtimes (for example ONNX Runtime or
TensorRT). `.pth`, `.bin` and `.safetensors` can be cataloged with the Other provider;
their execution requires a project-specific provider/executor. There is no claim
that an arbitrary model binary is compatible with YOLO. Model files can contain
executable PyTorch serialization; load weights only from trusted sources.

To train, select local dataset YAML, matching `.pt` starting weights, task,
epochs, image size, batch, device and a new output filename. Detection,
segmentation, pose and OBB share this executor; labels and starting weights must
match the chosen task. Classification has an inference executor but would need a
directory-based training plugin. The app does not create labels automatically.
Training uses workers=0 for Windows, writes checkpoints/metrics/plots under
`runs/training/<job-id>`, and copies `best.pt` into Models on success. A stopped or
failed run may leave partial results/checkpoints; inspect Files before restarting.
Resume/export/model conversion are extension points, not built-in controls.

### AprilTags and injection

Generate **tag36h11** IDs **0–586**, with distinct marker filenames. Generated
images include a white border. Download or use Print; print at actual size and
keep the border. The print page displays the entire image at 160 mm wide: for the
app's generated 800-pixel image, the black tag square is 120 mm. A browser/printer
can rescale it, so check the print dialog. This mission does not use metric pose
or a physical marker size.

Upload a real marker photo and run **Detect AprilTags in a photo**. Job artifacts
include an annotated image and `tags.json` with IDs, corners, decision margins,
hamming errors and the configured acceptance test. No detection can also be a
valid result; inspect lighting, focus, border, tag size and motion blur.

**Inject synthetic tag into an image** pastes a bordered marker at an exact pixel
position and size. The entire marker must fit. Download the output or copy it to
datasets from its job page, then run a detector test on it. This provides
repeatable image tests; it does not reproduce camera blur/lighting, calibrate a
camera, or validate physical robot performance. Video injection is not implemented.

Select the same tag ID in Missions. Tag detection margin, decimation and thread
count are configurable in the mission profile. Only tag36h11 is wired into the
built-in navigation executor; adding another family requires matching generation,
detection, ID validation and configuration support.

### Missions and trackers

Three navigation templates share the existing stop-and-stare controller:

| Template | Inputs | Finish condition |
| --- | --- | --- |
| Object → AprilTag | Model, class, tracker, tag ID | Centered object reaches area threshold, then centered tag reaches width threshold |
| Object approach | Model, class, tracker | Centered object reaches area threshold |
| AprilTag approach | Tag ID/detector settings | Centered tag reaches width threshold; no YOLO weights loaded |

Navigation currently loads local Ultralytics `.pt` detection/segmentation/pose
models and uses axis-aligned boxes. OBB/classification models can run inference
jobs but cannot supply these navigation inputs. Choose **ByteTrack**, **BoT-SORT**,
or **Detection only**. Track IDs persist within a stationary observation burst
and reset after movement; the app does not guarantee identity through a camera
jump or among several indistinguishable objects. BoT-SORT uses its standard
configuration, without enabling an extra ReID model download.

Configure the Arduino IPv4 address directly—there is no automatic UNO discovery.
Optionally set the PC interface shared with the phone, and check motor pins
in IN1, IN2, PWM order. The UI exposes thresholds, speeds, pulse holds,
acceleration/deceleration, settling, fresh-frame discard count, burst size,
confirmation count, timeouts and short-term position memory. Hold times exclude
PWM ramps; turns are timed, not measured degrees. Preserve/adjust these values in
named profiles; every job also retains its own complete snapshot.

**Preview is the default.** Choose a replay video for hardware-free testing, or
leave the source as live PhoneSense and connect the phone using its logged URL.
Live preview does not connect the Arduino. Enable physical driving only when
you intend motion. Video replay forbids driving. A preview simulates the timing
of pulses but cannot validate stopping distance, balance or motor wiring.

Starting a mission opens **Mission monitor** (`/live?id=<job-id>`). The sidebar
also opens `/live`, which selects a running mission or the latest recorded one.
Choose another session from its dropdown. Jobs & logs provides both View log and
Monitor links for missions.

The monitor shows:

- **PhoneSense connection links**, including links recovered from older terminal
  logs. They appear after model loading/warmup and camera-server startup. A replay
  does not start PhoneSense. Ended missions show their historical addresses but
  require a new live mission before the phone can connect.
- **Live camera**, up to 5 received JPEGs per second, including during motion.
  This is display-only: it does not run the detector or change navigation.
- **Last stationary detector image**, held unchanged during movement, and the
  current phase, mission leg, preview/drive mode and latest requested action.
- **Terminal output**: stdout and stderr from that mission's actual Python
  subprocess, refreshed every second. Follow output can be disabled while reading.
  View log puts this selected job above the history table. The page displays the
  latest 60 KB; Download full log retrieves the complete file, including startup.
- **Movement requests**, the latest 50 decisions with signed normalized left/right
  demands and hold durations. Holds exclude ramps. These are requested commands,
  not encoder measurements or acknowledgements from the Arduino; all decisions
  remain in the session CSV and terminal log.
- **Stop**, job parameters, results and artifact downloads.

Brief camera gaps put a mission into **WAIT_RECONNECT**. Motors are stopped and
PhoneSense stays running for `camera_reconnect_timeout` (60 seconds by default).
On the phone, tap Connect and Start camera again if the PhoneSense page returned
to its Connect gate; the installed client does not reconnect automatically.
After fresh frames resume, the runner discards the interrupted burst, settles,
flushes new frames and confirms the target again before planning another pulse.
Stop still works while waiting. If the reconnect limit expires, the mission exits.

Frame-arrival waiting (`observation_timeout`) and each detector call
(`inference_timeout`) have independent budgets: a slow/reconnecting camera cannot
use up the next detector call's allowance. Repeated stale results still stop the
mission, and genuinely timed-out inference is not silently retried concurrently.

The raw JPEG/status writer uses a background thread and a single latest snapshot;
it does not accumulate a video queue or write files in motor polling. At most
three camera viewers stream at once so HTTP workers remain available for control
requests. Closing a monitor tab releases its viewer slot.

Camera images, annotated observations and status JSON are published as immutable
versions under each job's `.display/` folder, retaining four recent versions of
each. Readers do not block replacement of a shared filename on Windows. Cleanup
of a version still held by a reader is deferred. Encoding and file writes happen
in the display thread; a display failure drops/retries the display update and
does not abort the mission. Snapshot downloads keep the familiar `preview.jpg`,
`camera.jpg` and `runtime.json` names. Existing jobs' older files remain readable.

During motion the runner still drains/ignores incoming video for control; it stops,
settles and discards new arrivals before collecting its next confirmation burst.
PhoneSense has receipt timestamps, not exposure timestamps; arbitrary network
buffering is still a limitation. Older missions can show terminal logs/URLs, but
camera/phase/command telemetry requires starting a new mission with the updated
runner. No browser reconnect can revive an ended camera server.

### Jobs and Stop

All slow operations run in separate Python processes. Logs refresh every
second. Inspect parameters, result JSON, previews, generated artifacts and
project-output links. One compute job owns model inference/training at a time;
one mission owns the phone/robot slot. Other job types can declare different
resources. Logs/history survive restart; unfinished jobs are marked interrupted
and never restarted automatically.

Mission Stop requests cooperative cancellation, checked during camera waiting,
observation and pulse polling. The runner's `finally` path stops motors and closes
the connection. Loading a model/native inference can delay cancellation until it
returns or times out, but no new motor pulse is scheduled in the meantime. A
server heartbeat also requests cancellation if the mission's server disappears.
Non-mission jobs can be terminated immediately; interrupted extraction/import/
training may leave partial output. A stop request is not an Arduino acknowledgement.
Network/host failures still require a physical power cutoff; this app adds no
firmware watchdog. Earlier demos remain editable/browsable but are not exposed
as drive buttons because several have hardware side effects at import time.

## Files and implementation

| File | Responsibility |
| --- | --- |
| `__main__.py` | Loopback Waitress server and one-server OS lock |
| `app.py` | HTML pages, contained file APIs, CSRF/origin/host checks |
| `catalog.py` | Providers, trackers, mission templates and job schemas |
| `jobs.py` | Validation, resources, subprocesses, persistent status/logs, Stop |
| `worker.py` | Fixed trusted Python executors and artifact creation |
| `mission_status.py` | Read mission snapshots and recover safe PhoneSense links from logs |
| `storage.py` | File containment, revisions/history, trash, ZIP/YAML validation |
| `templates/`, `static/` | Responsive browser UI; schema-rendered forms |

State is local under ignored `.control/`: model metadata, profiles, heartbeat,
job records/logs/previews, Trash and text history. Models/datasets/runs remain in
their existing folders. There is no database service, cloud publication, shell
command textbox or automatic execution of uploaded scripts. This is a single-user
local workbench, not a sandbox for untrusted code, weights or datasets.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check webapp/static/app.js
.\.venv\Scripts\python.exe -m pip check
```

The webapp tests use a temporary project and never drive motors. See
[VERIFICATION.md](VERIFICATION.md) for the executed integration checks and limits.

Primary documentation: [Flask uploads](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/),
[Ultralytics train](https://docs.ultralytics.com/modes/train/),
[Ultralytics track](https://docs.ultralytics.com/modes/track/),
[Roboflow exports](https://docs.roboflow.com/datasets/exporting-data).
