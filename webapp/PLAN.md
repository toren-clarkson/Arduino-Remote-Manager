# Main project control room — implementation plan

This app manages **Main**, not the enclosing class folder. Run it locally at
`http://127.0.0.1:5000`. It uses the project's Python environment, Flask/Waitress,
plain HTML/CSS/JavaScript, existing perception/motion modules, and background
Python processes. No cloud hosting or JavaScript build step is required.

## Pages and workflows

| Page | User workflow | Files / execution |
| --- | --- | --- |
| Home | See project folders, useful entry points and active jobs | Live directory inventory and links to Files |
| Datasets | Upload images/videos or a labeled YOLO ZIP, extract video frames, import a Roboflow version, browse and edit labels/YAML | `datasets/`, existing training helpers |
| Models | Upload local `.pt` weights, inspect classes/task, edit a display name and notes, rename the weight file or move it to Trash | `models/`, persistent metadata |
| Training | Select dataset YAML and starting weights, set epochs/image size/batch/device and output name; follow logs and results | One training process at a time; `runs/training/` and `models/` |
| AprilTags | Generate tag36h11 IDs 0–586, preview/download/print markers, test a detector on an uploaded image, inject a synthetic marker into a test image | `assets/tags/`, `datasets/`, job artifacts; synthetic images are identified as such |
| Missions | Choose object → tag, object-only or tag-only; pick model/class/tag; configure IPs, pins, thresholds, bursts, motion holds, ramps and settling | Validated configuration snapshots and the shared stop-and-stare runner |
| Jobs | Observe status/log tail, stationary mission preview, artifacts; request stop | Persistent job records/logs under `.control/jobs/` |
| Mission monitor | Connect the phone, view live JPEGs and stationary detections separately, follow actual process output, inspect phase and movement requests, Stop | `/live`, per-job telemetry snapshots and MJPEG display stream |
| Files | Navigate project directories, read/edit text, upload files, create folders, rename, download, trash/restore | Contained project paths; protected environment/secrets/runtime internals |
| Settings | Save reusable mission profiles and inspect runtime information | `.control/profiles/`; no API keys stored |

The sidebar provides task pages; Home also lists actual top-level project folders.
Files links retain directory paths so directories have their own browsable page.
Each task page links to its implementation/use documentation.

## Data lifecycle

1. Upload footage → extract diverse still frames → label/export in Roboflow (or
   edit YOLO `.txt` labels locally). Images/video alone are not labeled training data.
2. Import a labeled YOLO detection export ZIP or download a Roboflow version using
   `ROBOFLOW_API_KEY` inherited from the server environment. No key goes into browser
   forms, saved profiles, arguments or job logs.
3. Select its `data.yaml` and initial `.pt` weights, train, inspect the resulting
   classes, and select the model in a mission. ByteTrack associates detections;
   it does not need a separate tracker-training video.
4. Generate/print a tag, test detection on a real photo, then select that tag ID
   in the mission. Synthetic injection is an image test tool, not calibration or
   proof that a printed marker/camera/robot works.
5. Save mission profiles, replay a video without hardware, then explicitly enable
   physical driving when ready. Every mission uses stop, settle, fresh frames,
   confirm, one bounded pulse, stop. No detections during movement.

## Execution and persistence

Jobs use fixed Python entry points with JSON parameters, never a shell command
provided by the browser. Slow model loads, training and inference stay outside
HTTP request threads. Only one mission owns the phone/robot at a time, and one
compute-intensive job owns the inference/training slot. Job records include
config, timestamps, status, output references and a readable log. Completed
records survive restart. Interrupted records are marked interrupted; they are
never automatically restarted. Config snapshots keep each run reproducible.

Mission Stop writes a cooperative stop marker that is checked in camera waiting,
observation and movement polling; existing `finally` cleanup sends zero PWM and
closes the connection. Other job types may be terminated as subprocesses. Stop
requests do not claim physical acknowledgement from the Arduino. Host/network
failure still requires the existing robot power cutoff; no firmware watchdog is
added by this webapp.

Renaming/deleting inputs and editing source/config are blocked while jobs are
active. Replaced text and trashed files are retained locally for recovery.
Uploaded weights are executable PyTorch artifacts: use weights from sources you
trust. ZIP imports reject traversal/symlink entries and enforce extraction size
and member limits. Dataset YAML supplied to training is validated to keep image
paths local, and downloads/code hooks are disallowed.

## Local application boundaries

Bind only to loopback. Validate Host and Origin, require a per-session CSRF token
for writes, and do not expose arbitrary process execution. All file operations
resolve inside Main, reject symlinks and protect `.venv`, `.git`, `.env`, hidden
paths and control state. HTML/source files are downloaded as attachments rather
than executed in the app origin. File writes use optimistic revision checking.
The app is a trusted local workbench, not a multi-user deployment or sandbox for
untrusted models/datasets.

## Verification and limits

Exercise API workflows against a temporary project: uploads, archive validation,
text conflicts, CRUD/trash/restore, profile/config validation, job locking,
CSRF/path rejection, marker generation and mission-stop behavior. Run the existing
hardware-free suite. Verify a real headless replay through the job manager and
inspect the browser pages. Do not drive motors or train a lengthy custom model
as part of application testing. Roboflow downloads require the user's credentials.

References: [Flask file uploads](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/),
[Ultralytics training](https://docs.ultralytics.com/modes/train/),
[tracking](https://docs.ultralytics.com/modes/track/),
[Roboflow export](https://docs.roboflow.com/datasets/exporting-data).
