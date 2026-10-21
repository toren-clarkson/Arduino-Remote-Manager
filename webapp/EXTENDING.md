# Providers, trackers, missions and jobs

The control room separates **artifacts**, **capabilities**, **execution**,
**parameters**, **resource ownership** and **results**. A model file is not a
mission, and a tracker is not a model-training format. Register trusted Python
code in the project; the browser never submits arbitrary commands/modules.

## Add a job with a different input/output format

1. Add an entry to `JOBS` in `catalog.py`. Declare a unique ID, label, group,
   handler function name, resources, output format labels and field schema.
2. Implement that handler in `worker.py`. Signature:
   `handler(store, job_directory, parameters) -> JSON-serializable dict`.
3. Add any task-specific validation/normalization in `Jobs.prepare`. Generic
   schemas already validate required values, ranges, selects and contained paths.
4. Write job-local artifacts inside `job_directory`. Return `files` for project
   paths, `data` for structured output and `message` for a brief result. Write
   `preview.jpg` for a preview. Log progress with `print(..., flush=True)`.
5. For a **hardware** job, implement cooperative cancellation and cleanup, and
   declare `stop='cooperative'` in its catalog entry. Never expose an executor that may
   energize motors but is terminated as an ordinary subprocess. Its handler must
   poll `job_directory/'stop'`, watch the server heartbeat and run zero-PWM cleanup.

Example registration for a future offline CSV calibration report:

```python
'calibration_report': dict(
    label='Analyze calibration CSV', group='training', handler='calibration_report',
    resources=['compute'], outputs=['csv', 'json', 'plot'],
    fields=[field('input', 'Project-relative CSV'),
            field('samples', 'Minimum samples', 'number', 30, min=1, max=10000)],
)
```

Implement the handler before registering it. Validate `input` with
`store.path(..., exists=True)` and an allowed extension, rather than accepting
any machine path. The frontend renders registered job forms automatically on
Datasets, Models, Training, Tags or Missions according to the `group` value.
Existing groups are `datasets`, `models`, `training`, `tags`, `missions`.

Field kinds are `text`, `number`, `select`, `model`, `dataset`, `image`, `video`,
`media`, and `archive`. Model/media/dataset fields use live project inventories.
Select fields supply `options`; numeric fields supply `min` and `max`. Add a new
field-kind validator/renderer only when a truly new widget is needed. Add a
catalog group/page if the new domain merits its own navigation page.

Resource names are arbitrary strings: jobs with any overlapping resources cannot
run concurrently. Built-ins use `compute`, `download`, and `camera_robot`.
Do not remove robot ownership to achieve concurrency. Inputs cannot be renamed,
edited, replaced or trashed while jobs run. Output names are reserved across
active jobs. New executors must also refuse overwriting existing output files.

## Add a model provider or format

Add a `MODEL_PROVIDERS` entry with extensions, supported tasks, inspect/predict
capabilities, training formats/tasks and navigation-compatible tasks. The model
library derives accepted extensions and supports provider/name/notes metadata.
Implement a matching inspection, prediction or training executor and adjust
`Jobs.model` / job capability validation to route it. Merely cataloging a filename
does not make it compatible with an executor.

For example, an ONNX-specific executor might use ONNX Runtime and a JSON label
map without depending on Ultralytics exports. A classification training executor
would accept a directory-organized dataset instead of YOLO box YAML. Keep those
schemas explicit and reject incompatible combinations before hardware connection.
Currently Ultralytics backend availability is checked by the worker on load;
optional runtimes are not automatically installed by the app.

Model inspection caches task/classes as metadata. A renamed weight file keeps
its metadata, but saved mission profiles retain their paths and must be updated.
No executor loads artifacts into the web server process.

## Add a tracker

Register its identifier/description in `TRACKERS`. The current navigation
adapter accepts `bytetrack`, `botsort`, and `none`; update `Config.validate` and
`YOLOObjects._infer` to add another supported adapter. Do not let browser values
become arbitrary YAML paths. Trackers using ReID/extra models must declare these
inputs, validate them and document extra runtimes/resources.

The stationary-burst interface is deliberate: `begin_burst` resets state after
camera motion; `detect` returns a `Sighting` and annotated image. A tracker that
requires continuous video or odometry needs its own observation/execution design;
do not silently feed stale/movement images into the existing stop-and-stare loop.

## Add a mission with different inputs

For interchangeable pieces of the current mission, first see the
[component contracts](../missions/stop_and_stare/CONTROL.md#component-contracts).
`components.py` provides trusted lazy registries for object perception, steering,
position estimation and motor execution. Config validation and UI select choices
derive from these registries. Both YOLO and HSV use the same object leg. New Config
fields need validation and descriptions in `webapp/help.py`; job fields accept
`help` text. New artifact/input formats still require explicit Jobs validation and
form capabilities. Registration alone cannot translate an arbitrary model format
or replace the stationary timing contract of the runner.

Built-in object→tag, object-only and tag-only templates share `Config`, `Planner`
and `Runner`. `MISSIONS` supplies their names/descriptions; `mission_mode`
validation and the planner implement their state transitions. Config fields are
sent as a schema, and additional fields appear automatically in the mission form.

A substantially different mission (waypoints, color following, arm manipulation,
etc.) should register a separate **job** with `group='missions'`, its own schema,
executor and resources. It then receives its own generated form on Missions and
the same job/log/artifact controls. Declare/implement a cooperative stop policy for any
new hardware executor before enabling it. This avoids forcing every mission
into YOLO/class/tag fields or importing old scripts with motor side effects.

## Validation expectations

Add focused hardware-free tests for compatibility checks, input/path validation,
resource ownership, result formats and stop behavior. Use fake boards for motor
logic and video/image fixtures for perception. Run the existing suite. Keep
physical calibration and actual driving as explicit user actions; never start a
robot while merely listing providers, browsing files or importing a module.
