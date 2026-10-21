"""Capability catalog. Extend here and add a worker handler, without editing the UI.

Only trusted project Python may register executors. The browser cannot submit shell
commands, arbitrary modules or tracker YAML paths.
"""

MODEL_PROVIDERS = [
    dict(id='ultralytics', label='Ultralytics YOLO', extensions=['.pt', '.onnx', '.engine', '.tflite'],
         tasks=['detect', 'segment', 'pose', 'obb', 'classify'], inspect=True, predict=True,
         train_extensions=['.pt'], training_tasks=['detect', 'segment', 'pose', 'obb'], navigation_tasks=['detect', 'segment', 'pose']),
    dict(id='generic', label='Other model / artifact', extensions=['.pth', '.bin', '.safetensors', '.onnx', '.pt', '.tflite', '.engine'],
         tasks=['custom'], inspect=False, predict=False, train_extensions=[], training_tasks=[], navigation_tasks=[]),
]
TRACKERS = [
    dict(id='bytetrack', label='ByteTrack', provider='ultralytics', description='Associate boxes within each stationary burst.'),
    dict(id='botsort', label='BoT-SORT', provider='ultralytics', description='Alternative association within stationary bursts; no ReID model download.'),
    dict(id='none', label='Detection only', provider='ultralytics', description='No track IDs; confirm the selected class across stationary frames.'),
]
MISSIONS = [
    dict(id='object_then_tag', label='Object → AprilTag', inputs=['model','apriltag'], description='Find and approach an object, then find and approach the selected tag.'),
    dict(id='object_only', label='Object approach', inputs=['model'], description='Find and approach a selected object; finish at the area threshold.'),
    dict(id='tag_only', label='AprilTag approach', inputs=['apriltag'], description='Find and approach a selected tag; no YOLO weights required.'),
]


def field(key, label, kind='text', default=None, **kwargs):
    from .help import JOB_HELP
    kwargs.setdefault('help', JOB_HELP.get(key, f'Input for {label.lower()}.'))
    if kind == 'number':
        kwargs.setdefault('integer', type(default) is int)
    return dict(key=key, label=label, kind=kind, default=default, **kwargs)


# Each job declares a handler name (resolved only from the trusted worker module),
# parameter schema, resources, and artifact formats. The frontend renders schemas.
JOBS = {
    'dataset_zip': dict(label='Import labeled dataset ZIP', group='datasets', handler='dataset_zip', resources=['download'], outputs=['yolo_dataset'], fields=[field('archive', 'Uploaded ZIP', 'archive'), field('output', 'New dataset folder', default='labeled-dataset')]),
    'extract': dict(label='Extract video frames', group='datasets', handler='extract', resources=['compute'],
                    outputs=['images'], fields=[field('video', 'Source video', 'video'), field('output', 'New dataset folder', default='frames'), field('fps', 'Frames per second', 'number', 2.0, min=.05, max=60)]),
    'roboflow': dict(label='Import from Roboflow', group='datasets', handler='roboflow', resources=['download'],
                    outputs=['yolo_dataset'], fields=[field('workspace', 'Workspace ID'), field('project', 'Project ID'), field('version', 'Version', 'number', 1, min=1, max=100000), field('output', 'New dataset folder', default='roboflow-export')]),
    'train': dict(label='Train a YOLO model', group='training', handler='train', resources=['compute'],
                  outputs=['weights', 'metrics', 'plots'], fields=[field('data', 'Labeled dataset YAML', 'dataset'), field('model', 'Initial Ultralytics .pt model', 'model'), field('task', 'Label / model task', 'select', 'detect', options=['detect', 'segment', 'pose', 'obb']), field('output', 'New weight filename', default='custom-model.pt'), field('epochs', 'Epochs', 'number', 50, min=1, max=10000), field('imgsz', 'Image size', 'number', 416, min=32, max=2048), field('batch', 'Batch size', 'number', 8, min=1, max=1024), field('device', 'Device (cpu or GPU index)', default='cpu')]),
    'inspect_model': dict(label='Inspect model classes / task', group='models', handler='inspect_model', resources=['compute'], outputs=['json'], fields=[field('model', 'Ultralytics model', 'model')]),
    'predict': dict(label='Run inference on an image or video', group='models', handler='predict', resources=['compute'], outputs=['annotated_media', 'json'], fields=[field('model', 'Ultralytics model', 'model'), field('source', 'Image / video', 'media'), field('task', 'Model task', 'select', 'detect', options=['detect', 'segment', 'pose', 'obb', 'classify']), field('confidence', 'Confidence', 'number', .5, min=.01, max=1), field('imgsz', 'Image size', 'number', 416, min=32, max=2048), field('device', 'Device', default='cpu')]),
    'tag_generate': dict(label='Generate printable AprilTag', group='tags', handler='tag_generate', resources=[], outputs=['png'], fields=[field('tag_id', 'tag36h11 ID', 'number', 0, min=0, max=586), field('output', 'New marker filename', default='tag-0.png')]),
    'tag_check': dict(label='Detect AprilTags in a photo', group='tags', handler='tag_check', resources=['compute'], outputs=['annotated_image', 'json'], fields=[field('image', 'Uploaded image', 'image'), field('margin', 'Minimum decision margin', 'number', 20.0, min=0, max=1000), field('decimate', 'Quad decimation', 'number', 1.0, min=.1, max=8)]),
    'tag_inject': dict(label='Inject synthetic tag into an image', group='tags', handler='tag_inject', resources=['compute'], outputs=['synthetic_image'], fields=[field('image', 'Background image', 'image'), field('tag_id', 'tag36h11 ID', 'number', 0, min=0, max=586), field('size', 'Marker + white border (pixels)', 'number', 180, min=32, max=4000), field('x', 'Left edge (pixels)', 'number', 20, min=0, max=100000), field('y', 'Top edge (pixels)', 'number', 20, min=0, max=100000)]),
    'mission': dict(label='Stop-and-stare mission', group='missions', handler='mission', resources=['compute', 'camera_robot'], stop='cooperative', outputs=['csv', 'stationary_images'], fields=[]),
}


def catalog():
    return dict(providers=MODEL_PROVIDERS, trackers=TRACKERS, missions=MISSIONS, jobs=JOBS)
