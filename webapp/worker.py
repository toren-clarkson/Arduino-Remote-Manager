"""Background executors. Never run inside Flask's request process."""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from .catalog import JOBS
from .storage import Store, extract_dataset, write_json


def cli_call(module, argv):
    previous = sys.argv
    try:
        sys.argv = [module.__name__, *map(str, argv)]
        module.main()
    finally:
        sys.argv = previous


def dataset_zip(store, directory, p):
    output = store.path(p['output'])
    extract_dataset(store.path(p['archive'], exists=True), output)
    return dict(files=[store.relative(output)], message='Imported dataset. Select its data.yaml in Training.')


def extract(store, directory, p):
    from training.extract_frames import extract as extract_frames
    output = store.path(p['output'])
    output.mkdir(parents=True)
    extract_frames(store.path(p['video']), output, p['fps'])
    return dict(files=[store.relative(output)], message='Frames extracted. Label them before training.')


def roboflow(store, directory, p):
    from training import download_dataset
    cli_call(download_dataset, ['--workspace', p['workspace'], '--project', p['project'],
                               '--version', p['version'], '--output', store.path(p['output'])])
    return dict(files=[p['output']], message='Roboflow dataset downloaded.')


def inspect_model(store, directory, p):
    from ultralytics import YOLO
    model = YOLO(str(store.path(p['model'])))
    data = dict(task=model.task, classes=model.names, format=Path(p['model']).suffix)
    write_json(directory/'inspection.json', data)
    print(json.dumps(data, indent=2))
    return dict(data=data, message='Model inspected; classes are now shown in Models.')


def train(store, directory, p):
    import shutil
    import polars  # fail before training if the CSV/checkpoint runtime cannot load
    from ultralytics import YOLO
    model = YOLO(str(store.path(p['model'])))
    if model.task != p['task']:
        raise ValueError(f"Initial model task is {model.task}, but requested task is {p['task']}. Use matching weights and labels.")
    output = store.path(p['output'])
    if output.exists():
        raise ValueError('Model output already exists.')
    run = store.path('runs/training')/directory.name
    model.train(data=str(directory/'dataset.yaml'), epochs=p['epochs'], imgsz=p['imgsz'],
                batch=p['batch'], device=p['device'], workers=0,
                project=str(run.parent), name=run.name, exist_ok=False)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training did not produce best.pt.')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    return dict(files=[p['output'], store.relative(run)], data=dict(task=model.task, classes=model.names),
                message='Training finished; best.pt copied to the model library.')


def predict(store, directory, p):
    from ultralytics import YOLO
    model = YOLO(str(store.path(p['model'])), task=p['task'])
    if model.task != p['task']:
        raise ValueError(f"Model task is {model.task}, not {p['task']}.")
    count = 0
    last = None
    for result in model.predict(str(store.path(p['source'])), stream=True,
                                conf=p['confidence'], imgsz=p['imgsz'], device=p['device'],
                                project=str(directory), name='prediction', save=True, verbose=False):
        count += 1
        last = result.summary()
        if count == 1:
            import cv2
            cv2.imwrite(str(directory/'preview.jpg'), result.plot())
        if count % 30 == 0:
            print(f'Processed {count} frames', flush=True)
    data = dict(task=model.task, frames=count, last_frame=last)
    write_json(directory/'prediction.json', data)
    return dict(data=data, message=f'Inference finished: {count} frame(s). Outputs are in job artifacts.')


def tag_generate(store, directory, p):
    from tools import generate_tag
    cli_call(generate_tag, [p['tag_id'], '--output', store.path(p['output'])])
    return dict(files=[p['output']], message='Print with the complete white border visible; avoid glossy paper.')


def read_image(store, relative):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(store.path(relative).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0]*image.shape[1] > 30_000_000:
        raise ValueError('Use a valid image under 30 megapixels.')
    return image


def tag_check(store, directory, p):
    import cv2
    import numpy as np
    from pupil_apriltags import Detector
    image = read_image(store, p['image'])
    detector = Detector(families='tag36h11', nthreads=2, quad_decimate=p['decimate'], refine_edges=1)
    tags = detector.detect(np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    data = []
    for tag in tags:
        accepted = tag.hamming == 0 and tag.decision_margin >= p['margin']
        data.append(dict(id=int(tag.tag_id), margin=float(tag.decision_margin), hamming=int(tag.hamming),
                         accepted=accepted, center=tag.center.tolist(), corners=tag.corners.tolist()))
        color = (0, 220, 0) if accepted else (0, 100, 255)
        cv2.polylines(image, [tag.corners.astype(np.int32)], True, color, 2)
        cv2.putText(image, f'ID {tag.tag_id} margin {tag.decision_margin:.1f}',
                    tuple(tag.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    cv2.imwrite(str(directory/'preview.jpg'), image)
    write_json(directory/'tags.json', data)
    print(json.dumps(data, indent=2))
    return dict(data=data, message=f'{sum(t["accepted"] for t in data)} accepted tag(s), {len(data)} decoded.')


def tag_inject(store, directory, p):
    import cv2
    image = read_image(store, p['image'])
    x, y, size = p['x'], p['y'], p['size']
    if x+size > image.shape[1] or y+size > image.shape[0]:
        raise ValueError('The complete tag and white border must fit inside the background image.')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    edge = max(1, size//8)
    tag = cv2.aruco.generateImageMarker(dictionary, p['tag_id'], size-2*edge)
    tag = cv2.copyMakeBorder(tag, edge, edge, edge, edge, cv2.BORDER_CONSTANT, value=255)
    tag = cv2.cvtColor(cv2.resize(tag, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    image[y:y+size, x:x+size] = tag
    cv2.imwrite(str(directory/'synthetic-tag.png'), image)
    cv2.imwrite(str(directory/'preview.jpg'), image)
    return dict(message='Synthetic image created. Download it or copy it to datasets/ for a detector test; it is not a real-camera calibration.')


def mission(store, directory, p):
    from missions.stop_and_stare.config import Config
    from missions.stop_and_stare.run import main, StopRequested
    from missions.stop_and_stare.telemetry import MissionTelemetry
    telemetry = MissionTelemetry(directory, drive=p['drive'], replay=bool(p['video']))
    values = dict(p['config'])
    values['weights'] = store.path(values['weights'])
    values['body_model'] = str(store.path(values.get('body_model', 'calibrations/robot-body.json')))
    cfg = Config(**values)
    args = SimpleNamespace(config=cfg, video=store.path(p['video']) if p['video'] else None,
                           drive=p['drive'], headless=True, save_frames=p['save_frames'],
                           max_batches=p['max_batches'], stop_file=directory/'stop', preview_path=directory/'preview.jpg',
                           telemetry=telemetry)
    args.continuous_preview = p.get('continuous_preview', False)
    args.calibrate_body = p.get('calibrate_body', False)
    finished = threading.Event()
    def guard():
        # Cooperative fail-stop if this web server disappears. The firmware is unchanged.
        while not finished.wait(1):
            heartbeat = store.state/'heartbeat'
            if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime > 8:
                (directory/'stop').touch()
                return
    threading.Thread(target=guard, daemon=True).start()
    try:
        session = asyncio.run(main(args))
        telemetry.update(phase='FINISHED', camera_connected=False)
        if args.calibrate_body:
            return dict(files=[store.relative(session),store.relative(Path(cfg.body_model))],
                        message='Body response profile learned. Select body_control=learned and this body_model for a mission.')
        return dict(files=[store.relative(session)], message='Mission process finished; inspect its decisions and stationary images in the session log.')
    except StopRequested:
        telemetry.update(phase='STOPPED', camera_connected=False)
        return dict(message='Mission stop handled; zero-PWM/connection cleanup completed without reported errors.')
    except BaseException as exc:
        telemetry.update(phase='FAILED', error=str(exc), camera_connected=False)
        raise
    finally:
        finished.set()
        telemetry.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    store = Store(args.root)
    if len(args.job) != 32 or any(c not in 'abcdef0123456789' for c in args.job):
        parser.error('Invalid job ID')
    directory = store.state/'jobs'/args.job
    record = json.loads((directory/'job.json').read_text(encoding='utf-8'))
    os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '30000000')
    try:
        handler = globals()[JOBS[record['kind']]['handler']]
        result = handler(store, directory, record['params'])
        write_json(directory/'result.json', result or {})
    except BaseException as exc:
        write_json(directory/'result.json', dict(message=str(exc) or type(exc).__name__))
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
