"""Validated background jobs, resource ownership and cooperative mission stopping."""

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, fields
from pathlib import Path

from missions.stop_and_stare.config import Config
from .catalog import JOBS, MODEL_PROVIDERS
from .storage import IMAGES, VIDEOS, name, now, training_yaml, write_json

ACTIVE = {'starting', 'running', 'stop_requested'}


def config_values(store, values):
    allowed = {f.name for f in fields(Config)}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError('Unknown mission configuration setting.')
    values = dict(values)
    if 'weights' in values:
        values['weights'] = store.path(values['weights'])
    else:
        values['weights'] = store.path('models/yolo26n.pt')
    for key in ('left_pins', 'right_pins'):
        if key in values:
            if not isinstance(values[key], (list, tuple)):
                raise ValueError('Motor pins must be lists: [IN1, IN2, PWM].')
            values[key] = tuple(values[key])
    for key in ('target_class', 'arduino_ip', 'mission_mode', 'tracker', 'device'):
        if key in values and not isinstance(values[key], str):
            raise ValueError(f'{key} must be text.')
    if values.get('phone_ip') == '':
        values['phone_ip'] = None
    cfg = Config(**values)
    cfg.validate()
    if not re.fullmatch(r'cpu|mps|\d+(,\d+)*', cfg.device):
        raise ValueError('Device must be cpu, mps, or GPU indices such as 0.')
    result = asdict(cfg)
    result['weights'] = store.relative(cfg.weights)
    return result


class Jobs:
    def __init__(self, store):
        self.store = store
        self.lock = store.lock
        self.processes = {}
        self.records = {}
        self.recovering = {}
        self.closed = False
        for file in (store.state/'jobs').glob('*/job.json'):
            record = json.loads(file.read_text(encoding='utf-8'))
            if record['status'] in ACTIVE:
                record.update(status='interrupted', ended=now(), message='Server restarted; job not automatically resumed. Check the robot before relaunching.')
                (file.parent/'stop').touch()
                write_json(file, record)
                if record.get('pid'):
                    self.recovering[record['id']] = record['pid']
            self.records[record['id']] = record
        self.heartbeat()

    def heartbeat(self):
        def pulse():
            while not self.closed:
                (self.store.state/'heartbeat').touch()
                time.sleep(1)
        threading.Thread(target=pulse, daemon=True).start()

    def directory(self, ticket):
        if ticket not in self.records:
            raise ValueError('Unknown job.')
        return self.store.state/'jobs'/ticket

    def active(self):
        return [r for r in self.records.values() if r['status'] in ACTIVE]

    def require_idle(self):
        self.require_recovered()
        if self.active():
            raise ValueError('Stop or finish active jobs before editing, renaming, or deleting project inputs.')

    def require_recovered(self):
        import psutil
        for ticket, pid in list(self.recovering.items()):
            try:
                process = psutil.Process(pid)
                command = process.cmdline()
                if ticket in command and 'webapp.worker' in command and process.is_running():
                    raise ValueError('A job from the previous server is still stopping. Wait for its cleanup before launching or editing inputs.')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.recovering.pop(ticket, None)

    def prepare(self, kind, params, directory):
        if kind not in JOBS or not isinstance(params, dict):
            raise ValueError('Unknown job type or invalid parameters.')
        if kind == 'mission':
            if set(params) - {'config', 'drive', 'video', 'max_batches', 'save_frames'}:
                raise ValueError('Unknown mission launch option.')
            config = config_values(self.store, params.get('config', {}))
            drive = params.get('drive', False)
            if type(drive) is not bool or type(params.get('save_frames', True)) is not bool:
                raise ValueError('Drive and save_frames must be booleans.')
            video = params.get('video') or ''
            if video:
                path = self.store.path(video, exists=True)
                if not path.is_file() or path.suffix.lower() not in VIDEOS:
                    raise ValueError('Select an existing video.')
            if drive and video:
                raise ValueError('Video replay cannot enable motors.')
            if config['mission_mode'] != 'tag_only' and config['object_provider'] == 'yolo':
                self.model(config['weights'])
                if Path(config['weights']).suffix.lower() != '.pt':
                    raise ValueError('Navigation currently requires local Ultralytics .pt weights; other formats can run inference jobs.')
            batches = params.get('max_batches', 0)
            if type(batches) is not int or not 0 <= batches <= 100000:
                raise ValueError('max_batches must be an integer from 0 to 100000.')
            return dict(config=config, drive=drive, video=video, max_batches=batches, save_frames=params.get('save_frames', True))
        schema = JOBS[kind]['fields']
        if set(params) - {f['key'] for f in schema}:
            raise ValueError('Unknown job parameter.')
        normalized = {}
        for field in schema:
            key = field['key']
            value = params.get(key, field['default'])
            if value is None or value == '':
                raise ValueError(f"{field['label']} is required.")
            if field['kind'] == 'number':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{key} must be a finite number.')
                if not field.get('min', -math.inf) <= value <= field.get('max', math.inf):
                    raise ValueError(f'{key} is outside the allowed range.')
                if field.get('integer', False) and type(value) is not int:
                    raise ValueError(f'{key} must be an integer.')
            elif not isinstance(value, str) or len(value) > 1000:
                raise ValueError(f'{key} must be text (1000 characters max).')
            if field['kind'] == 'select' and value not in field['options']:
                raise ValueError(f'Unknown option for {key}.')
            if field['kind'] == 'model':
                self.model(value)
            if field['kind'] in {'video', 'media', 'image', 'dataset', 'archive'}:
                path = self.store.path(value, exists=True)
                allowed = {'video': VIDEOS, 'image': IMAGES, 'media': IMAGES | VIDEOS, 'dataset': {'.yaml', '.yml'}, 'archive': {'.zip'}}[field['kind']]
                if not path.is_file() or path.suffix.lower() not in allowed:
                    raise ValueError(f'Invalid file type for {key}.')
            normalized[key] = value
        if 'device' in normalized and not re.fullmatch(r'cpu|mps|\d+(,\d+)*', normalized['device']):
            raise ValueError('Use cpu, mps or numeric GPU indices for device.')
        if 'output' in normalized:
            filename = name(normalized['output'])
            folder = 'models' if kind == 'train' else 'assets/tags' if kind == 'tag_generate' else 'datasets'
            if kind == 'train' and Path(filename).suffix != '.pt':
                raise ValueError('Training output must end in .pt.')
            if kind == 'tag_generate' and Path(filename).suffix.lower() != '.png':
                raise ValueError('Tag output must end in .png.')
            output = self.store.path(folder+'/'+filename)
            if output.exists():
                raise ValueError('Output already exists. Choose a new name.')
            normalized['output'] = self.store.relative(output)
        if kind == 'train':
            if Path(normalized['model']).suffix.lower() != '.pt':
                raise ValueError('Training requires .pt weights.')
            training_yaml(self.store, normalized['data'], directory/'dataset.yaml')
        if kind == 'roboflow' and not os.environ.get('ROBOFLOW_API_KEY'):
            raise ValueError('Set ROBOFLOW_API_KEY in the server environment and restart the server first.')
        return normalized

    def model(self, relative):
        path = self.store.path(relative, exists=True)
        if not path.is_relative_to(self.store.path('models')) or not path.is_file():
            raise ValueError('Select a model file in models/.')
        meta = self.store.model_metadata().get(relative, {})
        if meta.get('provider', 'ultralytics') != 'ultralytics':
            raise ValueError('This model provider has no executor for this job. Register one in webapp/catalog.py and worker.py.')
        if path.suffix.lower() not in MODEL_PROVIDERS[0]['extensions']:
            raise ValueError('This format is cataloged but has no Ultralytics executor.')
        return path

    def launch(self, kind, params):
        with self.lock:
            self.require_recovered()
            if kind not in JOBS:
                raise ValueError('Unknown job type.')
            if 'camera_robot' in JOBS[kind]['resources'] and JOBS[kind].get('stop') != 'cooperative':
                raise ValueError('Hardware executors must declare and implement cooperative stopping.')
            resources = set(JOBS[kind]['resources'])
            for active in self.active():
                if resources.intersection(JOBS[active['kind']]['resources']):
                    raise ValueError(f"Resources are in use by {active['label']}. Stop or finish it first.")
            ticket = uuid.uuid4().hex
            directory = self.store.state/'jobs'/ticket
            directory.mkdir()
            try:
                params = self.prepare(kind, params, directory)
                if params.get('output') and any(r['params'].get('output') == params['output'] for r in self.active()):
                    raise ValueError('Another job has reserved that output name.')
            except BaseException:
                import shutil
                shutil.rmtree(directory)
                raise
            record = dict(id=ticket, kind=kind, label=JOBS[kind]['label'], params=params,
                          status='starting', created=now(), result=None)
            self.records[ticket] = record
            write_json(directory/'job.json', record)
            environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
            try:
                with (directory/'output.log').open('wb') as log:
                    process = subprocess.Popen([sys.executable, '-m', 'webapp.worker', '--root', str(self.store.root), '--job', ticket],
                        cwd=self.store.root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except OSError as exc:
                record.update(status='failed', ended=now(), message=str(exc))
                write_json(directory/'job.json', record)
                raise ValueError('Could not start the job process.') from exc
            self.processes[ticket] = process
            record.update(status='running', started=now(), pid=process.pid)
            write_json(directory/'job.json', record)
            threading.Thread(target=self.monitor, args=(ticket, process), daemon=True).start()
            return dict(record)

    def monitor(self, ticket, process):
        code = process.wait()
        with self.lock:
            record = self.records[ticket]
            directory = self.directory(ticket)
            result = directory/'result.json'
            outcome = json.loads(result.read_text(encoding='utf-8')) if result.exists() else None
            stopped = record['status'] == 'stop_requested'
            record.update(status='stopped' if stopped else 'succeeded' if code == 0 else 'failed',
                          ended=now(), exit_code=code, result=outcome)
            if outcome and outcome.get('message'):
                record['message'] = outcome['message']
            if record['kind'] == 'inspect_model' and code == 0 and outcome:
                metadata = self.store.model_metadata()
                model = record['params']['model']
                metadata.setdefault(model, {}).update(inspection=outcome.get('data'))
                write_json(self.store.state/'models.json', metadata)
            write_json(directory/'job.json', record)
            self.processes.pop(ticket, None)

    def stop(self, ticket):
        with self.lock:
            directory = self.directory(ticket)
            record = self.records[ticket]
            if record['status'] not in ACTIVE:
                return dict(record)
            record.update(status='stop_requested', message='Stop requested; waiting for process cleanup.')
            (directory/'stop').touch()
            write_json(directory/'job.json', record)
            # Hardware jobs must run finally/zero-PWM cleanup. Never terminate them here.
            process = self.processes.get(ticket)
            if JOBS[record['kind']].get('stop') != 'cooperative' and process and process.poll() is None:
                process.terminate()
            return dict(record)

    def all(self):
        with self.lock:
            return sorted((dict(r) for r in self.records.values()), key=lambda r: r['created'], reverse=True)

    def detail(self, ticket):
        with self.lock:
            directory = self.directory(ticket)
            record = dict(self.records[ticket])
            log = directory/'output.log'
            if log.exists():
                with log.open('rb') as handle:
                    handle.seek(max(0, log.stat().st_size-60000))
                    record['log'] = handle.read().decode('utf-8', errors='replace')
            else:
                record['log'] = ''
            if record['kind'] == 'mission':
                from .mission_status import mission_status
                record['mission'] = mission_status(directory)
            from robotics_lab.snapshots import snapshot_exists
            record['preview'] = snapshot_exists(directory, 'preview.jpg')
            record['artifacts'] = [p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()
                                   and p.name not in {'stop', 'job.json', 'output.log', 'preview.tmp.jpg'}
                                   and not p.name.endswith('.tmp') and not p.is_symlink()
                                   and not p.is_relative_to(directory/'.display')][:1000]
            for snapshot in ('preview.jpg', 'camera.jpg', 'runtime.json'):
                if snapshot_exists(directory, snapshot) and snapshot not in record['artifacts']:
                    record['artifacts'].append(snapshot)
            return record

    def close(self):
        for record in self.active():
            self.stop(record['id'])
        self.closed = True
