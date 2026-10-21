"""Loopback-only project workbench API and HTML shell."""

import atexit
import hmac
import io
import json
import math
import os
import secrets
import shutil
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, render_template, request, send_file, session
from werkzeug.exceptions import HTTPException

from missions.stop_and_stare.config import Config
from missions.stop_and_stare.components import OPTIONS
from robotics_lab.paths import PROJECT_ROOT
from robotics_lab.snapshots import read_snapshot, snapshot_path, snapshot_exists
from .catalog import MODEL_PROVIDERS, catalog
from .help import CONFIG_HELP
from .jobs import Jobs, config_values
from .storage import IMAGES, TEXT, VIDEOS, Store, name, write_json

MODEL_EXTENSIONS = {extension for provider in MODEL_PROVIDERS for extension in provider['extensions']}


def create_app(root=PROJECT_ROOT, *, testing=False):
    app = Flask(__name__)
    app.config.update(SECRET_KEY=secrets.token_hex(32), MAX_CONTENT_LENGTH=2*1024**3,
                      SESSION_COOKIE_SAMESITE='Strict', SESSION_COOKIE_HTTPONLY=True,
                      TESTING=testing)
    store = Store(root)
    jobs = Jobs(store)
    app.extensions.update(store=store, jobs=jobs)
    atexit.register(jobs.close)
    camera_slots = threading.BoundedSemaphore(3)

    @app.before_request
    def local_only():
        if urlsplit('http://'+request.host).hostname not in {'localhost', '127.0.0.1', '::1'}:
            return jsonify(error='Use the localhost address.'), 403
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('Origin')
            if origin and origin != request.host_url.rstrip('/'):
                return jsonify(error='Cross-origin changes are blocked.'), 403
            expected = session.get('csrf', '')
            token = request.headers.get('X-CSRF-Token', '')
            if not expected or not hmac.compare_digest(expected, token):
                return jsonify(error='Session token missing. Reload the page.'), 403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(ValueError)
    @app.errorhandler(OSError)
    @app.errorhandler(UnicodeError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    def body():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object.')
        return data

    @app.get('/')
    @app.get('/datasets')
    @app.get('/models')
    @app.get('/training')
    @app.get('/tags')
    @app.get('/missions')
    @app.get('/live')
    @app.get('/jobs')
    @app.get('/files')
    @app.get('/settings')
    def index():
        session.setdefault('csrf', secrets.token_hex(32))
        return render_template('index.html', csrf=session['csrf'])

    @app.get('/api/session')
    def session_info():
        session.setdefault('csrf', secrets.token_hex(32))
        return jsonify(csrf=session['csrf'])

    @app.get('/api/state')
    def state():
        with store.lock:
            metadata = store.model_metadata()
            models = store.discover('models', MODEL_EXTENSIONS)
            for model in models:
                model.update(metadata.get(model['path'], {}))
                model.setdefault('provider', 'ultralytics' if Path(model['path']).suffix.lower() in MODEL_PROVIDERS[0]['extensions'] else 'generic')
            defaults = asdict(Config())
            defaults['weights'] = 'models/yolo26n.pt'
            schema = []
            for field in fields(Config):
                default = defaults[field.name]
                schema.append(dict(key=field.name, default=default,
                                   help=CONFIG_HELP[field.name],
                                   kind='pins' if field.name.endswith('_pins') else 'number' if isinstance(default, (int, float)) else 'text',
                                   options=list(OPTIONS[field.name]) if field.name in OPTIONS else None,
                                   integer=type(default) is int))
            profiles = [dict(id=p.stem, **json.loads(p.read_text(encoding='utf-8')))
                        for p in (store.state/'profiles').glob('*.json')]
            return jsonify(root=str(store.root), folders=[e for e in store.listing() if e['directory']],
                           models=models, datasets=store.discover('datasets', {'.yaml', '.yml'}),
                           media=store.discover('datasets', IMAGES | VIDEOS | {'.zip'}) + store.discover('assets', IMAGES | VIDEOS) + store.discover('runs/verification', IMAGES | VIDEOS),
                           tags=store.discover('assets/tags', IMAGES), jobs=jobs.all(),
                           catalog=catalog(), defaults=defaults, config_schema=schema, profiles=profiles,
                           runtime=dict(python=os.sys.version.split()[0], executable=os.sys.executable,
                                        roboflow_key=bool(os.environ.get('ROBOFLOW_API_KEY'))))

    @app.get('/api/files')
    def files():
        relative = request.args.get('path', '')
        return jsonify(path=relative, entries=store.listing(relative))

    @app.get('/api/text')
    def text():
        return jsonify(store.text(request.args.get('path', '')))

    @app.put('/api/text')
    def edit_text():
        data = body()
        if not isinstance(data.get('text'), str):
            raise ValueError('Provide text.')
        with store.lock:
            jobs.require_idle()
            return jsonify(store.save_text(data['path'], data['text'], data.get('revision')))

    @app.get('/api/file')
    def file_download():
        path = store.path(request.args.get('path', ''), exists=True)
        if not path.is_file():
            raise ValueError('Choose a file.')
        inline = request.args.get('inline') == '1' and path.suffix.lower() in IMAGES | VIDEOS
        return send_file(path, as_attachment=not inline, download_name=path.name, conditional=True)

    @app.post('/api/folders')
    def create_folder():
        data = body()
        with store.lock:
            parent = store.path(data.get('parent', ''), exists=True)
            path = store.path(store.relative(parent/name(data['name'])))
            path.mkdir()
            return jsonify(path=store.relative(path))

    @app.post('/api/new-file')
    def new_file():
        data = body()
        with store.lock:
            jobs.require_idle()
            parent = store.path(data.get('parent', ''), exists=True)
            path = store.path(store.relative(parent/name(data['name'])))
            if path.suffix.lower() not in TEXT:
                raise ValueError('Choose a supported text extension, such as .md, .py, .json or .yaml.')
            path.open('x', encoding='utf-8').close()
            return jsonify(path=store.relative(path))

    @app.post('/api/upload')
    def upload():
        folder = store.path(request.form.get('folder', 'datasets/uploads'))
        files = request.files.getlist('files')
        if not files or len(files) > 500:
            raise ValueError('Upload 1–500 files at a time.')
        if folder == store.root:
            raise ValueError('Choose a project subfolder for uploads.')
        with store.lock:
            jobs.require_idle()
            destinations = []
            for item in files:
                filename = name(item.filename)
                path = store.path(store.relative(folder/filename))
                if path.suffix.lower() not in TEXT | IMAGES | VIDEOS | MODEL_EXTENSIONS | {'.zip'}:
                    raise ValueError('Unsupported upload format.')
                if path.exists() or path in destinations:
                    raise ValueError(f'{filename} already exists. Rename your file or choose a new folder.')
                if path.is_relative_to(store.path('models')) and path.suffix.lower() not in MODEL_EXTENSIONS:
                    raise ValueError('Upload model weights to models/; put other files in datasets/ or Files.')
                destinations.append(path)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = []
            committed = []
            try:
                for item, destination in zip(files, destinations):
                    temp = destination.with_name('.upload-'+secrets.token_hex(12))
                    temporary.append(temp)
                    item.save(temp)
                for temp, destination in zip(temporary, destinations):
                    # Exclusive create also protects against another local tool creating a file.
                    with temp.open('rb') as source, destination.open('xb') as target:
                        committed.append(destination)
                        shutil.copyfileobj(source, target)
            except BaseException:
                for path in committed:
                    path.unlink(missing_ok=True)
                raise
            finally:
                for temp in temporary:
                    temp.unlink(missing_ok=True)
            return jsonify(files=[store.relative(p) for p in destinations])

    @app.post('/api/rename')
    def rename():
        data = body()
        with store.lock:
            jobs.require_idle()
            source = store.path(data['path'], exists=True)
            if source == store.root or source.is_dir() and len(source.relative_to(store.root).parts) == 1:
                raise ValueError('Top-level project folders cannot be renamed.')
            destination = store.path(store.relative(source.with_name(name(data['name']))))
            if destination.exists():
                raise ValueError('Destination already exists.')
            source.rename(destination)
            metadata = store.model_metadata()
            if data['path'] in metadata:
                metadata[store.relative(destination)] = metadata.pop(data['path'])
                write_json(store.state/'models.json', metadata)
            return jsonify(path=store.relative(destination))

    @app.post('/api/trash')
    def trash():
        data = body()
        with store.lock:
            jobs.require_idle()
            return jsonify(id=store.trash(data['path']))

    @app.get('/api/trash')
    def trash_list():
        return jsonify(records=store.trash_list())

    @app.post('/api/restore')
    def restore():
        data = body()
        with store.lock:
            jobs.require_idle()
            return jsonify(path=store.restore(data['id']))

    @app.put('/api/models')
    def model_metadata():
        data = body()
        path = store.path(data['path'], exists=True)
        if not path.is_relative_to(store.path('models')) or path.suffix.lower() not in MODEL_EXTENSIONS:
            raise ValueError('Select a model in models/.')
        if data.get('provider', 'ultralytics') not in {p['id'] for p in MODEL_PROVIDERS}:
            raise ValueError('Unknown model provider.')
        with store.lock:
            jobs.require_idle()
            metadata = store.model_metadata()
            metadata.setdefault(data['path'], {}).update(label=str(data.get('label', path.stem))[:100],
                notes=str(data.get('notes', ''))[:4000], provider=data.get('provider', 'ultralytics'))
            write_json(store.state/'models.json', metadata)
        return jsonify(ok=True)

    @app.post('/api/profiles')
    def save_profile():
        data = body()
        config = config_values(store, data.get('config', {}))
        profile_name = name(data['name'])
        write_json(store.state/'profiles'/(profile_name+'.json'), dict(name=profile_name, config=config))
        return jsonify(id=profile_name)

    @app.delete('/api/profiles/<profile_id>')
    def delete_profile(profile_id):
        path = store.state/'profiles'/(name(profile_id)+'.json')
        path.unlink(missing_ok=True)
        return jsonify(ok=True)

    @app.get('/api/jobs')
    def job_list():
        return jsonify(jobs=jobs.all())

    @app.get('/print-tag')
    def print_tag():
        path = store.path(request.args.get('path', ''), exists=True)
        if not path.is_relative_to(store.path('assets/tags')) or path.suffix.lower() not in IMAGES:
            raise ValueError('Choose a marker image from assets/tags/.')
        return render_template('print.html', path=store.relative(path), title=path.name)

    @app.post('/api/jobs')
    def launch_job():
        data = body()
        return jsonify(jobs.launch(data.get('kind'), data.get('params', {}))), 201

    @app.get('/api/jobs/<ticket>')
    def job_detail(ticket):
        return jsonify(jobs.detail(ticket))

    @app.get('/api/jobs/<ticket>/camera')
    def mission_camera(ticket):
        directory = jobs.directory(ticket)
        if jobs.records[ticket]['kind'] != 'mission':
            raise ValueError('Only missions have a live camera stream.')
        if not camera_slots.acquire(blocking=False):
            return jsonify(error='Three camera viewers are already connected. Close an unused monitor tab.'), 429
        released = False
        release_lock = threading.Lock()
        def release():
            nonlocal released
            with release_lock:
                if not released:
                    released = True
                    camera_slots.release()
        def stream():
            last_modified = None
            try:
                while True:
                    path = snapshot_path(directory, 'camera.jpg')
                    try:
                        if path is None:
                            raise FileNotFoundError('No camera frame yet')
                        modified = path.stat().st_mtime_ns
                        if modified != last_modified:
                            jpeg = read_snapshot(directory, 'camera.jpg')
                            last_modified = modified
                            yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
                    except OSError:
                        pass
                    if jobs.records[ticket]['status'] not in {'starting', 'running', 'stop_requested'}:
                        break
                    time.sleep(.2)
                yield b'--frame--\r\n'
            finally:
                release()
        response = Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        response.call_on_close(release)  # also handles HEAD / an unconsumed iterator
        return response

    @app.get('/api/jobs/<ticket>/observation')
    def mission_observation(ticket):
        directory = jobs.directory(ticket)
        if not snapshot_exists(directory, 'preview.jpg'):
            return jsonify(error='No stationary observation yet.'), 404
        return Response(read_snapshot(directory, 'preview.jpg'), mimetype='image/jpeg')

    def color_mission(ticket):
        directory = jobs.directory(ticket)
        record = jobs.records[ticket]
        config = record.get('params', {}).get('config', {})
        if (record['kind'] != 'mission' or record['status'] not in {'starting', 'running'}
                or config.get('object_provider') != 'hsv' or config.get('mission_mode') == 'tag_only'):
            raise ValueError('Choose a running OpenCV HSV object mission.')
        from .mission_status import mission_status
        if mission_status(directory).get('leg') in {'tag', 'done'}:
            raise ValueError('The object leg has ended.')
        return directory

    @app.post('/api/jobs/<ticket>/color-frame')
    def freeze_color_frame(ticket):
        with store.lock:
            directory = color_mission(ticket)
            jpeg = read_snapshot(directory, 'camera.jpg')
            folder = directory/'.color-picker'
            folder.mkdir(exist_ok=True)
            token = secrets.token_hex(16)
            (folder/(token+'.jpg')).write_bytes(jpeg)
            # Retain a few pinned images so browser polling cannot change the sample.
            for old in sorted(folder.glob('*.jpg'), key=lambda p:p.stat().st_mtime_ns)[:-4]:
                try:
                    old.unlink()
                except OSError:
                    pass
            return jsonify(token=token, image=f'/api/jobs/{ticket}/color-frame/{token}')

    def pinned_frame(ticket, token):
        if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('Invalid color frame token')
        path = jobs.directory(ticket)/'.color-picker'/(token+'.jpg')
        if not path.is_file() or time.time()-path.stat().st_mtime > 120:
            raise ValueError('Color frame expired. Freeze a new camera frame.')
        return path.read_bytes()

    @app.get('/api/jobs/<ticket>/color-frame/<token>')
    def color_frame(ticket, token):
        return Response(pinned_frame(ticket, token), mimetype='image/jpeg')

    @app.post('/api/jobs/<ticket>/color')
    def select_color(ticket):
        from missions.stop_and_stare.color import sample_hsv
        data = body()
        for key in ('x', 'y'):
            value = data.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Choose image coordinates between 0 and 1.')
        with store.lock:
            directory = color_mission(ticket)
            hsv = sample_hsv(pinned_frame(ticket, data.get('token')), data['x'], data['y'])
            token = secrets.token_hex(16)
            command = dict(id=token, hsv=hsv)
            write_json(directory/f'.color-request-{time.time_ns():020d}-{token}.json', command)
            for old in sorted(directory.glob('.color-request-*.json'))[:-4]:
                try:
                    old.unlink()
                except OSError:
                    pass
            return jsonify(hsv=hsv, request=token, message='Color queued; applied while stopped before a fresh observation.')

    @app.post('/api/jobs/<ticket>/stop')
    def stop_job(ticket):
        return jsonify(jobs.stop(ticket))

    @app.get('/api/jobs/<ticket>/artifact')
    def artifact(ticket):
        directory = jobs.directory(ticket)
        relative = request.args.get('name', '')
        if relative in {'preview.jpg', 'camera.jpg', 'runtime.json'}:
            inline = request.args.get('inline') == '1' and relative.endswith('.jpg')
            return send_file(io.BytesIO(read_snapshot(directory, relative)), as_attachment=not inline,
                             download_name=relative)
        if not relative or any(p.startswith('.') or p == '..' for p in Path(relative.replace('\\', '/')).parts) or ':' in relative:
            raise ValueError('Invalid artifact name.')
        path = (directory/relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.is_symlink():
            raise ValueError('Unknown artifact.')
        inline = request.args.get('inline') == '1' and path.suffix.lower() in IMAGES | VIDEOS
        return send_file(path, as_attachment=not inline, conditional=True)

    @app.post('/api/jobs/<ticket>/copy-artifact')
    def copy_artifact(ticket):
        data = body()
        with store.lock:
            jobs.require_idle()
            directory = jobs.directory(ticket)
            source = (directory/name(data['name'])).resolve()
            if not source.is_relative_to(directory) or source.suffix.lower() not in IMAGES | VIDEOS or not source.is_file():
                raise ValueError('Choose an image/video artifact.')
            destination = store.path('datasets/job-outputs/'+name(data.get('output', source.name)))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as handle, destination.open('xb') as target:
                shutil.copyfileobj(handle, target)
            return jsonify(path=store.relative(destination))

    return app
