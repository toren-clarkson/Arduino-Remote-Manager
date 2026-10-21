"""Contained project file operations and small, atomic JSON records."""

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TEXT = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.csv', '.ini', '.cfg', '.html', '.css', '.js'}
IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
BLOCKED = {'__pycache__', 'node_modules', 'build', 'dist'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def name(value):
    value = str(value).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_. -]{0,99}', value) or value.endswith(('.', ' ')):
        raise ValueError('Use a name starting with a letter/number, then letters, numbers, spaces, _, - or . (100 characters max).')
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *{f'COM{i}' for i in range(10)}, *{f'LPT{i}' for i in range(10)}}:
        raise ValueError('This filename is reserved by Windows.')
    return value


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state = self.root / '.control'
        self.lock = threading.RLock()
        for directory in ('jobs', 'profiles', 'trash', 'history'):
            (self.state / directory).mkdir(parents=True, exist_ok=True)

    def path(self, value='', *, exists=False):
        # Treat both separator styles consistently, including when tested off Windows.
        raw = str(value).replace('\\', '/')
        if ':' in raw or '\x00' in raw or Path(raw).is_absolute():
            raise ValueError('Use a path relative to Main.')
        parts = Path(raw).parts
        if any(p in {'.', '..'} or p.startswith('.') or p in BLOCKED for p in parts):
            raise ValueError('Hidden, generated, environment and parent paths are protected.')
        candidate = self.root.joinpath(*parts)
        for parent in (candidate, *candidate.parents):
            if parent == self.root:
                break
            if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
                raise ValueError('Symlinks and junctions are not managed by this app.')
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError('Path must stay within Main.')
        if resolved.suffix.lower() in {'.pem', '.key', '.pfx', '.p12'}:
            raise ValueError('Certificate/key files are protected.')
        if exists and not resolved.exists():
            raise ValueError('File or folder does not exist.')
        return resolved

    def relative(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def entry(self, path):
        stat = path.stat()
        return dict(name=path.name, path=self.relative(path), directory=path.is_dir(),
                    bytes=stat.st_size if path.is_file() else 0, modified=stat.st_mtime,
                    editable=path.is_file() and path.suffix.lower() in TEXT and stat.st_size <= 2_000_000,
                    image=path.suffix.lower() in IMAGES, video=path.suffix.lower() in VIDEOS)

    def listing(self, relative=''):
        directory = self.path(relative, exists=True)
        if not directory.is_dir():
            raise ValueError('Choose a directory.')
        result = []
        for child in directory.iterdir():
            try:
                child = self.path(self.relative(child), exists=True)
                result.append(self.entry(child))
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda x: (not x['directory'], x['name'].casefold()))

    def discover(self, folder, extensions):
        base = self.path(folder)
        if not base.exists():
            return []
        results = []
        for directory, children, files in os.walk(base, followlinks=False):
            children[:] = [p for p in children if not p.startswith('.') and p not in BLOCKED
                           and not Path(directory, p).is_symlink()
                           and not (hasattr(Path(directory, p), 'is_junction') and Path(directory, p).is_junction())]
            for filename in files:
                if Path(filename).suffix.lower() in extensions:
                    try:
                        path = self.path(self.relative(Path(directory, filename)), exists=True)
                        results.append(self.entry(path))
                    except (ValueError, OSError):
                        continue
        return sorted(results, key=lambda x: x['path'])

    def text(self, relative):
        path = self.path(relative, exists=True)
        if path.suffix.lower() not in TEXT or path.stat().st_size > 2_000_000:
            raise ValueError('Text editing supports known text formats up to 2 MB.')
        data = path.read_bytes()
        return dict(path=self.relative(path), text=data.decode('utf-8-sig'), revision=hashlib.sha256(data).hexdigest())

    def save_text(self, relative, text, revision):
        with self.lock:
            current = self.text(relative)
            if current['revision'] != revision:
                raise ValueError('File changed since you opened it. Reload before saving.')
            if len(text.encode('utf-8')) > 2_000_000:
                raise ValueError('Text is too large.')
            path = self.path(relative)
            shutil.copy2(path, self.state / 'history' / (uuid.uuid4().hex + path.suffix))
            temporary = path.with_name('.' + path.name + '.control-tmp')
            temporary.write_text(text, encoding='utf-8')
            os.replace(temporary, path)
            return self.text(relative)

    def trash(self, relative):
        path = self.path(relative, exists=True)
        if path == self.root or len(path.relative_to(self.root).parts) == 1 and path.is_dir():
            raise ValueError('Top-level project folders cannot be trashed.')
        ticket = uuid.uuid4().hex
        directory = self.state / 'trash' / ticket
        directory.mkdir()
        write_json(directory / 'record.json', dict(id=ticket, path=self.relative(path), time=now()))
        shutil.move(str(path), str(directory / 'payload'))
        return ticket

    def trash_list(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in (self.state/'trash').glob('*/record.json')
                if (p.parent/'payload').exists()]

    def restore(self, ticket):
        if not re.fullmatch('[a-f0-9]{32}', ticket):
            raise ValueError('Invalid trash record.')
        source = self.state/'trash'/ticket
        record = json.loads((source/'record.json').read_text(encoding='utf-8'))
        path = self.path(record['path'])
        if path.exists():
            raise ValueError('Destination already exists; rename it first.')
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source/'payload'), str(path))
        return record['path']

    def model_metadata(self):
        path = self.state/'models.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def update_model(self, path, label, notes):
        with self.lock:
            metadata = self.model_metadata()
            metadata[path] = dict(label=str(label)[:100], notes=str(notes)[:4000])
            write_json(self.state/'models.json', metadata)


def extract_dataset(archive_path, output):
    """Bounded, all-or-nothing import of a local YOLO export ZIP."""
    if output.exists():
        raise ValueError('Choose a new dataset directory.')
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > 50000 or sum(i.file_size for i in infos) > 4*1024**3:
            raise ValueError('Archive exceeds 50,000 entries or 4 GB uncompressed.')
        targets = set()
        for item in infos:
            raw = item.filename.replace('\\', '/')
            parts = Path(raw).parts
            target = (output/raw).resolve()
            if (not parts or any(p.startswith('.') or p == '..' or ':' in p for p in parts)
                    or not target.is_relative_to(output.resolve())
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe path in dataset ZIP.')
            for part in parts:
                name(part)
            if not item.is_dir() and target.suffix.lower() not in IMAGES | {'.txt', '.yaml', '.yml', '.json', '.md', '.csv'}:
                raise ValueError('Dataset ZIP may contain images, labels, YAML, JSON, Markdown and CSV only.')
            key = str(target).casefold()
            if key in targets:
                raise ValueError('Duplicate filename in dataset ZIP.')
            targets.add(key)
        output.mkdir(parents=True)
        try:
            for item in infos:
                target = output/item.filename.replace('\\', '/')
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open('xb') as destination:
                        shutil.copyfileobj(source, destination)
        except BaseException:
            shutil.rmtree(output)
            raise


def training_yaml(store, relative, output):
    """Normalize local YOLO data paths; never allow dataset download scripts/URLs."""
    import yaml
    source = store.path(relative, exists=True)
    datasets = store.path('datasets')
    if source.suffix.lower() not in {'.yaml', '.yml'} or not source.is_relative_to(datasets):
        raise ValueError('Select a YAML file inside datasets/.')
    try:
        data = yaml.safe_load(source.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError('Invalid dataset YAML: '+str(exc)) from None
    if not isinstance(data, dict):
        raise ValueError('Dataset YAML must contain a mapping of settings.')
    allowed = {'path', 'train', 'val', 'test', 'names', 'nc', 'channels', 'kpt_shape', 'flip_idx', 'roboflow'}
    unknown = set(data) - allowed
    if unknown:
        settings = ', '.join(sorted(map(str, unknown)))
        raise ValueError(f'Unsupported dataset YAML setting(s): {settings}. Download hooks and executable settings are not allowed.')
    # Export provenance is not a training setting. Keep it in the user's original
    # file, but omit it from the job's normalized YAML. Never execute/fetch it.
    data.pop('roboflow', None)
    if not isinstance(data.get('names'), (dict, list)) or not data['names']:
        raise ValueError('Dataset YAML must define class names.')
    for directory, children, files in os.walk(source.parent, followlinks=False):
        for child in children + files:
            store.path(store.relative(Path(directory, child)), exists=True)
    base_value = str(data.get('path') or '.')
    if '://' in base_value:
        raise ValueError('Remote dataset paths are not supported.')
    base = (source.parent/base_value).resolve()
    for split in ('train', 'val', 'test'):
        value = data.get(split)
        if not value:
            if split in {'train', 'val'}:
                raise ValueError('Dataset needs train and val image directories.')
            continue
        is_list = isinstance(value, list)
        values = value if is_list else [value]
        normalized = []
        for value in values:
            if not isinstance(value, str) or '://' in value:
                raise ValueError('Dataset splits must be local directories.')
            value = value.replace('\\', '/')
            candidate = (base/value).resolve()
            # Some Roboflow ZIPs place data.yaml next to train/valid/test while
            # exporting ../train/images etc. Match Ultralytics' missing-path
            # fallback for one leading ../, only without an explicit dataset root.
            # Existing sibling-directory references keep their original meaning.
            if not candidate.is_dir() and not data.get('path') and value.startswith('../'):
                local = (source.parent/value[3:]).resolve()
                if local.is_relative_to(source.parent) and local.is_dir():
                    candidate = local
            if not candidate.is_relative_to(datasets) or not candidate.is_dir():
                raise ValueError(f'{split}: image directory {value!r} was not found inside datasets/. Check paths relative to {store.relative(source.parent)}/.')
            store.path(store.relative(candidate), exists=True)
            # Ultralytics follows images/labels below these paths. Refuse link escapes.
            for directory, children, files in os.walk(candidate, followlinks=False):
                for child in children + files:
                    store.path(store.relative(Path(directory, child)), exists=True)
            normalized.append(str(candidate))
        data[split] = normalized if is_list else normalized[0]
    data['path'] = str(source.parent)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return data
