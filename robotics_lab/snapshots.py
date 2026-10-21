"""Immutable, bounded monitor snapshots: never replace a file held by a reader."""

import os
import time
import uuid
from pathlib import Path

NAMES = {'camera.jpg', 'preview.jpg', 'runtime.json'}


def snapshot_path(directory, name):
    if name not in NAMES:
        raise ValueError('Unknown snapshot name')
    name = Path(name)
    versions = (directory/'.display').glob(name.stem+'_*'+name.suffix)
    latest = max(versions, default=None)
    if latest is not None:
        return latest
    legacy = directory/name
    return legacy if legacy.is_file() else None


def snapshot_exists(directory, name):
    return snapshot_path(directory, name) is not None


def read_snapshot(directory, name):
    # Retention cleanup can remove an old version between selection and open.
    for _ in range(3):
        path = snapshot_path(directory, name)
        if path is None:
            break
        try:
            return path.read_bytes()
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f'No {name} snapshot yet')


def publish_snapshot(directory, name, data):
    if name not in NAMES:
        raise ValueError('Unknown snapshot name')
    folder = directory/'.display'
    folder.mkdir(exist_ok=True)
    name = Path(name)
    published = folder/f'{name.stem}_{time.time_ns():020d}_{uuid.uuid4().hex}{name.suffix}'
    temporary = published.with_suffix('.tmp')
    try:
        temporary.write_bytes(data)
        # The destination is unique and has never been exposed to readers.
        for attempt in range(4):
            try:
                os.replace(temporary, published)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.01)  # this function is called by the display thread only
    finally:
        temporary.unlink(missing_ok=True)
    for old in sorted(folder.glob(name.stem+'_*'+name.suffix))[:-4]:
        try:
            old.unlink()
        except OSError:
            pass  # Windows may still have a reader; retry cleanup next publication.
    return published
