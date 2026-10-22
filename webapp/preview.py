"""Connect only to a worker's authenticated loopback JPEG service."""

import json
from urllib.request import Request, urlopen

from robotics_lab.snapshots import read_snapshot


def open_preview(directory, path='stream', channel='camera'):
    data = json.loads(read_snapshot(directory, 'runtime.json')).get('preview_ipc', {})
    port, token = data.get('port'), data.get('token', '')
    if (type(port) is not int or not 1024 <= port <= 65535 or not isinstance(token, str)
            or len(token) != 64 or any(c not in '0123456789abcdef' for c in token)):
        raise ValueError('No live worker stream')
    if path not in {'frame', 'stream'} or channel not in {'camera', 'observation'}:
        raise ValueError('Invalid preview channel')
    return urlopen(Request(f'http://127.0.0.1:{port}/{path}?channel={channel}',
                           headers={'X-Preview-Token': token}), timeout=3)


def latest_frame(directory, channel='camera'):
    try:
        with open_preview(directory, 'frame', channel) as response:
            return response.read()
    except (OSError, ValueError):
        return read_snapshot(directory, 'camera.jpg' if channel == 'camera' else 'preview.jpg')
