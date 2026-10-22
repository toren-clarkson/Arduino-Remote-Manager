"""Read-only mission display data, including connection links from older logs."""

import ipaddress
import json
import re
from urllib.parse import urlsplit
from robotics_lab.snapshots import read_snapshot, snapshot_exists


def phone_links(text):
    result = []
    for value in re.findall(r'https://[^\s<>"\x1b]+/phone\b', text):
        try:
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.path != '/phone' or not parsed.port:
                continue
            if parsed.hostname != 'localhost':
                ipaddress.ip_address(parsed.hostname)
        except ValueError:
            continue
        if value not in result:
            result.append(value)
    return result


def mission_status(directory):
    runtime = {}
    if snapshot_exists(directory, 'runtime.json'):
        try:
            runtime = json.loads(read_snapshot(directory, 'runtime.json'))
        except (OSError, ValueError):
            pass
    # Keep initial PhoneSense URLs even when they have scrolled out of the tail.
    log = directory/'output.log'
    text = ''
    if log.exists():
        with log.open('rb') as handle:
            text = handle.read(64000).decode('utf-8', errors='replace')
    runtime['phone_urls'] = phone_links(text+'\n'+'\n'.join(runtime.get('phone_urls', [])))
    runtime['camera_available'] = snapshot_exists(directory, 'camera.jpg')
    runtime['observation_available'] = snapshot_exists(directory, 'preview.jpg')
    runtime.pop('preview_ipc', None)  # worker-to-server transport details are not UI data
    return runtime
