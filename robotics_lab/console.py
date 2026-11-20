"""Plain terminal settings for standalone programs. No services or hardware here."""

import json
import math
import sys
from pathlib import Path

from robotics_lab.paths import PROJECT_ROOT, MODELS


def ask(label, default, convert=str, validate=lambda value: True):
    while True:
        raw = input(f'{label} [{default}]: ').strip()
        try:
            value = default if not raw else convert(raw)
            if not validate(value):
                raise ValueError('outside the allowed range')
            return value
        except (ValueError, TypeError) as exc:
            print(f'Please try again: {exc}')


def choose(label, values, default=0):
    if not values:
        raise ValueError(f'No choices available for {label}')
    print(f'\n{label}')
    for i, value in enumerate(values, 1):
        print(f'  {i}. {value}')
    return ask('Choice', default+1, int, lambda n: type(n) is int and 1 <= n <= len(values))-1


def weights_menu(default):
    paths = sorted(MODELS.glob('*.pt'))
    default = Path(default).resolve()
    if default not in paths and default.is_file():
        paths.insert(0, default)
    index = choose('YOLO detection weights', [p.name for p in paths]+['Enter another local path'],
                   paths.index(default) if default in paths else 0)
    if index == len(paths):
        path = ask('Weights path', str(default), str, lambda s: Path(s).is_file())
        return Path(path).resolve()
    return paths[index]


def class_menu(model, default='Red-Ball', allow_any=False):
    names = list(model.names.values())
    options = names + (['Any class (largest object)'] if allow_any else [])
    index = choose('Object class inside these YOLO weights', options,
                   names.index(default) if default in names else 0)
    return names[index] if index < len(names) else None


def pins(raw):
    values = tuple(int(part.strip()) for part in raw.split(','))
    if len(values) != 3 or len(set(values)) != 3 or any(not 0 <= p <= 13 for p in values):
        raise ValueError('Use IN1,IN2,PWM, three distinct UNO pins from 0 to 13')
    if values[2] not in (3, 5, 6, 9, 10, 11):
        raise ValueError('The third pin must support PWM: 3,5,6,9,10,11')
    return values


def configure_script(namespace):
    """Called only in __main__. --setup keeps all existing direct CLI usage valid."""
    if '--setup' not in sys.argv:
        return
    sys.argv.remove('--setup')
    script = Path(namespace['__file__']).stem
    saved_path = PROJECT_ROOT/'calibrations'/'console'/f'{script}.json'
    try:
        saved = json.loads(saved_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    print(f'\n{script}: Enter keeps the displayed setting. Ctrl+C cancels.\n')
    settings = {}
    model_key = next((k for k in ('WEIGHTS', 'MODEL_PATH') if k in namespace), None)
    if model_key:
        path = weights_menu(saved.get(model_key, namespace[model_key]))
        from ultralytics import YOLO
        model = YOLO(str(path))
        if model.task != 'detect':
            raise ValueError('Choose object-detection weights')
        namespace[model_key] = path
        namespace['TARGET_CLASS'] = class_menu(model, saved.get('TARGET_CLASS', namespace.get('TARGET_CLASS')),
                                               allow_any=script == 'default-yolo-tracking')
        settings.update({model_key: str(path), 'TARGET_CLASS': namespace['TARGET_CLASS']})
    tag_key = next((k for k in ('TAG_ID', 'TARGET_TAG_ID') if k in namespace), None)
    if tag_key:
        value = ask('AprilTag 36h11 ID (0..586)', saved.get(tag_key, namespace[tag_key]) or 0,
                    int, lambda n: type(n) is int and 0 <= n <= 586)
        namespace[tag_key] = settings[tag_key] = value
    if 'DETECTOR_BACKEND' in namespace:
        options = ['opencv', 'apriltag3']
        namespace['DETECTOR_BACKEND'] = settings['DETECTOR_BACKEND'] = options[
            choose('AprilTag detector', options)]
    if 'LEFT_MOTOR' in namespace:
        for key in ('LEFT_MOTOR', 'RIGHT_MOTOR'):
            current = saved.get(key, list(namespace[key]))
            default = ','.join(map(str, current))
            selected = pins(ask(f'{key} IN1,IN2,PWM', default, str,
                                 lambda s: bool(pins(s))))
            original = namespace[key]
            namespace[key] = type(original)(*selected) if hasattr(original, '_fields') else selected
            settings[key] = list(selected)
        if set(namespace['LEFT_MOTOR']) & set(namespace['RIGHT_MOTOR']):
            raise ValueError('Left and right motors cannot share pins')
    fields = {
        'MIN_PWM': ('Starting motor PWM (0..220)', 0, 220),
        'BASE_SPEED': ('Base speed (0..1)', 0, 1),
        'FORWARD_SPEED': ('Forward speed (0..1)', 0, 1),
        'TURN_SPEED': ('Turn speed (0..1)', 0, 1),
        'SEARCH_SPEED': ('Search speed (0..1)', 0, 1),
        'FORWARD_HOLD': ('Forward hold seconds', .02, 2),
        'TURN_HOLD': ('Turn hold seconds', .02, 2),
        'SETTLE_SECONDS': ('Camera settling seconds', .1, 5),
        'OBJECT_STOP_AREA': ('Ball arrival: box area fraction', .001, .8),
        'TAG_STOP_WIDTH': ('Tag arrival: edge/image width', .01, .9),
        'TAG_STOP_FRACTION': ('Tag arrival: edge/image width', .01, .9),
        'COLOR_STOP_RADIUS': ('Color arrival radius in pixels', 1, 2000),
        'TAG_STOP_SIZE': ('Tag arrival size in pixels', 1, 2000),
        'IMAGE_SIZE': ('YOLO image size', 160, 1280),
        'CONFIDENCE': ('YOLO confidence', .1, .95),
        'INFERENCE_TIMEOUT': ('Stationary inference timeout seconds', .1, 60),
        'BLINK_COUNT': ('Blink count', 1, 1000),
        'BLINK_SECONDS': ('Blink interval in seconds', .05, 10),
    }
    for key, (label, low, high) in fields.items():
        if key not in namespace:
            continue
        convert = int if type(namespace[key]) is int else float
        value = ask(label, saved.get(key, namespace[key]), convert,
                    lambda n: isinstance(n, (int, float)) and math.isfinite(n) and low <= n <= high)
        namespace[key] = settings[key] = value
    # Old scripts that drive by default offer preview from the same terminal menu.
    if script in ('robust-april-tag', 'default-yolo-tracking'):
        if choose('Run mode', ['Camera preview', 'Physical robot']) == 0:
            if '--camera-only' not in sys.argv:
                sys.argv.append('--camera-only')
    elif script == 'stop-and-stare':
        if '--drive' not in sys.argv and choose('Run mode', ['Camera preview', 'Physical robot']) == 1:
            sys.argv.append('--drive')
    elif script not in ('test_wifi_led',) :
        print('This legacy experiment uses its existing hardware behavior. The adaptive mission has preview/calibration modes.')
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(json.dumps(settings, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Settings saved for the next --setup run. Explicit CLI arguments take precedence.\n')
