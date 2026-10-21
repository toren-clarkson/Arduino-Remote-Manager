"""Trusted component factories. Browser settings select names, never Python imports.

Register extensions here so both the web server and worker load the same registry.
Factories are lazy: listing settings must not load models or connect hardware.
"""

from importlib import import_module


def factory(module, name):
    def build(*args, **kwargs):
        return getattr(import_module(module, package=__package__), name)(*args, **kwargs)
    return build


OBJECT_PROVIDERS = {
    'yolo': factory('.perception', 'YOLOObjects'),
    'hsv': factory('.color', 'HSVObjects'),
}
STEERING = {
    'timed': factory('.steering', 'TimedSteering'),
    'pid': factory('.steering', 'PIDSteering'),
}
ESTIMATORS = {
    'none': factory('.estimation', 'RawPosition'),
    'kalman': factory('.estimation', 'KalmanPosition'),
}
MOTION = {'ramped': factory('.motion', 'PulseDriver')}
OPTIONS = dict(object_provider=OBJECT_PROVIDERS, steering_controller=STEERING,
               position_estimator=ESTIMATORS, motion_driver=MOTION)


def build(setting, cfg, **kwargs):
    return OPTIONS[setting][getattr(cfg, setting)](cfg, **kwargs)
