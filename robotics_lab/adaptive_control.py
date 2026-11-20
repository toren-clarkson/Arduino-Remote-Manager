"""Small, inspectable online body models; no hardware or neural-network training.

Each model maps integrated signed PWM/255 to changes in [x, log(target extent)].
Separate models are needed for each visual target because apparent size is local.
"""

import json
import math
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotics_lab.paths import PROJECT_ROOT

PROFILE_DIR = PROJECT_ROOT / 'calibrations' / 'adaptive'


def vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError('Expected two finite numbers')
    return result


@dataclass
class BodyModel:
    jacobian: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 50)
    samples: int = 0

    @property
    def ready(self):
        singular = np.linalg.svd(self.jacobian, compute_uv=False)
        return (self.samples >= 4 and singular[-1] > .015
                and singular[0] / singular[-1] < 40)

    @classmethod
    def fit(cls, efforts, changes):
        x, y = np.asarray(efforts, float), np.asarray(changes, float)
        if (x.ndim != 2 or x.shape[1] != 2 or y.shape != x.shape or len(x) < 4
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.linalg.matrix_rank(x) < 2):
            raise ValueError('Calibration needs independent, finite wheel probes')
        matrix = np.linalg.lstsq(x, y, rcond=None)[0].T
        model = cls(matrix, np.eye(2) * 20, len(x))
        if not model.ready or np.max(np.linalg.norm(y - x @ matrix.T, axis=1)) > .10:
            raise ValueError('Calibration has too little motion or inconsistent response. '
                             'Keep the target stationary/in view; check grip and probe PWM/duration.')
        # Both directions of each wheel must produce a measurable opposing response.
        for axis in range(2):
            positives, negatives = y[x[:, axis] > .005], y[x[:, axis] < -.005]
            if len(positives) == 0 or len(negatives) == 0:
                raise ValueError('Probe both directions of both wheels')
            positive, negative = positives.mean(axis=0), negatives.mean(axis=0)
            if (not np.isfinite([positive, negative]).all()
                    or min(np.linalg.norm(positive), np.linalg.norm(negative)) < .008
                    or np.dot(positive, negative) >= 0):
                raise ValueError('A wheel did not move consistently in both directions; recalibrate')
        return model

    def update(self, effort, change):
        """RLS with forgetting; reject jumps and prevent sudden gain/sign changes."""
        u, delta = vector(effort), vector(change)
        if np.linalg.norm(u) < .015 or np.max(np.abs(delta)) > .35:
            return False
        residual = delta - self.jacobian @ u
        if np.linalg.norm(residual) > .12:
            return False
        gain = self.covariance @ u / (.995 + u @ self.covariance @ u)
        candidate = self.jacobian + np.outer(residual, gain)
        covariance = (self.covariance - np.outer(gain, u) @ self.covariance) / .995
        trial = BodyModel(candidate, covariance, self.samples + 1)
        if (not trial.ready or np.linalg.norm(candidate-self.jacobian) > .10
                or np.sign(np.linalg.det(candidate)) != np.sign(np.linalg.det(self.jacobian))):
            return False
        self.jacobian = candidate
        self.covariance = (covariance + covariance.T) / 2
        self.samples += 1
        return True

    def command(self, desired, max_pwm):
        if not self.ready:
            raise ValueError('Calibrate this target before driving')
        # Damped inverse: keep poorly observed directions from demanding huge PWM.
        j = self.jacobian
        output = j.T @ np.linalg.solve(j @ j.T + np.eye(2) * .0025, vector(desired))
        limit = max_pwm / 255
        output *= min(1., limit / max(np.max(np.abs(output)), 1e-9))
        return output


class EffortHistory:
    """Integrate applied host PWM, including ramps, against observation receipt times."""
    def __init__(self):
        self.entries = deque()

    def record(self, now, pwm):
        self.entries.append((now, vector(pwm).copy()))
        while len(self.entries) > 2 and self.entries[1][0] < now - 12:
            self.entries.popleft()

    def integral(self, start, end):
        result, previous = np.zeros(2), np.zeros(2)
        cursor = start
        for stamp, value in self.entries:
            if stamp <= start:
                previous = value
            elif stamp < end:
                result += previous * (stamp-cursor)
                cursor, previous = stamp, value
            else:
                break
        return result + previous * max(0., end-cursor)


@dataclass
class RobotProfile:
    name: str = 'robot-floor'
    surface: str = 'Describe the surface'
    left: tuple = (7, 4, 5)
    right: tuple = (9, 8, 6)
    max_pwm: int = 165
    minimum_pwm: tuple = (0, 0)
    slew: float = 400.
    probe_seconds: float = .35
    settle: float = .6
    camera_delay: float = .12
    models: dict = field(default_factory=dict)
    signatures: dict = field(default_factory=dict)
    color_hsv: tuple | None = None

    def validate(self):
        if self.color_hsv is not None and (len(self.color_hsv) != 3 or any(
                type(n) is not int or not 0 <= n <= limit for n,limit in zip(self.color_hsv,(179,255,255)))):
            raise ValueError('HSV must be three integers: H 0..179, S/V 0..255')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,59}', self.name):
            raise ValueError('Profile name: 1..60 letters, numbers, hyphens or underscores')
        pins = list(self.left) + list(self.right)
        if (len(self.left) != 3 or len(self.right) != 3 or len(set(pins)) != 6
                or any(type(p) is not int or not 0 <= p <= 13 for p in pins)
                or self.left[2] not in (3, 5, 6, 9, 10, 11)
                or self.right[2] not in (3, 5, 6, 9, 10, 11)):
            raise ValueError('Choose six distinct UNO pins; enable pins must support PWM')
        if not (40 <= self.max_pwm <= 220 and 50 <= self.slew <= 1500
                and .1 <= self.probe_seconds <= 1 and .2 <= self.settle <= 3
                and 0 <= self.camera_delay <= .5):
            raise ValueError('Profile movement/timing settings are out of range')
        if len(self.minimum_pwm) != 2 or any(type(n) is not int or not 0 <= n <= self.max_pwm for n in self.minimum_pwm):
            raise ValueError('Invalid minimum moving PWM values')
        for model in self.models.values():
            if (model.jacobian.shape != (2, 2) or model.covariance.shape != (2, 2)
                    or not np.isfinite(model.jacobian).all()
                    or not np.isfinite(model.covariance).all()
                    or not np.allclose(model.covariance, model.covariance.T)
                    or np.min(np.linalg.eigvalsh(model.covariance)) <= 0 or not model.ready):
                raise ValueError('Invalid or uncalibrated saved body model')

    @property
    def path(self):
        self.validate()
        return PROFILE_DIR / f'{self.name}.json'

    def save(self):
        self.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema=1, name=self.name, surface=self.surface, left=self.left,
                       right=self.right, max_pwm=self.max_pwm, minimum_pwm=self.minimum_pwm, slew=self.slew,
                       probe_seconds=self.probe_seconds, settle=self.settle,
                       camera_delay=self.camera_delay, signatures=self.signatures, color_hsv=self.color_hsv,
                       models={key: dict(jacobian=m.jacobian.tolist(), covariance=m.covariance.tolist(),
                                         samples=m.samples) for key, m in self.models.items()})
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, self.path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.pop('schema', None) != 1:
            raise ValueError('This is not an adaptive profile (historical body profiles are incompatible)')
        models = {key: BodyModel(np.asarray(m['jacobian'], float), np.asarray(m['covariance'], float),
                                 m['samples']) for key, m in data.pop('models').items()}
        profile = cls(**data, models=models)
        profile.left, profile.right = tuple(profile.left), tuple(profile.right)
        profile.validate()
        return profile


class VisualServo:
    """Prediction bridges short gaps; only real observations can complete a stage."""
    def __init__(self, model, threshold, max_pwm, delay=.12):
        self.model, self.threshold, self.max_pwm, self.delay = model, threshold, max_pwm, delay
        self.history = EffortHistory()
        self.feature = self.stamp = self.identity = self.previous = None
        self.side, self.arrivals, self.acquired = 1, 0, 0
        self.done = False
        self.missing = False
        self.rejected = 0
        self.confirmed = False

    def observe(self, target, stamp, now):
        if target is None or now-stamp > .55:
            self.previous = None
            self.missing = True
            self.arrivals = 0
            self.acquired = 0
            return False
        identity, x, extent = target
        if not np.isfinite([x, extent, stamp]).all() or not -1 <= x <= 1 or not 0 < extent <= 1.5:
            return self.observe(None, stamp, now)
        feature = np.array([x, math.log(extent)])
        if self.stamp is not None and stamp <= self.stamp:
            return False
        same = self.identity == identity
        if not same or (self.stamp is not None and stamp-self.stamp > .6):
            self.acquired = self.arrivals = 0
            self.confirmed = False
        if same and self.feature is not None and stamp-self.stamp < .6:
            predicted = self.feature + self.model.jacobian @ self.history.integral(
                self.stamp-self.delay, stamp-self.delay)
            if np.max(np.abs(feature-predicted)) > .3:
                self.rejected += 1
                self.previous = None
                self.missing = True
                self.arrivals = self.acquired = 0
                return False
        if self.previous is not None and same:
            old, old_stamp = self.previous
            dt = stamp-old_stamp
            if .08 <= dt <= .6:
                effort = self.history.integral(old_stamp-self.delay, stamp-self.delay)
                self.model.update(effort, feature-old)
        self.previous = (feature, stamp)
        self.feature, self.stamp, self.identity = feature, stamp, identity
        self.acquired = self.acquired + 1 if same else 1
        self.confirmed = self.acquired >= 2
        self.missing = False
        if abs(x) > .1:
            self.side = 1 if x > 0 else -1
        self.arrivals = self.arrivals + 1 if abs(x) < .12 and extent >= self.threshold else 0
        self.done = self.arrivals >= 3
        return True

    def command(self, now):
        if self.done:
            return np.zeros(2), 'reached'
        age = math.inf if self.stamp is None else now-self.stamp
        if self.feature is not None and age <= .40 and self.confirmed:
            prediction = self.model.jacobian @ self.history.integral(self.stamp-self.delay, now)
            feature = self.feature + np.clip(prediction, -.15, .15)
            # Aim just past the threshold so proportional approach does not stop
            # asymptotically short of the three real arrival confirmations.
            desired = np.array([-1.1*feature[0], .65*max(0., math.log(self.threshold)-feature[1]+.03)])
            desired[1] *= max(0., 1-abs(feature[0])/.45)
            desired = np.clip(desired, [-.35, 0.], [.35, .22])
            if self.missing or age > .25:
                desired *= min(1., max(0., (.40-age)/.15))
            return self.model.command(desired, self.max_pwm), 'predict' if self.missing else 'track'
        # A brief stationary reacquisition period after the prediction horizon.
        if age <= .9:
            return np.zeros(2), 'reacquire'
        # Search only turns: ask for an image shift opposite the last-seen side.
        # A 0.35s turn / 0.65s pause preserves chances of decoding a sharp tag.
        if now % 1.0 < .35:
            return self.model.command([-.18*self.side, 0.], min(self.max_pwm, 150)), 'search'
        return np.zeros(2), 'search pause'
