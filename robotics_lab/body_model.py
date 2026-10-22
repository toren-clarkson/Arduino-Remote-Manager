"""Small local visual-response model, learned from stationary before/after pairs.

Maps signed PWM-seconds of each motor to target image displacement and log scale.
It learns actuator polarity/relative response; it is not metric odometry or an
encoder speed controller. No neural network or training service is involved.
"""

import json
import math
from pathlib import Path

import numpy as np

from robotics_lab.paths import PROJECT_ROOT


def body_path(cfg):
    path = Path(cfg.body_model)
    return path if path.is_absolute() else PROJECT_ROOT/path


def fingerprint(cfg):
    return dict(left_pins=list(cfg.left_pins), right_pins=list(cfg.right_pins),
                min_pwm=cfg.min_pwm, acceleration=cfg.acceleration, deceleration=cfg.deceleration)


def feature(sighting):
    # Object area has squared distance scaling; convert to a linear-size proxy.
    return np.array([sighting.x, math.log(sighting.size)*(0.5 if sighting.kind == 'object' else 1.)])


def expected_impulse(left, right, hold, cfg):
    """Ideal signed PWM/255 integral including both configured ramps and hold."""
    pwm = np.array([0 if s == 0 else math.copysign(cfg.min_pwm+abs(s)*(255-cfg.min_pwm), s)
                    for s in (left, right)])
    magnitude = np.abs(pwm)
    up_time = float(max(magnitude))/cfg.acceleration
    area = magnitude*up_time-magnitude**2/(2*cfg.acceleration)
    area += magnitude*hold+magnitude**2/(2*cfg.deceleration)
    return np.sign(pwm)*area/255


class VisualBodyModel:
    def __init__(self, matrix, *, metadata, max_demand, samples=()):
        self.matrix = np.asarray(matrix, dtype=float)
        self.metadata = metadata
        self.max_demand = float(max_demand)
        self.samples = list(samples)
        self.updates = 0
        self.covariance = np.eye(2)*.2
        if (self.matrix.shape != (2,2) or not np.isfinite(self.matrix).all()
                or not 0 < self.max_demand <= 1 or np.linalg.cond(self.matrix) > 80):
            raise ValueError('Body model is invalid or cannot distinguish steering from approach')

    @classmethod
    def fit(cls, samples, cfg):
        if len(samples) < 8:
            raise ValueError('Need repeated positive/negative tests of both motors')
        u = np.asarray([s['impulse'] for s in samples], dtype=float)
        y = np.asarray([s['delta'] for s in samples], dtype=float)
        if (u.shape != (len(samples),2) or y.shape != u.shape or not np.isfinite(u).all()
                or not np.isfinite(y).all() or np.linalg.cond(u) > 20):
            raise ValueError('Calibration commands do not independently excite both motors')
        # Both directions must produce detectable, repeatable motion for each motor.
        for axis in range(2):
            for sign in (-1,1):
                group = y[u[:,axis]*sign > .001]
                if len(group)<2 or np.linalg.norm(group.mean(axis=0)) < .008:
                    raise ValueError('A motor response is too small. Check friction, wiring, or camera visibility; no model saved')
        matrix = np.linalg.lstsq(u,y,rcond=None)[0].T
        error = np.linalg.norm(u@matrix.T-y)/max(np.linalg.norm(y),1e-9)
        if error > .30:
            raise ValueError('Motor response was inconsistent. Keep the camera attached and the marker stationary; no model saved')
        model = cls(matrix,metadata=fingerprint(cfg),max_demand=cfg.body_probe_speed,samples=samples)
        model.fit_error = float(error)
        return model

    @classmethod
    def load(cls, path, cfg):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema') != 1 or data.get('hardware') != fingerprint(cfg):
            raise ValueError('Body model does not match motor pins/PWM/ramps. Recalibrate or restore its hardware settings')
        return cls(data['matrix'],metadata=data['hardware'],max_demand=data['max_demand'],samples=data.get('samples',[]))

    def save_new(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        data = dict(schema=1,kind='local_visual_response',matrix=self.matrix.tolist(),hardware=self.metadata,
                    max_demand=self.max_demand,samples=self.samples,fit_error=getattr(self,'fit_error',None))
        # A failed calibration must never overwrite an existing robot profile.
        with path.open('x',encoding='utf-8') as stream:
            json.dump(data,stream,indent=2,allow_nan=False)

    def adapt(self, impulse, delta):
        """Bounded RLS correction; caller must verify fresh same-target evidence."""
        u, y = np.asarray(impulse,dtype=float), np.asarray(delta,dtype=float)
        if u.shape != (2,) or y.shape != (2,) or not np.isfinite(u).all() or not np.isfinite(y).all():
            return False
        prediction = self.matrix@u
        if np.linalg.norm(u)<.01 or np.linalg.norm(y-prediction) > max(.05,.5*np.linalg.norm(prediction)):
            return False
        gain = self.covariance@u/(.995+u@self.covariance@u)
        candidate = self.matrix+np.outer(y-prediction,gain)
        if np.linalg.cond(candidate)>80 or np.linalg.norm(candidate-self.matrix) > .03*np.linalg.norm(self.matrix):
            return False
        self.matrix = candidate
        self.covariance = (self.covariance-np.outer(gain,u)@self.covariance)/.995
        self.updates += 1
        return True

    def choose(self, sighting, step, cfg):
        """Choose a bounded discrete pulse by predicting candidates, not blind PID."""
        if not (step.name.startswith('turn_') or step.name == 'forward'):
            # Search without a target is also mapped through learned steering polarity.
            if not step.name.startswith('search_'):
                return step
            x = -.5 if step.name.endswith('left') else .5
            desired = np.array([-.2*x,0.])
            speed, high = cfg.search_speed,cfg.search_seconds
        elif step.name.startswith('turn_'):
            desired = np.array([-.35*sighting.x,0.])
            speed, high = cfg.turn_speed,step.hold_seconds
        else:
            threshold = cfg.object_stop_area if sighting.kind=='object' else cfg.tag_stop_width
            remaining = math.log(threshold/sighting.size)*(.5 if sighting.kind=='object' else 1.)
            desired = np.array([-.2*sighting.x,min(.06,max(.005,.3*remaining))])
            speed, high = cfg.forward_speed,step.hold_seconds
        speed = min(speed,self.max_demand)
        choices = [-speed,-speed/2,0.,speed/2,speed]
        best = None
        for left in choices:
            for right in choices:
                if left==right==0:
                    continue
                for left_hold in np.linspace(0,high,4):
                    for right_hold in np.linspace(0,high,4):
                        holds = (float(left_hold),float(right_hold))
                        response = self.matrix@expected_impulse(left,right,np.array(holds),cfg)
                        # Require predicted progress; do not run a contrary/ambiguous pulse.
                        if response@desired <= 0:
                            continue
                        if step.name=='forward' and response[1] <= 0:
                            continue
                        if step.name=='forward' and abs(response[0]-desired[0]) > .06:
                            continue  # do not replace a straight approach with a large yaw
                        cost = float(np.sum((response-desired)**2*np.array([2.,4.])))
                        if best is None or cost<best[0]:
                            best = (cost,left,right,holds)
        if best is None:
            raise ValueError('Body model cannot find a bounded pulse toward the target; recalibrate')
        from missions.stop_and_stare.controller import Step
        return Step('body_'+step.name,best[1],best[2],max(best[3]),best[3])
