"""Timestamped horizontal memory; predictions never authorize approach/arrival."""

import math

import numpy as np


class RawPosition:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.time = None
        self.x = 0.0

    def update(self, x, now):
        self.x, self.time = x, now
        return x

    def predict(self, now):
        if self.time is None or not 0 <= now-self.time <= self.cfg.position_memory_seconds:
            return None
        return self.x

    def moved(self, step, elapsed, now):
        pass


class KalmanPosition(RawPosition):
    """Constant-velocity Kalman model in normalized image coordinates.

    Motor movement invalidates velocity learned while stationary. Unknown ego
    motion increases covariance instead of inventing a calibrated yaw response.
    """
    def reset(self):
        super().reset()
        self.state = np.zeros(2)
        self.covariance = np.eye(2)
        self.predicted_at = None

    def _advance(self, now):
        if self.predicted_at is None:
            self.predicted_at = now
            return
        dt = max(0.0, now-self.predicted_at)
        transition = np.array([[1., dt], [0., 1.]])
        q = self.cfg.filter_process_noise
        noise = q*np.array([[dt+dt**3/3, dt**2/2], [dt**2/2, dt]])
        self.state = transition@self.state
        self.covariance = transition@self.covariance@transition.T+noise
        self.predicted_at = max(now, self.predicted_at)

    def update(self, x, now):
        if self.time is None or now <= self.time or now-self.time > self.cfg.position_memory_seconds:
            self.reset()
            self.state[0] = x
            self.covariance = np.diag([self.cfg.filter_measurement_noise, 1.])
            self.predicted_at = now
        else:
            self._advance(now)
            gain = self.covariance[:, 0]/(self.covariance[0, 0]+self.cfg.filter_measurement_noise)
            self.state += gain*(x-self.state[0])
            # Joseph form preserves positive covariance under floating-point error.
            residual = np.eye(2)-np.outer(gain, [1., 0.])
            self.covariance = residual@self.covariance@residual.T + np.outer(gain, gain)*self.cfg.filter_measurement_noise
        self.time = now
        self.x = float(np.clip(self.state[0], -1, 1))
        return self.x

    def predict(self, now):
        if self.time is None or not 0 <= now-self.time <= self.cfg.position_memory_seconds:
            return None
        self._advance(now)
        if math.sqrt(max(0, self.covariance[0, 0])) > self.cfg.filter_max_uncertainty:
            return None
        return float(np.clip(self.state[0], -1, 1))

    def moved(self, step, elapsed, now):
        if self.time is None:
            return
        self._advance(now)
        self.state[1] = 0.0
        self.covariance += np.diag([self.cfg.filter_motion_noise*max(0, elapsed), 1.])
