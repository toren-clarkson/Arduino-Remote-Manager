"""Perception-independent, testable image-based navigation for robust-april-tag.py."""

import math
from dataclasses import dataclass

from robotics_lab.motor_control import TimedSearch


@dataclass(frozen=True)
class Settings:
    acquire_frames: int = 3
    acquire_seconds: float = 0.12
    acquire_gap: float = 0.25
    measurement_fresh: float = 0.16
    grace_seconds: float = 0.30
    search_after: float = 0.65
    memory_seconds: float = 4.0
    max_result_age: float = 0.35
    filter_seconds: float = 0.12
    max_velocity: float = 1.0
    max_prediction_shift: float = 0.15
    innovation_gate: float = 0.55
    center_deadband: float = 0.05
    arrival_alignment: float = 0.12
    arrival_frames: int = 3
    arrival_seconds: float = 0.15
    base_speed: float = 0.22
    steering_gain: float = 0.35
    max_turn: float = 0.22
    search_speed: float = 0.18
    search_turn: float = 1.1
    search_pause: float = 0.8
    close_fraction: float = 0.75


@dataclass(frozen=True)
class Observation:
    key: str
    x: float  # normalized image position: -1 left, +1 right
    y: float
    size: float  # tag side length / image width, or color radius / image width


@dataclass(frozen=True)
class Motion:
    phase: str
    left: float = 0.0
    right: float = 0.0
    hard_stop: bool = False


class Navigator:
    """Only real, new, timely detections can acquire a target or confirm arrival.

    Position/velocity are image estimates, not world position or robot odometry.
    Missing detections never refresh last_seen, confirmation count, or arrival.
    """

    def __init__(self, stop_size, *, expected_key=None, settings=None):
        self.cfg = settings or Settings()
        self.stop_size = stop_size
        self.expected_key = expected_key
        self.search = TimedSearch(self.cfg.search_turn, self.cfg.search_pause)
        self.reset()

    def reset(self):
        self.locked_key = self.expected_key
        self.last_packet = self.last_seen = -math.inf
        self.x = self.y = self.size = self.vx = 0.0
        self.raw_x = 0.0
        self.side = self.search_side = 1
        self.streak = self.close_streak = 0
        self.streak_started = self.close_started = 0.0
        self.candidate_key = None
        self.acquired = self.reached = False
        self.phase = "WAIT"
        self.search.reset()

    def ingest(self, observation, frame_time, now):
        """Return False for an obsolete packet; rejected targets still count as processed frames."""
        if frame_time <= self.last_packet or not 0 <= now - frame_time <= self.cfg.max_result_age:
            return False
        self.last_packet = frame_time
        if observation is None or (self.locked_key is not None and observation.key != self.locked_key):
            self.streak = self.close_streak = 0
            return True
        if (not all(math.isfinite(v) for v in (observation.x, observation.y, observation.size))
                or abs(observation.x) > 1.1 or abs(observation.y) > 1.1 or observation.size <= 0):
            self.streak = self.close_streak = 0
            return True

        gap = frame_time - self.last_seen
        new_identity = self.candidate_key != observation.key
        if self.acquired and gap <= self.cfg.search_after:
            prediction = self.predicted_x(frame_time)
            if abs(observation.x - prediction) > self.cfg.innovation_gate:
                self.streak = self.close_streak = 0
                return True
            # A sudden huge size jump should not count as arrival.
            if self.size > 0 and observation.size > self.size * 2.0:
                self.streak = self.close_streak = 0
                return True

        if (not self.acquired and self.streak and gap <= self.cfg.acquire_gap
                and (abs(observation.x-self.raw_x) > self.cfg.innovation_gate
                     or (self.size > 0 and max(observation.size/self.size, self.size/observation.size) > 2))):
            self.streak = self.close_streak = 0
        if gap > self.cfg.grace_seconds or new_identity:
            self.acquired = False
            self.streak = 0
        if gap > self.cfg.acquire_gap:
            self.streak = self.close_streak = 0
        if not self.streak:
            self.streak_started = frame_time
        self.streak += 1
        self.candidate_key = observation.key

        if gap > self.cfg.search_after or not math.isfinite(gap) or new_identity:
            self.x, self.y, self.size, self.vx = observation.x, observation.y, observation.size, 0.0
        else:
            alpha = 1 - math.exp(-max(gap, 0.001) / self.cfg.filter_seconds)
            measured_velocity = (observation.x - self.raw_x) / max(gap, 0.001)
            measured_velocity = max(-self.cfg.max_velocity, min(self.cfg.max_velocity, measured_velocity))
            self.vx = 0.8 * self.vx + 0.2 * measured_velocity
            self.x += alpha * (observation.x - self.x)
            self.y += alpha * (observation.y - self.y)
            self.size += alpha * (observation.size - self.size)
        self.raw_x = observation.x
        self.last_seen = frame_time
        if abs(observation.x) > self.cfg.center_deadband:
            self.side = 1 if observation.x > 0 else -1
        if self.streak >= self.cfg.acquire_frames and frame_time - self.streak_started >= self.cfg.acquire_seconds:
            self.acquired = True
            self.locked_key = observation.key

        if observation.size >= self.stop_size and abs(observation.x) <= self.cfg.arrival_alignment:
            if not self.close_streak:
                self.close_started = frame_time
            self.close_streak += 1
            if (self.acquired and self.close_streak >= self.cfg.arrival_frames
                    and frame_time - self.close_started >= self.cfg.arrival_seconds):
                self.reached = True
        else:
            self.close_streak = 0
        return True

    def predicted_x(self, now):
        age = max(0.0, min(self.cfg.grace_seconds, now - self.last_seen))
        shift = max(-self.cfg.max_prediction_shift, min(self.cfg.max_prediction_shift, self.vx * age))
        return max(-1.0, min(1.0, self.x + shift))

    def _steer(self, now, *, coast=False):
        x = self.predicted_x(now)
        if abs(x) <= self.cfg.center_deadband:
            x = 0.0
        turn = max(-self.cfg.max_turn, min(self.cfg.max_turn, self.cfg.steering_gain * x))
        # Slow with increasing target size and steering error, before the arrival threshold.
        distance_scale = max(0.0, min(1.0, (self.stop_size - self.size) / (self.stop_size * 0.5)))
        forward = self.cfg.base_speed * distance_scale * max(0.0, 1.0 - abs(x) / 0.7)
        if coast:
            fade = max(0.0, 1.0 - (now - self.last_seen) / self.cfg.grace_seconds)
            forward *= 0.4 * fade
            turn *= 0.5 * fade
            if self.size >= self.stop_size * self.cfg.close_fraction:
                forward = 0.0
        if self.close_streak:
            forward = 0.0  # verify the stopping condition with fresh images while stationary
        return max(-1.0, min(1.0, forward + turn)), max(-1.0, min(1.0, forward - turn))

    def tick(self, now, *, camera_live, stopped):
        if not camera_live:
            self.reset()
            self.phase = "STALE"
            return Motion("STALE", hard_stop=True)
        age = now - self.last_seen
        if self.reached:
            self.phase = "ARRIVED"
            return Motion("ARRIVED")
        if self.acquired and age <= self.cfg.measurement_fresh:
            self.phase = "TRACK"
            return Motion(self.phase, *self._steer(now))
        if self.acquired and age <= self.cfg.grace_seconds:
            self.phase = "COAST"
            return Motion(self.phase, *self._steer(now, coast=True))
        if age > self.cfg.grace_seconds:
            self.acquired = False
        if age <= self.cfg.grace_seconds and self.streak:
            self.phase = "ACQUIRE"
            return Motion(self.phase)
        if age <= self.cfg.search_after:
            self.phase = "SETTLE"
            return Motion(self.phase)
        if self.phase != "SEARCH":
            self.search.reset()
            self.search_side = self.side if age <= self.cfg.memory_seconds else 1
        previous = self.search.phase
        turning = self.search.update(now, stopped=stopped)
        if previous == "pause" and turning:
            self.search_side = self.side if age <= self.cfg.memory_seconds else 1
        self.phase = "SEARCH"
        speed = self.cfg.search_speed * self.search_side if turning else 0.0
        return Motion(self.phase, speed, -speed)
