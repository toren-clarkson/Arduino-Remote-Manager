"""Pure stop-and-stare planning; it never reads a camera or writes motor pins."""

import math
from dataclasses import dataclass
from statistics import median

from .config import Config
from .components import build


@dataclass(frozen=True)
class Sighting:
    kind: str
    identity: str
    x: float  # -1 left, +1 right
    size: float  # object area fraction OR tag edge-width fraction
    track_id: int | None = None
    observed_at: float | None = None  # local packet receipt; never a phone clock


@dataclass(frozen=True)
class Step:
    name: str
    left: float = 0
    right: float = 0
    hold_seconds: float = 0
    wheel_holds: tuple[float, float] | None = None  # body controller may brake a faster wheel earlier


def confirmed_sighting(samples, cfg: Config):
    """Confirm actual sightings of one identity; missing samples are not predictions."""
    usable = [sample for sample in samples if sample is not None
              and all(math.isfinite(v) for v in (sample.x, sample.size))
              and -1 <= sample.x <= 1 and sample.size > 0]
    if not usable:
        return None
    groups = {}
    for sample in usable:
        groups.setdefault((sample.kind, sample.identity, sample.track_id), []).append(sample)
    group = max(groups.values(), key=len)
    if len(group) < cfg.confirmations:
        return None
    if max(s.x for s in group)-min(s.x for s in group) > 0.25:
        return None
    sizes = [s.size for s in group]
    if max(sizes)/min(sizes) > 2:
        return None
    # Use the smallest confirmed size: one oversized box cannot declare arrival.
    return Sighting(group[0].kind, group[0].identity, median(s.x for s in group),
                    min(sizes), group[0].track_id,
                    max((s.observed_at for s in group if s.observed_at is not None), default=None))


class Planner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.leg = "tag" if cfg.mission_mode == "tag_only" else "object"
        self.empty_batches = 0
        self.last_seen = -math.inf
        self.side = 1
        self.last_x = 0.0
        self.steering = build('steering_controller', cfg)
        self.estimator = build('position_estimator', cfg)

    def reset_memory(self):
        self.empty_batches, self.last_seen, self.side, self.last_x = 0, -math.inf, 1, 0.0
        self.steering.reset()
        self.estimator.reset()

    def preferred_position(self, now):
        return self.estimator.predict(now)

    def motion_completed(self, step, elapsed, now):
        self.estimator.moved(step, elapsed, now)

    def plan(self, sighting: Sighting | None, now: float) -> Step:
        if self.leg == "done":
            return Step("done")
        expected = self.cfg.target_class.casefold() if self.leg == "object" else f"tag:{self.cfg.tag_id}"
        if sighting is not None and (sighting.kind != self.leg or sighting.identity.casefold() != expected):
            sighting = None
        if sighting is None:
            self.steering.reset()
            self.empty_batches += 1
            if self.empty_batches < self.cfg.empty_batches_before_search:
                return Step("look_again")
            predicted = self.preferred_position(now)
            side = ((1 if predicted > 0 else -1) if predicted is not None and abs(predicted) > self.cfg.center_tolerance
                    else self.side if now-self.last_seen <= self.cfg.position_memory_seconds else 1)
            return Step("search_left" if side < 0 else "search_right",
                        self.cfg.search_speed*side, -self.cfg.search_speed*side, self.cfg.search_seconds)
        self.empty_batches = 0
        self.last_seen = now
        self.last_x = sighting.x
        measured_at = sighting.observed_at if sighting.observed_at is not None else now
        filtered_x = self.estimator.update(sighting.x, measured_at)
        # A delayed filter cannot authorize arrival or turn against a fresh detection.
        error = sighting.x if filtered_x*sighting.x < 0 else filtered_x
        if abs(sighting.x) > self.cfg.center_tolerance and abs(error) <= self.cfg.center_tolerance:
            error = sighting.x
        if abs(error) > self.cfg.center_tolerance:
            self.side = 1 if error > 0 else -1
            duration = self.steering.duration(error, measured_at)
            return Step("turn_left" if self.side < 0 else "turn_right",
                        self.cfg.turn_speed*self.side, -self.cfg.turn_speed*self.side, duration)
        self.steering.reset()
        threshold = self.cfg.object_stop_area if self.leg == "object" else self.cfg.tag_stop_width
        if sighting.size >= threshold:
            self.leg = "tag" if self.leg == "object" and self.cfg.mission_mode != "object_only" else "done"
            self.reset_memory()
            return Step("object_reached" if self.leg == "tag" else "done")
        distance_scale = max(0, min(1, 1-sighting.size/threshold))
        duration = self.cfg.forward_min_seconds + distance_scale * (
            self.cfg.forward_max_seconds-self.cfg.forward_min_seconds)
        return Step("forward", self.cfg.forward_speed, self.cfg.forward_speed, duration)
