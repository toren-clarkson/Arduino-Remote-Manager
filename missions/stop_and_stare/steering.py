"""Image-error controllers return bounded pulse lengths, not continuous driving."""


class TimedSteering:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        pass

    def duration(self, error, now):
        return self.cfg.turn_min_seconds + min(1, abs(error)) * (
            self.cfg.turn_max_seconds-self.cfg.turn_min_seconds)


class PIDSteering(TimedSteering):
    """Variable-dt PID magnitude, anti-windup and reset across gaps/deadband.

    Direction always follows the fresh measured error. PID cannot reverse a turn
    because of derivative kick or accumulated error from a previous target.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reset()

    def reset(self):
        self.previous = None
        self.integral = 0.0
        self.derivative = 0.0

    def duration(self, error, now):
        cfg = self.cfg
        dt = 0 if self.previous is None else now-self.previous[1]
        if dt <= 0 or dt > cfg.position_memory_seconds or (
                self.previous is not None and self.previous[0]*error < 0):
            self.reset()
            dt = 0
        derivative = (error-self.previous[0])/dt if dt else 0.0
        self.derivative = .5*self.derivative + .5*derivative
        candidate = max(-cfg.pid_integral_limit, min(cfg.pid_integral_limit, self.integral+error*dt))
        raw = cfg.pid_kp*error + cfg.pid_ki*candidate + cfg.pid_kd*self.derivative
        # Do not accumulate error that pushes an already saturated output further.
        if abs(raw) <= cfg.turn_max_seconds or raw*error <= 0:
            self.integral = candidate
        self.previous = (error, now)
        effort = max(0.0, raw if error > 0 else -raw)
        return max(cfg.turn_min_seconds, min(cfg.turn_max_seconds, effort))
