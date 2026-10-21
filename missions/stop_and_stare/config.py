"""Edit hardware and movement calibration here; CLI flags override the model/class/IP."""

from dataclasses import dataclass
from pathlib import Path

from robotics_lab.paths import MODELS


@dataclass(frozen=True)
class Config:
    mission_mode: str = "object_then_tag"  # object_then_tag, object_only, tag_only
    arduino_ip: str = "10.13.96.68"
    phone_ip: str | None = None  # PC interface shared with the phone
    left_pins: tuple[int, int, int] = (7, 4, 5)  # IN1, IN2, PWM
    right_pins: tuple[int, int, int] = (9, 8, 6)
    weights: Path = MODELS / "yolo26n.pt"
    target_class: str = "bottle"  # exact model class name; change for custom weights
    tag_id: int = 0
    tag_min_margin: float = 20.0
    tag_quad_decimate: float = 1.0
    tag_threads: int = 2
    confidence: float = 0.5
    image_size: int = 416
    device: str = "cpu"
    tracker: str = "bytetrack"  # bytetrack, botsort, none
    object_provider: str = "yolo"  # yolo or hsv (OpenCV color blobs)
    color_h: int = -1  # -1 waits for a color selection in the mission monitor
    color_s: int = 200
    color_v: int = 200
    color_h_tolerance: int = 10
    color_s_tolerance: int = 80
    color_v_tolerance: int = 80
    color_min_area: float = 0.001  # contour area / full image area
    steering_controller: str = "timed"  # timed or pid; both produce discrete pulses
    pid_kp: float = 0.25  # seconds of turn hold / normalized image error
    pid_ki: float = 0.0
    pid_kd: float = 0.02
    pid_integral_limit: float = 0.3
    position_estimator: str = "none"  # none or kalman
    filter_process_noise: float = 0.02
    filter_measurement_noise: float = 0.01
    filter_motion_noise: float = 0.25
    filter_max_uncertainty: float = 0.5
    motion_driver: str = "ramped"
    object_stop_area: float = 0.12  # bounding-box area / full image area
    tag_stop_width: float = 0.24  # mean tag edge / full image width
    center_tolerance: float = 0.12  # normalized -1..1 horizontal error
    settle_seconds: float = 0.6
    discard_frames: int = 2  # fresh arrivals AFTER settling; never reuse movement frames
    observe_frames: int = 3
    confirmations: int = 2
    observation_timeout: float = 6.0
    inference_timeout: float = 8.0
    camera_timeout: float = 1.5
    camera_reconnect_timeout: float = 60.0  # wait at zero PWM; keep PhoneSense alive
    max_result_age: float = 1.0  # local receipt to inference completion
    empty_batches_before_search: int = 2
    position_memory_seconds: float = 8.0
    forward_speed: float = 0.18  # normalized motor demand, not measured velocity
    turn_speed: float = 0.18
    search_speed: float = 0.18
    forward_min_seconds: float = 0.08  # holds AFTER ramp-up, BEFORE ramp-down
    forward_max_seconds: float = 0.25
    turn_min_seconds: float = 0.08
    turn_max_seconds: float = 0.22
    search_seconds: float = 0.20
    min_pwm: int = 120
    acceleration: float = 300  # PWM counts/second; ramp time adds to every step
    deceleration: float = 400
    motor_interval: float = 0.02
    ramp_timeout: float = 4.0

    def validate(self):
        import math
        import ipaddress
        from dataclasses import fields
        defaults = Config()
        for field in fields(self):
            value, default = getattr(self, field.name), getattr(defaults, field.name)
            if isinstance(default, (int, float)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite number")
                if isinstance(default, int) and not isinstance(value, int):
                    raise ValueError(f"{field.name} must be an integer")
        from .components import OPTIONS
        for key, choices in OPTIONS.items():
            if not isinstance(getattr(self, key), str) or getattr(self, key) not in choices:
                raise ValueError(f"{key}: choose from {', '.join(choices)}")
        if not -1 <= self.color_h <= 179 or any(not 0 <= v <= 255 for v in (self.color_s, self.color_v)):
            raise ValueError("HSV needs H=-1 (pick color) or 0..179, S/V=0..255")
        if not 0 <= self.color_h_tolerance <= 90 or any(not 0 <= v <= 255 for v in (self.color_s_tolerance, self.color_v_tolerance)):
            raise ValueError("Color tolerances: H=0..90, S/V=0..255")
        if not 0 < self.color_min_area <= 1:
            raise ValueError("color_min_area must be in (0, 1]")
        for key in ('pid_kp', 'pid_ki', 'pid_kd', 'pid_integral_limit'):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ('filter_process_noise', 'filter_measurement_noise', 'filter_motion_noise', 'filter_max_uncertainty'):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.mission_mode not in {"object_then_tag", "object_only", "tag_only"}:
            raise ValueError("Unknown mission mode")
        if self.tracker not in {"bytetrack", "botsort", "none"}:
            raise ValueError("Choose bytetrack, botsort or none")
        ipaddress.IPv4Address(self.arduino_ip)
        if self.phone_ip:
            ipaddress.IPv4Address(self.phone_ip)
        if not isinstance(self.left_pins, (tuple, list)) or not isinstance(self.right_pins, (tuple, list)):
            raise ValueError("Motor pins must be lists/tuples")
        pins = (*self.left_pins, *self.right_pins)
        if (len(self.left_pins) != 3 or len(self.right_pins) != 3 or len(set(pins)) != 6
                or any(type(p) is not int or not 0 <= p <= 19 for p in pins)):
            raise ValueError("Provide six distinct UNO digital pins in [0, 19], as IN1, IN2, PWM")
        if not 1 <= self.tag_threads <= 16 or self.tag_min_margin < 0 or self.tag_quad_decimate <= 0:
            raise ValueError("Tag threads must be 1–16; margin >= 0; decimation > 0")
        for name in ("settle_seconds", "observation_timeout", "inference_timeout", "camera_timeout",
                     "max_result_age", "acceleration", "deceleration", "motor_interval", "ramp_timeout",
                     "forward_min_seconds", "forward_max_seconds", "turn_min_seconds", "turn_max_seconds",
                     "search_seconds", "position_memory_seconds", "camera_reconnect_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("object_stop_area", "tag_stop_width", "center_tolerance", "confidence",
                     "forward_speed", "turn_speed", "search_speed"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 1 <= self.confirmations <= self.observe_frames or self.discard_frames < 1:
            raise ValueError("Need 1 <= confirmations <= observe_frames, and discard_frames >= 1")
        if self.empty_batches_before_search < 1 or not self.target_class.strip():
            raise ValueError("Provide a target class and at least one empty batch before searching")
        if self.forward_min_seconds > self.forward_max_seconds or self.turn_min_seconds > self.turn_max_seconds:
            raise ValueError("Minimum movement duration cannot exceed maximum")
        if not 0 <= self.min_pwm <= 255:
            raise ValueError("min_pwm must be in [0, 255]")
        if not 0 <= self.tag_id < 587:
            raise ValueError("tag_id must be in [0, 586] for tag36h11")
