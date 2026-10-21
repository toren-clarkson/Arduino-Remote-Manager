"""Local YOLO + ByteTrack for the object leg; AprilTag 3 for the return leg."""

import math

import cv2
import numpy as np

from .controller import Sighting


class YOLOObjects:
    def __init__(self, cfg):
        from ultralytics import YOLO
        if not cfg.weights.is_file():
            raise FileNotFoundError(f"Weights not found: {cfg.weights}. See training/README.md.")
        self.cfg = cfg
        self.model = YOLO(str(cfg.weights))
        if self.model is not None and self.model.task not in {"detect", "segment", "pose"}:
            raise ValueError("Navigation requires detection, segmentation or pose weights with axis-aligned boxes")
        if self.model is not None:
            print("YOLO classes:", self.model.names)
        matching = [i for i, name in (self.model.names.items() if self.model else [])
                    if name.casefold() == cfg.target_class.casefold()]
        if self.model is not None and not matching:
            raise ValueError(f"Class {cfg.target_class!r} not found. Choose a printed class or train custom weights.")
        self.class_id = matching[0] if matching else None
        self.track_id = None
        self.preferred_x = None

    @property
    def ready(self):
        return True

    def warmup(self):
        if self.model is None:
            return
        # Tracker imports/native initialization can be slow on the first call. Do this
        # before opening the camera/board, outside a stationary observation deadline.
        blank = np.zeros((self.cfg.image_size, self.cfg.image_size, 3), np.uint8)
        self._infer(blank)
        self.begin_burst()

    def begin_burst(self, preferred_x=None):
        # IDs persist across stationary frames only. Camera jumps invalidate track identities.
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
        self.track_id = None
        self.preferred_x = preferred_x

    def _infer(self, frame):
        options = dict(classes=[self.class_id], conf=self.cfg.confidence,
                       imgsz=self.cfg.image_size, device=self.cfg.device, verbose=False)
        if self.cfg.tracker == "none":
            return self.model.predict(frame, **options)
        return self.model.track(frame, persist=True, tracker=self.cfg.tracker+".yaml", **options)

    def detect(self, frame):
        height, width = frame.shape[:2]
        result = self._infer(frame)[0]
        candidates = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None]*len(boxes)
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(float, box)
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                    continue
                area = (x2-x1)*(y2-y1)/(width*height)
                candidates.append((Sighting("object", self.cfg.target_class,
                                            (x1+x2)/width-1, area,
                                            int(track_id) if track_id is not None else None), box))
        if self.track_id is not None:
            candidates = [item for item in candidates if item[0].track_id == self.track_id]
        if not candidates:
            return None, frame
        if self.preferred_x is None:
            sighting, box = max(candidates, key=lambda item: item[0].size)
        else:
            sighting, box = min(candidates, key=lambda item: abs(item[0].x-self.preferred_x))
        self.track_id = sighting.track_id
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{sighting.identity} ID {sighting.track_id} area {sighting.size:.1%}",
                    (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        return sighting, frame


class Perception:
    """Route stationary frames to interchangeable object providers and AprilTag 3."""
    def __init__(self, cfg):
        from .components import build
        self.cfg = cfg
        self.objects = build('object_provider', cfg) if cfg.mission_mode != 'tag_only' else None
        self.detector = None
        if cfg.mission_mode != 'object_only':
            from pupil_apriltags import Detector
            self.detector = Detector(families='tag36h11', nthreads=cfg.tag_threads,
                quad_decimate=cfg.tag_quad_decimate, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    def ready(self, leg):
        return leg != 'object' or self.objects is None or self.objects.ready

    def set_color(self, hsv):
        if self.cfg.object_provider != 'hsv' or self.objects is None:
            raise ValueError('This mission does not use HSV color detection')
        self.objects.set_color(hsv)

    def warmup(self):
        if self.objects:
            self.objects.warmup()

    def begin_burst(self, preferred_x=None):
        if self.objects:
            self.objects.begin_burst(preferred_x)

    def detect(self, jpeg, leg):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Invalid JPEG')
        if leg == 'object':
            return self.objects.detect(frame)
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches = [tag for tag in self.detector.detect(np.ascontiguousarray(gray))
                   if tag.tag_id == self.cfg.tag_id and tag.hamming == 0 and tag.decision_margin >= self.cfg.tag_min_margin]
        if not matches:
            return None, frame
        tag = max(matches, key=lambda t: abs(cv2.contourArea(t.corners.astype(np.float32))))
        side = float(np.linalg.norm(tag.corners-np.roll(tag.corners, 1, axis=0), axis=1).mean())/width
        sighting = Sighting("tag", f"tag:{self.cfg.tag_id}", float(2*tag.center[0]/width-1), side)
        cv2.polylines(frame, [tag.corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"tag {tag.tag_id} width {side:.1%}", (10, height-15),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        return sighting, frame
