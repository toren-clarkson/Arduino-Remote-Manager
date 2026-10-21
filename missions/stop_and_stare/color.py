"""OpenCV HSV object provider. No model weights, training or board imports."""

import cv2
import numpy as np

from .controller import Sighting


def sample_hsv(jpeg, x, y):
    """Sample the exact frozen camera JPEG selected by normalized coordinates."""
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Invalid camera JPEG')
    height, width = frame.shape[:2]
    col, row = min(width-1, int(x*width)), min(height-1, int(y*height))
    # Median S/V resists compression noise; circular hue average handles red wrap.
    patch = cv2.cvtColor(frame[max(0,row-2):row+3, max(0,col-2):col+3], cv2.COLOR_BGR2HSV)
    angles = patch[:, :, 0].astype(float)*2*np.pi/180
    hue = int(round(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())*180/(2*np.pi))) % 180
    return [hue, int(np.median(patch[:, :, 1])), int(np.median(patch[:, :, 2]))]


class HSVObjects:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hsv = None if cfg.color_h < 0 else (cfg.color_h, cfg.color_s, cfg.color_v)
        self.preferred_x = None

    @property
    def ready(self):
        return self.hsv is not None

    def set_color(self, hsv):
        if (not isinstance(hsv, (tuple, list)) or len(hsv) != 3
                or any(type(v) is not int for v in hsv)
                or not 0 <= hsv[0] <= 179 or any(not 0 <= v <= 255 for v in hsv[1:])):
            raise ValueError('Invalid HSV selection')
        self.hsv = tuple(hsv)

    def warmup(self):
        pass

    def begin_burst(self, preferred_x=None):
        self.preferred_x = preferred_x

    def detect(self, frame):
        if not self.ready:
            return None, frame
        h, s, v = self.hsv
        cfg = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low = (0, max(0, s-cfg.color_s_tolerance), max(0, v-cfg.color_v_tolerance))
        high = (179, min(255, s+cfg.color_s_tolerance), min(255, v+cfg.color_v_tolerance))
        sv = cv2.inRange(hsv, low, high)
        # Hue is circular in OpenCV's 8-bit range 0..179.
        delta = np.abs(hsv[:, :, 0].astype(np.int16)-h)
        hue = (np.minimum(delta, 180-delta) <= cfg.color_h_tolerance).astype(np.uint8)*255
        mask = cv2.bitwise_and(sv, hue)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        height, width = frame.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if cv2.contourArea(contour)/(height*width) < cfg.color_min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            sighting = Sighting('object', cfg.target_class, (2*x+w)/width-1, w*h/(width*height))
            candidates.append((sighting, (x,y,w,h)))
        if not candidates:
            return None, frame
        sighting, (x,y,w,h) = (max(candidates, key=lambda c:c[0].size) if self.preferred_x is None else
                               min(candidates, key=lambda c:abs(c[0].x-self.preferred_x)))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'HSV {self.hsv} area {sighting.size:.1%}', (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
        return sighting, frame
