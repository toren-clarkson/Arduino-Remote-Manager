"""Latest-frame sources. Timestamps mean laptop receipt/read, not phone exposure time."""

import math
from dataclasses import dataclass

import cv2
import phonesense


@dataclass(frozen=True)
class Packet:
    sequence: int
    received_at: float
    jpeg: bytes


class Frames:
    def __init__(self, cfg, video=None):
        self.cfg, self.video = cfg, video
        self.cam = self.capture = self.latest = None
        self.sequence = 0
        self.eof = False
        self.last_jpeg = None
        self.next_read = 0.0

    def open(self):
        if self.video:
            self.capture = cv2.VideoCapture(str(self.video))
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = fps if math.isfinite(fps) and fps > 0 else 20
        else:
            self.cam = phonesense.start()
            if self.cfg.phone_ip:
                self.cam.select_ip(self.cfg.phone_ip)
            print("Phone URLs:", *self.cam.phone_urls, sep="\n  ", flush=True)
            print("On your phone: Connect, Start camera, and allow camera access.", flush=True)

    def poll(self, now):
        jpeg = None
        if self.capture is not None and not self.eof and now >= self.next_read:
            self.next_read = now+1/self.fps
            ok, image = self.capture.read()
            if not ok:
                self.eof = True
            else:
                ok, encoded = cv2.imencode(".jpg", image)
                if ok:
                    jpeg = encoded.tobytes()
        elif self.cam is not None:
            image = self.cam.jpeg
            if image is not None and image is not self.last_jpeg:
                jpeg = image
        if jpeg is not None:
            self.last_jpeg = jpeg
            self.sequence += 1
            self.latest = Packet(self.sequence, now, jpeg)
        return self.latest

    def healthy(self, now):
        return (self.latest is not None and now-self.latest.received_at <= self.cfg.camera_timeout
                and not self.eof and (self.cam is None or self.cam.is_streaming))

    def close(self):
        if self.cam:
            self.cam.stop()
        if self.capture:
            self.capture.release()
