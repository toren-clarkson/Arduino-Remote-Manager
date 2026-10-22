"""Bounded, display-only telemetry. Disk work never runs in motor polling."""

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from robotics_lab.snapshots import publish_snapshot
from robotics_lab.live_camera import LiveCamera


class MissionTelemetry:
    def __init__(self, directory, *, drive=False, replay=False):
        self.directory = directory
        self.state = dict(phase='STARTING', leg='object', drive=drive, replay=replay,
                          phone_urls=[], camera_connected=False, commands=[])
        self.commands = deque(maxlen=50)
        self.condition = threading.Condition()
        self.pending = None  # one latest snapshot; never accumulate camera frames
        self.jpeg = None
        self.sequence = None
        self.observation = None
        self.observation_sequence = 0
        self.last_warning = -float('inf')
        self.closed = False
        self.live = LiveCamera()
        self.state['preview_ipc'] = self.live.credentials
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()
        self.update()

    def update(self, *, jpeg=None, sequence=None, observation=None, **fields):
        with self.condition:
            if self.closed:
                return
            self.state.update(fields)
            self.state['updated_at'] = datetime.now(timezone.utc).isoformat()
            if jpeg is not None:
                self.jpeg, self.sequence = jpeg, sequence
                self.live.publish(jpeg, sequence)
            if observation is not None:
                self.observation = observation
                self.observation_sequence += 1
            self.pending = (dict(self.state), self.jpeg, self.sequence, self.observation, self.observation_sequence)
            self.condition.notify()

    def publish_observation(self, frame):
        # Detector frames are not mutated after publication. Hold just the latest
        # reference; encoding/writing/retries belong exclusively to the writer.
        self.update(observation=frame)

    def _warning(self, exc):
        now = time.monotonic()
        if now-self.last_warning >= 5:
            print(f'Mission display update skipped (control continues): {exc}', flush=True)
            self.last_warning = now

    def _publish(self, path, data):
        try:
            publish_snapshot(self.directory, path.name, data)
            return True
        except OSError as exc:
            self._warning(exc)
        return False

    def event(self, row):
        if row['event'] == 'phase':
            self.update(phase=row['phase'], leg=row['leg'])
        elif row['event'] == 'decision':
            command = dict(row, requested_at=datetime.now(timezone.utc).isoformat())
            self.commands.append(command)
            self.update(commands=list(self.commands), last_command=command)
        elif row['event'] in {'stationary_frame','preview_frame'}:
            self.update(last_detection={k: row[k] for k in ('identity', 'track_id', 'x', 'size', 'frame_age_ms') if k in row})

    def _writer(self):
        written_sequence = None
        written_observation = 0
        encoded_observation = 0
        encoded_jpeg = None
        last_camera_disk = last_observation_disk = last_state_disk = -float('inf')
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.pending is None:
                    return
                state, jpeg, sequence, observation, observation_sequence = self.pending
                self.pending = None
            try:
                now = time.monotonic()
                if jpeg is not None and sequence != written_sequence and (self.closed or now-last_camera_disk >= 1):
                    if self._publish(self.directory/'camera.jpg', jpeg):
                        written_sequence = sequence
                        last_camera_disk = now
                if observation is not None and observation_sequence != encoded_observation:
                    import cv2
                    ok, encoded = cv2.imencode('.jpg', observation)
                    if ok:
                        encoded_jpeg = encoded.tobytes()
                        encoded_observation = observation_sequence
                        self.live.publish(encoded_jpeg, observation_sequence, 'observation')
                if encoded_jpeg is not None and encoded_observation != written_observation and (
                        self.closed or now-last_observation_disk >= 1):
                    if self._publish(self.directory/'preview.jpg', encoded_jpeg):
                        written_observation = observation_sequence
                        last_observation_disk = now
                # The browser cache key advances only after the image is published.
                state['observation_sequence'] = written_observation
                if self.closed or now-last_state_disk >= .25:
                    self._publish(self.directory/'runtime.json', json.dumps(state, allow_nan=False).encode('utf-8'))
                    last_state_disk = now
            except Exception as exc:
                # Display, encoding and file failures cannot terminate robot control.
                self._warning(exc)

    def close(self):
        with self.condition:
            self.pending = (dict(self.state), self.jpeg, self.sequence, self.observation, self.observation_sequence)
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=2)
        self.live.close()
