"""Observe while stopped -> one timed movement -> stop -> settle -> new observation.

From the project root: python -m missions.stop_and_stare.run --target-class bottle
Add --drive to enable the robot. See this folder's README.md for setup and training.
"""

import argparse
import asyncio
import csv
import json
import os
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robotics_lab.paths import RUNS
from missions.stop_and_stare.camera import Frames
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Planner, confirmed_sighting
from missions.stop_and_stare.perception import Perception
from missions.stop_and_stare.components import build


class StopRequested(Exception):
    pass


class VideoEnded(Exception):
    pass


class CameraUnavailable(RuntimeError):
    """A recoverable camera gap; never evidence that the target is absent."""


class Runner:
    def __init__(self, cfg, source, perception, driver, session, *, headless=False, save_frames=False,
                 stop_file=None, preview_path=None, telemetry=None, body=None):
        self.cfg, self.source, self.perception, self.driver = cfg, source, perception, driver
        self.planner = Planner(cfg)
        self.body = body
        self.pending_body = None
        self.session = session
        self.headless, self.save_frames = headless, save_frames
        self.stop_file, self.preview_path = stop_file, preview_path
        self.command_directory = stop_file.parent if stop_file else None
        self.last_color_request = ''
        self.telemetry = telemetry
        self.last_telemetry = -float('inf')
        self.last_window_update = -float('inf')
        self.camera_arrivals = deque(maxlen=120)
        self.last_camera_sequence = None
        self.detector_ms = None
        self.phase = "START"
        self.display = np.zeros((480, 640, 3), np.uint8)
        self.window = "Stop-and-stare object -> AprilTag"
        self.step_name = ""
        self.frame_number = 0
        self.file = (session/"events.csv").open("x", newline="", encoding="utf-8")
        self.log = csv.DictWriter(self.file, fieldnames=[
            "time", "event", "phase", "leg", "sequence", "identity", "track_id", "x", "size",
            "frame_age_ms", "inference_ms", "step", "left", "right", "hold_seconds",
            "left_hold_seconds", "right_hold_seconds",
        ])
        self.log.writeheader()

    def event(self, name, **fields):
        packet = self.source.latest
        row = dict(time=time.monotonic(), event=name, phase=self.phase, leg=self.planner.leg,
                   sequence=packet.sequence if packet else "")
        row.update(fields)
        self.log.writerow(row)
        self.file.flush()
        if self.telemetry:
            self.telemetry.event(row)

    def poll(self, phase):
        if self.stop_file and self.stop_file.exists():
            raise StopRequested
        if phase != self.phase:
            self.phase = phase
            self.event("phase")
            print(f"Phase: {phase} | leg: {self.planner.leg}", flush=True)
        self.source.poll(time.monotonic())  # drain arrivals during movement; never decode or infer here
        now = time.monotonic()
        packet = self.source.latest
        if packet and packet.sequence != self.last_camera_sequence:
            self.camera_arrivals.append(packet.received_at)
            self.last_camera_sequence = packet.sequence
        if self.telemetry and now-self.last_telemetry >= 1/self.cfg.preview_fps:
            packet = self.source.latest
            arrivals = [t for t in self.camera_arrivals if now-t <= 2]
            fps = (len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 and arrivals[-1]>arrivals[0] else 0
            self.telemetry.update(phase=self.phase, leg=self.planner.leg,
                camera_fps=round(fps, 1), detector_ms=self.detector_ms,
                camera_connected=self.source.healthy(now),
                frame_age_seconds=now-packet.received_at if packet else None,
                jpeg=packet.jpeg if packet else None, sequence=packet.sequence if packet else None)
            self.last_telemetry = now
        if not self.headless and now-self.last_window_update >= 1/30:
            self.last_window_update = now
            preview = self.display.copy()
            cv2.putText(preview, f"{self.planner.leg.upper()} | {phase} | {self.step_name}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
            if phase in ("RAMP_UP", "MOVE", "BRAKE", "SETTLE", "FLUSH"):
                cv2.putText(preview, "Video ignored; showing last stationary image", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
            cv2.imshow(self.window, preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                raise StopRequested

    async def motion_tick(self, phase):
        self.poll(phase)
        if self.source.eof:
            raise VideoEnded
        if not self.source.healthy(time.monotonic()):
            raise CameraUnavailable("Camera connection became stale during movement; robot stopped")

    async def wait_new(self, after, deadline, phase):
        while time.monotonic() < deadline:
            self.poll(phase)
            packet = self.source.latest
            if self.source.eof:
                raise VideoEnded
            if packet is not None and not self.source.healthy(time.monotonic()):
                raise CameraUnavailable('Camera frames became stale while observing')
            if packet is not None and packet.sequence > after:
                return packet
            await asyncio.sleep(.01)
        raise CameraUnavailable('No fresh stationary camera frame arrived before the frame-wait timeout')

    async def recover_camera(self, reason):
        await self.driver.stop()
        self.planner.reset_memory()
        self.pending_body = None
        self.event('camera_gap')
        print(f'Camera interrupted: {reason}. Motors stopped; PhoneSense stays running. '
              f'Waiting up to {self.cfg.camera_reconnect_timeout:g}s for fresh frames. '
              'On the phone, tap Connect and Start camera if needed.', flush=True)
        sequence = self.source.sequence
        deadline = time.monotonic()+self.cfg.camera_reconnect_timeout
        while time.monotonic() < deadline:
            self.poll('WAIT_RECONNECT')
            if self.source.eof:
                raise VideoEnded
            if self.source.sequence > sequence and self.source.healthy(time.monotonic()):
                self.event('camera_reconnected')
                print('Fresh camera frames resumed. Discarding the interrupted observation and starting a new stationary burst.', flush=True)
                return
            await asyncio.sleep(.02)
        raise TimeoutError(f'No fresh camera frames within {self.cfg.camera_reconnect_timeout:g}s; mission stopped')

    async def observe(self):
        if self.driver.control is not None and not self.driver.control.is_stopped:
            raise RuntimeError("Refusing to observe while motor PWM is nonzero")
        settle_until = time.monotonic()+self.cfg.settle_seconds
        while time.monotonic() < settle_until:
            self.poll("SETTLE")
            if self.source.eof:
                raise VideoEnded
            if not self.source.healthy(time.monotonic()):
                raise CameraUnavailable('Camera frames became stale while settling')
            await asyncio.sleep(.01)
        sequence = self.source.sequence  # gate everything received before settling completed
        for _ in range(self.cfg.discard_frames):
            packet = await self.wait_new(sequence, time.monotonic()+self.cfg.observation_timeout, "FLUSH")
            sequence = packet.sequence
        preferred = self.planner.preferred_position(time.monotonic())
        self.perception.begin_burst(preferred)
        samples = []
        stale_inference_seconds = 0.0
        while len(samples) < self.cfg.observe_frames:
            packet = await self.wait_new(sequence, time.monotonic()+self.cfg.observation_timeout, "OBSERVE")
            sequence = packet.sequence
            started = time.monotonic()
            camera_gap = False
            task = asyncio.create_task(asyncio.to_thread(self.perception.detect, packet.jpeg, self.planner.leg))
            try:
                while not task.done():
                    self.poll("OBSERVE")
                    camera_gap = camera_gap or not self.source.healthy(time.monotonic())
                    # Camera waiting must not consume the detector's own budget.
                    if time.monotonic() >= started+self.cfg.inference_timeout:
                        raise TimeoutError("Stationary inference timed out; reduce image size or use a faster model")
                    await asyncio.sleep(.01)
                sighting, self.display = task.result()
                self.detector_ms = round((time.monotonic()-started)*1000, 1)
                if camera_gap or not self.source.healthy(time.monotonic()):
                    raise CameraUnavailable('Camera was interrupted during stationary inference')
                if self.telemetry:
                    self.telemetry.publish_observation(self.display)
                elif self.preview_path:
                    # Legacy callers without telemetry must also survive a busy
                    # preview file. Web missions publish on the background writer.
                    try:
                        temporary = self.preview_path.with_name("preview.tmp.jpg")
                        if cv2.imwrite(str(temporary), self.display):
                            os.replace(temporary, self.preview_path)
                    except (OSError, cv2.error) as exc:
                        print(f'Preview update skipped: {exc}', flush=True)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            age = time.monotonic()-packet.received_at
            if age > self.cfg.max_result_age:
                self.event("stale_result", frame_age_ms=age*1000)
                # A slow/broken detector is not evidence that the scene is empty.
                stale_inference_seconds += time.monotonic()-started
                if stale_inference_seconds >= self.cfg.observation_timeout:
                    raise TimeoutError("Inference is too slow for max_result_age; no movement was planned")
                continue
            if sighting is not None:
                sighting = replace(sighting, observed_at=packet.received_at)
            samples.append(sighting)
            stale_inference_seconds = 0.0
            self.frame_number += 1
            if self.save_frames:
                directory = self.session/"observations"
                directory.mkdir(exist_ok=True)
                (directory/f"{self.frame_number:06d}_{self.planner.leg}.jpg").write_bytes(packet.jpeg)
            fields = dict(sequence=packet.sequence, frame_age_ms=age*1000,
                          inference_ms=(time.monotonic()-started)*1000)
            if sighting:
                fields.update(identity=sighting.identity, track_id=sighting.track_id, x=sighting.x, size=sighting.size)
            self.event("stationary_frame", **fields)
        return confirmed_sighting(samples, self.cfg)

    async def run(self, max_batches=0):
        if not self.headless:
            cv2.namedWindow(self.window)
        batches = 0
        try:
            # Connecting the phone and granting camera permission can take a while.
            # There is no motion and no observation deadline until the first frame.
            while not self.source.healthy(time.monotonic()):
                self.poll("WAIT_CAMERA")
                if self.source.eof:
                    raise VideoEnded
                await asyncio.sleep(.02)
            while self.planner.leg != "done":
                try:
                    await self.run_batch()
                except CameraUnavailable as exc:
                    await self.recover_camera(exc)
                    continue  # incomplete bursts cannot generate search/movement
                batches += 1
                if max_batches and batches >= max_batches:
                    return "batch limit reached"
            self.poll("DONE")
            return {'object_only': 'object reached', 'tag_only': 'AprilTag reached'}.get(
                self.cfg.mission_mode, 'object and AprilTag reached')
        except VideoEnded:
            self.event("video_ended")
            return "video ended; replay is a detector/decision test, not proof of a completed physical mission"
        finally:
            self.file.close()
            cv2.destroyAllWindows()

    async def run_batch(self):
        self.apply_color_request()
        ready = getattr(self.perception, 'ready', lambda leg: True)
        while not ready(self.planner.leg):
            # An unset HSV color must never turn into an automatic search pulse.
            self.poll('WAIT_COLOR')
            if self.source.eof:
                raise VideoEnded
            if not self.source.healthy(time.monotonic()):
                raise CameraUnavailable('Camera frames became stale while waiting for a color')
            self.apply_color_request()
            await asyncio.sleep(.02)
        sighting = await self.observe()
        if self.body is not None and self.pending_body is not None:
            before, impulse, ended = self.pending_body
            if (self.cfg.body_learning=='tag_online' and sighting is not None and sighting.kind=='tag'
                    and sighting.identity==before.identity and sighting.kind==before.kind
                    and time.monotonic()-ended <= self.cfg.position_memory_seconds):
                from robotics_lab.body_model import feature
                accepted = self.body.adapt(impulse,feature(sighting)-feature(before))
                if self.telemetry:
                    self.telemetry.update(body_updates=self.body.updates,body_update_accepted=accepted)
            self.pending_body = None
        if self.apply_color_request():
            return  # discard the old-color burst; observe again with the new color
        self.poll("PLAN")
        step = self.planner.plan(sighting, time.monotonic())
        if self.body is not None:
            step = self.body.choose(sighting,step,self.cfg)
        self.step_name = step.name
        holds = step.wheel_holds or (step.hold_seconds,step.hold_seconds)
        self.event("decision", step=step.name, left=step.left, right=step.right, hold_seconds=step.hold_seconds,
                   left_hold_seconds=holds[0],right_hold_seconds=holds[1])
        print(f"{self.planner.leg}: {step.name}", flush=True)
        print(f"Command requested: {step.name} | left={step.left:+.3f}, right={step.right:+.3f} "
              f"| hold={step.hold_seconds:.3f}s | {'DRIVE' if self.driver.drive else 'PREVIEW'}", flush=True)
        if step.wheel_holds:
            print(f'Independent wheel holds: left={holds[0]:.3f}s, right={holds[1]:.3f}s',flush=True)
        if step.left != 0 or step.right != 0:
            started = time.monotonic()
            await self.driver.pulse(step, self.motion_tick)
            self.planner.motion_completed(step, time.monotonic()-started, time.monotonic())
            if self.body is not None and sighting is not None and self.driver.drive and self.driver.last_impulse is not None:
                self.pending_body = (sighting,self.driver.last_impulse,time.monotonic())
            self.event("motors_stopped")

    def apply_color_request(self):
        if self.cfg.object_provider != 'hsv' or self.planner.leg != 'object' or not self.command_directory:
            return False
        requests = sorted(self.command_directory.glob('.color-request-*.json'))
        if not requests or requests[-1].name <= self.last_color_request:
            return False
        path = requests[-1]
        data = json.loads(path.read_text(encoding='utf-8'))
        self.perception.set_color(data['hsv'])
        self.planner.reset_memory()
        self.pending_body = None
        self.last_color_request = path.name
        print(f"Color selected: HSV {data['hsv']}. Starting a fresh stationary observation.", flush=True)
        if self.telemetry:
            self.telemetry.update(selected_hsv=data['hsv'], color_request=data['id'])
        return True


async def continuous_preview(runner, limit=0):
    """Latest-frame detector display. No planning, pulse or Arduino connection."""
    if runner.driver.drive:
        raise ValueError('Continuous preview cannot enable motors')
    sequence = count = 0
    try:
        while not runner.source.healthy(time.monotonic()):
            runner.poll('WAIT_CAMERA')
            if runner.source.eof:
                raise VideoEnded
            await asyncio.sleep(.01)
        runner.perception.begin_burst()
        while True:
            try:
                runner.apply_color_request()
                if not getattr(runner.perception, 'ready', lambda leg: True)(runner.planner.leg):
                    runner.poll('WAIT_COLOR')
                    if runner.source.eof:
                        raise VideoEnded
                    await asyncio.sleep(.01)
                    continue
                packet = await runner.wait_new(sequence, time.monotonic()+runner.cfg.observation_timeout, 'DETECT_PREVIEW')
                sequence = packet.sequence
                started = time.monotonic()
                task = asyncio.create_task(asyncio.to_thread(runner.perception.detect, packet.jpeg, runner.planner.leg))
                try:
                    while not task.done():
                        runner.poll('DETECT_PREVIEW')
                        if time.monotonic()-started > runner.cfg.inference_timeout:
                            raise TimeoutError('Detector preview inference timed out')
                        await asyncio.sleep(.002)
                    sighting, runner.display = task.result()
                    runner.detector_ms = round((time.monotonic()-started)*1000, 1)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                if runner.apply_color_request():
                    continue
                if runner.telemetry:
                    runner.telemetry.publish_observation(runner.display)
                fields = dict(inference_ms=runner.detector_ms, frame_age_ms=(time.monotonic()-packet.received_at)*1000)
                if sighting:
                    fields.update(identity=sighting.identity, x=sighting.x, size=sighting.size, track_id=sighting.track_id)
                runner.event('preview_frame', **fields)
                count += 1
                if limit and count >= limit:
                    return 'preview frame limit reached'
            except CameraUnavailable as exc:
                await runner.recover_camera(exc)
    except VideoEnded:
        return 'preview video ended'
    finally:
        runner.file.close()
        cv2.destroyAllWindows()


async def main(args):
    telemetry = getattr(args, 'telemetry', None)
    if telemetry:
        telemetry.update(phase='LOADING_MODEL')
    cfg = getattr(args, "config", None) or replace(Config(), weights=args.weights, target_class=args.target_class,
                                                  arduino_ip=args.arduino_ip, tag_id=args.tag_id)
    cfg.validate()
    if getattr(args,'calibrate_body',False):
        if not args.drive or args.video:
            raise ValueError('Body calibration requires explicit driving and a live robot camera')
        cfg = replace(cfg,mission_mode='tag_only',body_control='off')
        from robotics_lab.body_model import body_path
        if body_path(cfg).exists():
            raise ValueError('Choose a new body_model filename; calibration never overwrites profiles')
    body = None
    if cfg.body_control == 'learned' and not getattr(args,'continuous_preview',False):
        from robotics_lab.body_model import VisualBodyModel, body_path
        body = VisualBodyModel.load(body_path(cfg),cfg)  # validate before opening hardware
    if getattr(args, 'continuous_preview', False) and args.drive:
        raise ValueError('Continuous detector preview cannot enable motors')
    perception = Perception(cfg)  # validate weights/class before opening motor connection
    print("Preparing perception before opening the camera or motor connection...", flush=True)
    if telemetry:
        telemetry.update(phase='WARMUP')
    await asyncio.wait_for(asyncio.to_thread(perception.warmup), timeout=60)
    if getattr(args, "stop_file", None) and args.stop_file.exists():
        raise StopRequested
    source = Frames(cfg, args.video)
    driver = build('motion_driver', cfg, drive=args.drive)
    session = RUNS/"stop_and_stare"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True)
    try:
        if telemetry:
            telemetry.update(phase='START_CAMERA')
        source.open()
        if telemetry:
            telemetry.update(phase='CONNECTING_ROBOT' if args.drive else 'WAIT_CAMERA',
                             phone_urls=list(source.cam.phone_urls) if source.cam else [],
                             session=str(session))
        await driver.open()
        print("DRIVING enabled." if args.drive else "PREVIEW ONLY: no Arduino connection or motor commands.")
        print("Session:", session)
        runner = Runner(cfg, source, perception, driver, session,
                        headless=args.headless, save_frames=args.save_frames,
                        stop_file=getattr(args, "stop_file", None), preview_path=getattr(args, "preview_path", None),
                        telemetry=telemetry, body=body)
        if getattr(args, 'calibrate_body', False):
            from missions.stop_and_stare.calibration import calibrate
            result = await calibrate(runner)
        elif getattr(args, 'continuous_preview', False):
            result = await continuous_preview(runner, args.max_batches)
        else:
            result = await runner.run(args.max_batches)
        print(result)
        return session
    finally:
        try:
            await driver.close()
        finally:
            source.close()


def arguments():
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=defaults.weights)
    parser.add_argument("--config", type=Path, help="JSON Config values (CLI model/class/IP flags ignored when supplied)")
    parser.add_argument("--stop-file", type=Path, help="Stop cooperatively when this marker exists")
    parser.add_argument("--target-class", default=defaults.target_class)
    parser.add_argument("--tag-id", type=int, default=defaults.tag_id)
    parser.add_argument("--arduino-ip", default=defaults.arduino_ip)
    parser.add_argument("--drive", action="store_true", help="Enable actual robot movement; default is preview only")
    parser.add_argument('--continuous-preview', action='store_true', help='Continuous detector display, motors disabled')
    parser.add_argument('--calibrate-body', action='store_true', help='Learn camera response to motor pulses using a fixed AprilTag; requires --drive')
    parser.add_argument("--video", type=Path, help="Replay a video, never connect to the robot")
    parser.add_argument("--headless", action="store_true", help="Replay without a preview window")
    parser.add_argument("--save-frames", action="store_true", help="Save accepted stationary images with session logs")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit observation batches; 0 runs until done or quit")
    args = parser.parse_args()
    if args.continuous_preview and (args.drive or args.calibrate_body):
        parser.error('Continuous preview cannot drive or calibrate')
    if args.calibrate_body and (not args.drive or args.video):
        parser.error('Body calibration requires --drive and a live robot camera')
    if args.video and args.drive:
        parser.error("--video and --drive cannot be combined")
    if args.config:
        from dataclasses import fields
        values = json.loads(args.config.read_text(encoding="utf-8"))
        if set(values) - {f.name for f in fields(Config)}:
            parser.error("Unknown Config field")
        if "weights" in values:
            values["weights"] = Path(values["weights"])
        args.config = Config(**values)
        args.config.validate()
    if args.max_batches < 0:
        parser.error("--max-batches must be >= 0")
    return args


def cli():
    try:
        asyncio.run(main(arguments()))
    except (StopRequested, KeyboardInterrupt):
        print("Stopped.")


if __name__ == "__main__":
    cli()
