"""Camera-gap recovery and independent detector deadlines; no physical hardware."""

import asyncio
import csv
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np

from missions.stop_and_stare.camera import Packet
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Sighting
from missions.stop_and_stare.motion import PulseDriver
from missions.stop_and_stare.run import Runner, StopRequested
from robotics_lab.motor_control import SmoothMotorController


class Source:
    def __init__(self):
        self.latest=None
        self.sequence=0
        self.eof=False
    def poll(self, now):
        self.sequence+=1
        self.latest=Packet(self.sequence,now,b'frame')
    def healthy(self, now):return self.latest is not None


class Detector:
    def __init__(self):self.calls=0;self.bursts=0
    def begin_burst(self,preferred):self.bursts+=1
    def detect(self,jpeg,leg):
        self.calls+=1
        time.sleep(.003)
        return Sighting(leg,'bottle' if leg=='object' else 'tag:0',0,.9,1),np.zeros((30,40,3),np.uint8)


class CameraRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def config(self,**values):
        defaults=dict(settle_seconds=.002,discard_frames=1,observe_frames=2,confirmations=2,
                      observation_timeout=.1,inference_timeout=.3,camera_reconnect_timeout=.3,
                      motor_interval=.001,acceleration=10000,deceleration=10000)
        return replace(Config(),**(defaults|values))

    async def test_slow_frame_arrivals_do_not_consume_inference_deadline(self):
        class DelayedFrameRunner(Runner):
            async def wait_new(self,after,deadline,phase):
                # Each arrival fits a .2s frame allowance, while several waits
                # exceed the old shared burst deadline before inference starts.
                await asyncio.sleep(.12)
                self.poll(phase)
                return self.source.latest
        class SlowDetector(Detector):
            def detect(self,*args):
                time.sleep(.025)
                return super().detect(*args)
        cfg=self.config(observation_timeout=.2,discard_frames=2,max_result_age=2,inference_timeout=1)
        with tempfile.TemporaryDirectory() as folder:
            runner=DelayedFrameRunner(cfg,Source(),SlowDetector(),PulseDriver(cfg),Path(folder),headless=True)
            try:
                result=await runner.observe()
                self.assertIsNotNone(result)
                self.assertEqual(runner.perception.calls,2)
            finally:runner.file.close()

    async def exercise_gap(self,moving):
        cfg=self.config(forward_min_seconds=.02,forward_max_seconds=.02)
        holder={}
        class GapSource(Source):
            dropped=False
            resumed=False
            resume_at=0
            def poll(self,now):
                runner=holder['runner']
                trigger=runner.phase=='MOVE' if moving else runner.phase=='OBSERVE' and runner.perception.calls>=1
                if trigger and not self.dropped:
                    self.dropped=True;self.resume_at=now+.06
                if self.dropped and now<self.resume_at:return
                if self.dropped:self.resumed=True
                super().poll(now)
            def healthy(self,now):return self.latest is not None and (not self.dropped or self.resumed)
        source=GapSource()
        class GapDetector(Detector):
            def detect(self,jpeg,leg):
                result,image=super().detect(jpeg,leg)
                if moving and not source.resumed:
                    result=Sighting(leg,'bottle',0,.01,1)
                return result,image
        driver=PulseDriver(cfg)
        if moving:
            driver.control=SmoothMotorController(AsyncMock(),driver.motors,acceleration=10000,deceleration=10000)
            await driver.control.stop()
        else:driver.stop=AsyncMock()
        with tempfile.TemporaryDirectory() as folder:
            runner=Runner(cfg,source,GapDetector(),driver,Path(folder),headless=True)
            holder['runner']=runner
            original_poll=runner.poll
            def checked_poll(phase):
                if phase=='WAIT_RECONNECT' and moving:self.assertTrue(driver.control.is_stopped)
                original_poll(phase)
            runner.poll=checked_poll
            self.assertEqual(await runner.run(),'object and AprilTag reached')
            with (Path(folder)/'events.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            gap=next(i for i,r in enumerate(rows) if r['event']=='camera_gap')
            resumed=next(i for i,r in enumerate(rows) if r['event']=='camera_reconnected')
            self.assertFalse(any(r['event']=='decision' for r in rows[gap:resumed]))
            self.assertTrue(source.resumed)
            self.assertGreaterEqual(runner.perception.bursts,3)
            if moving:self.assertTrue(driver.control.is_stopped)
            else:driver.stop.assert_awaited()

    async def test_observation_drop_discards_burst_and_waits_without_searching(self):
        await self.exercise_gap(False)

    async def test_moving_drop_stops_pwm_before_waiting_and_reobserves(self):
        await self.exercise_gap(True)

    async def test_real_inference_timeout_remains_bounded(self):
        class HungDetector(Detector):
            def detect(self,*args):
                time.sleep(.06)
                return super().detect(*args)
        cfg=self.config(inference_timeout=.01)
        with tempfile.TemporaryDirectory() as folder:
            driver=PulseDriver(cfg);driver.pulse=AsyncMock()
            runner=Runner(cfg,Source(),HungDetector(),driver,Path(folder),headless=True)
            with self.assertRaisesRegex(TimeoutError,'Stationary inference timed out'):
                await runner.run()
            driver.pulse.assert_not_awaited()

    async def test_stop_works_during_reconnect_wait(self):
        class Offline(Source):
            def poll(self,now):pass
            def healthy(self,now):return False
        cfg=self.config()
        with tempfile.TemporaryDirectory() as folder:
            stop=Path(folder)/'stop'
            driver=PulseDriver(cfg);driver.stop=AsyncMock()
            runner=Runner(cfg,Offline(),Detector(),driver,Path(folder),headless=True,stop_file=stop)
            async def request_stop():
                await asyncio.sleep(.025);stop.touch()
            task=asyncio.create_task(request_stop())
            try:
                with self.assertRaises(StopRequested):await runner.recover_camera('test gap')
                driver.stop.assert_awaited_once()
            finally:
                await task;runner.file.close()

    async def test_reconnect_wait_has_a_configured_limit(self):
        class Offline(Source):
            def poll(self,now):pass
            def healthy(self,now):return False
        cfg=self.config(camera_reconnect_timeout=.025)
        with tempfile.TemporaryDirectory() as folder:
            runner=Runner(cfg,Offline(),Detector(),PulseDriver(cfg),Path(folder),headless=True)
            try:
                with self.assertRaisesRegex(TimeoutError,'No fresh camera frames within'):
                    await runner.recover_camera('test gap')
            finally:runner.file.close()


if __name__=='__main__':unittest.main()
