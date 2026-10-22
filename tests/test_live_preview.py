"""In-memory JPEG IPC and continuous detector preview, without robot movement."""

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import AsyncMock

import cv2
import numpy as np

from missions.stop_and_stare.camera import Packet
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.motion import PulseDriver
from missions.stop_and_stare.perception import Perception
from missions.stop_and_stare.run import Runner,continuous_preview
from missions.stop_and_stare.telemetry import MissionTelemetry
from robotics_lab.live_camera import LiveCamera
from webapp.app import create_app


class LiveTests(unittest.TestCase):
    def test_ipc_keeps_latest_frame_and_authenticates_clients(self):
        live=LiveCamera()
        base=f'http://127.0.0.1:{live.server.server_port}'
        try:
            with self.assertRaises(HTTPError):urlopen(base+'/frame',timeout=2)
            for i in range(50):live.publish(str(i).encode(),i)
            request=Request(base+'/frame',headers={'X-Preview-Token':live.token})
            with urlopen(request,timeout=2) as response:self.assertEqual(response.read(),b'49')
            live.publish(b'annotated',1,'observation')
            request=Request(base+'/frame?channel=observation',headers={'X-Preview-Token':live.token})
            with urlopen(request,timeout=2) as response:self.assertEqual(response.read(),b'annotated')
            self.assertEqual(len(live.frames),2)
        finally:live.close()

    def test_live_raw_publication_does_not_write_a_file_per_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            telemetry=MissionTelemetry(Path(folder))
            try:
                for i in range(100):telemetry.update(jpeg=b'jpeg'+str(i).encode(),sequence=i)
                self.assertEqual(telemetry.live.frames['camera'],(99,b'jpeg99'))
            finally:telemetry.close()
            self.assertLessEqual(len(list((Path(folder)/'.display').glob('camera_*.jpg'))),2)

    def test_jobs_reject_unsafe_preview_calibration_combinations(self):
        with tempfile.TemporaryDirectory() as folder:
            app=create_app(Path(folder),testing=True);jobs=app.extensions['jobs']
            try:
                directory=Path(folder)/'test';directory.mkdir()
                base=dict(config=dict(mission_mode='tag_only'))
                for extra in (dict(continuous_preview=True,drive=True),dict(calibrate_body=True,drive=False),
                              dict(calibrate_body=True,drive=True,video='some.avi')):
                    with self.subTest(extra=extra),self.assertRaises(ValueError):
                        jobs.prepare('mission',{**base,**extra},directory)
                accepted=jobs.prepare('mission',{**base,'continuous_preview':True},directory)
                self.assertFalse(accepted['drive'])
            finally:jobs.close()


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_detection_skips_settling_and_never_plans_or_pulses(self):
        cfg=replace(Config(),object_provider='hsv',color_h=0,color_s=255,color_v=255,mission_mode='object_only')
        image=np.zeros((120,160,3),np.uint8);image[30:90,40:120]=(0,0,255)
        encoded=cv2.imencode('.jpg',image)[1].tobytes()
        class Source:
            latest=None;sequence=0;eof=False
            def poll(self,now):
                self.sequence+=1;self.latest=Packet(self.sequence,now,encoded)
            def healthy(self,now):return True
        with tempfile.TemporaryDirectory() as folder:
            driver=PulseDriver(cfg);driver.pulse=AsyncMock()
            runner=Runner(cfg,Source(),Perception(cfg),driver,Path(folder),headless=True)
            runner.observe=AsyncMock(side_effect=AssertionError('stop-and-stare burst in continuous preview'))
            await continuous_preview(runner,limit=12)
            runner.observe.assert_not_awaited()
            driver.pulse.assert_not_awaited()
            log=(Path(folder)/'events.csv').read_text()
            self.assertEqual(log.count('preview_frame'),12)
            self.assertNotIn(',decision,',log)
            self.assertNotIn('SETTLE',log)


if __name__=='__main__':unittest.main()
