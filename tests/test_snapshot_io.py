"""Windows snapshot sharing and display failure isolation; no hardware."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np

from robotics_lab.snapshots import publish_snapshot, read_snapshot
from missions.stop_and_stare.camera import Packet
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Sighting
from missions.stop_and_stare.motion import PulseDriver
from missions.stop_and_stare.run import Runner
from missions.stop_and_stare.telemetry import MissionTelemetry


class SnapshotSharingTests(unittest.TestCase):
    def test_reader_can_stay_open_while_new_snapshots_are_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=publish_snapshot(root,'runtime.json',b'old snapshot')
            with path.open('rb') as reader:
                for n in range(30):
                    publish_snapshot(root,'runtime.json',str(n).encode())
                    self.assertEqual(read_snapshot(root,'runtime.json'),str(n).encode())
                self.assertEqual(reader.read(),b'old snapshot')
            publish_snapshot(root,'runtime.json',b'final')
            self.assertLessEqual(len(list((root/'.display').glob('*.json'))),4)

    @unittest.skipUnless(os.name=='nt','Windows sharing regression')
    def test_ordinary_windows_reader_reproduces_the_reported_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'preview.jpg'
            temporary=Path(folder)/'preview.tmp'
            path.write_bytes(b'old')
            temporary.write_bytes(b'new')
            with path.open('rb'):
                with self.assertRaises(PermissionError):
                    os.replace(temporary,path)
            os.replace(temporary,path)


class PreviewFailureTests(unittest.IsolatedAsyncioTestCase):
    def build_runner(self, root, telemetry=None):
        cfg=replace(Config(),settle_seconds=.01,discard_frames=1,observe_frames=2,confirmations=2)
        class Source:
            latest=None
            sequence=0
            eof=False
            def poll(self, now):
                self.sequence+=1
                self.latest=Packet(self.sequence,now,b'raw')
            def healthy(self,now):
                return True
        class Detector:
            def begin_burst(self,preferred):pass
            def detect(self,jpeg,leg):
                key='bottle' if leg=='object' else 'tag:0'
                return Sighting(leg,key,0,.9,1),np.zeros((30,40,3),np.uint8)
        driver=PulseDriver(cfg)
        driver.pulse=AsyncMock()
        return Runner(cfg,Source(),Detector(),driver,root,headless=True,
                      preview_path=root/'preview.jpg',telemetry=telemetry)

    async def test_legacy_preview_permission_error_cannot_abort_mission(self):
        with tempfile.TemporaryDirectory() as folder:
            runner=self.build_runner(Path(folder))
            with patch('missions.stop_and_stare.run.os.replace',side_effect=PermissionError('file in use')):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(await runner.run(),'object and AprilTag reached')
            runner.driver.pulse.assert_not_awaited()

    @unittest.skipUnless(os.name=='nt','Windows sharing regression')
    async def test_locked_display_does_not_abort_control_or_status_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            preview=root/'preview.jpg'
            preview.write_bytes(cv2.imencode('.jpg',np.zeros((30,40,3),np.uint8))[1].tobytes())
            telemetry=MissionTelemetry(root)
            runner=self.build_runner(root,telemetry)
            try:
                with preview.open('rb'), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(await runner.run(),'object and AprilTag reached')
                    telemetry.close()
                state=json.loads(read_snapshot(root,'runtime.json'))
                self.assertEqual(state['phase'],'DONE')
                self.assertGreater(state['observation_sequence'],0)
                self.assertEqual(len(state['commands']),2)
                self.assertFalse(telemetry.thread.is_alive())
            finally:
                telemetry.close()

    async def test_background_preview_write_failure_does_not_abort_control(self):
        def fail_preview(directory,name,data):
            if name=='preview.jpg':
                raise PermissionError('display is blocked')
            return publish_snapshot(directory,name,data)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            telemetry=MissionTelemetry(root)
            runner=self.build_runner(root,telemetry)
            try:
                with patch('missions.stop_and_stare.telemetry.publish_snapshot',side_effect=fail_preview), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(await runner.run(),'object and AprilTag reached')
                    telemetry.close()
                state=json.loads(read_snapshot(root,'runtime.json'))
                self.assertEqual(state['phase'],'DONE')
                self.assertEqual(state['observation_sequence'],0)
            finally:
                telemetry.close()


if __name__=='__main__':
    unittest.main()
