"""Monitor telemetry, old terminal links and MJPEG streams; no real hardware."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from missions.stop_and_stare.camera import Packet
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.motion import PulseDriver
from missions.stop_and_stare.run import Runner
from missions.stop_and_stare.telemetry import MissionTelemetry
from webapp.app import create_app
from webapp.mission_status import phone_links
from robotics_lab.snapshots import read_snapshot, snapshot_exists


class TelemetryTests(unittest.TestCase):
    def test_snapshots_and_commands_flush_on_close(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            telemetry=MissionTelemetry(root, drive=False, replay=True)
            for n in range(75):
                telemetry.event(dict(event='decision',step=f'turn-{n}',left=.18,right=-.18,hold_seconds=.2,leg='object'))
            telemetry.update(phase='MOVE',jpeg=b'jpeg-bytes',sequence=7,camera_connected=True)
            telemetry.close()
            state=json.loads(read_snapshot(root,'runtime.json'))
            self.assertEqual(state['phase'],'MOVE')
            self.assertEqual(state['last_command']['step'],'turn-74')
            self.assertEqual(len(state['commands']),50)
            self.assertEqual(read_snapshot(root,'camera.jpg'),b'jpeg-bytes')
            self.assertFalse(telemetry.thread.is_alive())

    def test_motion_poll_publishes_raw_bytes_without_detection(self):
        class Source:
            latest=None
            sequence=0
            def poll(self, now):
                self.sequence+=1
                self.latest=Packet(self.sequence,now,b'raw-moving-camera')
            def healthy(self, now):
                return True
        class Detector:
            def detect(self,*args):
                raise AssertionError('Motion display must never invoke the detector')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            telemetry=MissionTelemetry(root)
            runner=Runner(Config(),Source(),Detector(),PulseDriver(Config()),root,headless=True,telemetry=telemetry)
            try:
                runner.poll('MOVE')
                telemetry.close()
                state=json.loads(read_snapshot(root,'runtime.json'))
                self.assertEqual(state['phase'],'MOVE')
                self.assertTrue(state['camera_connected'])
                self.assertEqual(read_snapshot(root,'camera.jpg'),b'raw-moving-camera')
                self.assertFalse(snapshot_exists(root,'preview.jpg'))
            finally:
                runner.file.close()
                telemetry.close()


class MonitorApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.app=create_app(self.root,testing=True)
        self.client=self.app.test_client()
        self.jobs=self.app.extensions['jobs']
        self.id='f'*32
        self.directory=self.root/'.control/jobs'/self.id
        self.directory.mkdir()
        self.jobs.records[self.id]=dict(id=self.id,kind='mission',label='Mission',status='stopped',
                                       params={'video':'','drive':False,'config':{}},created='2026')

    def tearDown(self):
        self.jobs.close()
        self.temp.cleanup()

    def test_old_phone_urls_survive_a_long_log_tail(self):
        (self.directory/'output.log').write_text('Phone URLs:\n  https://10.2.3.4:8080/phone\n'+'later log output\n'*6000)
        detail=self.client.get(f'/api/jobs/{self.id}').json
        self.assertNotIn('Phone URLs',detail['log'])
        self.assertEqual(detail['mission']['phone_urls'],['https://10.2.3.4:8080/phone'])
        self.assertEqual(self.client.get('/live').status_code,200)
        response=self.client.get(f'/api/jobs/{self.id}/artifact?name=output.log')
        self.assertGreater(len(response.data),60000)
        response.close()

    def test_phone_links_accept_only_valid_connection_addresses(self):
        links=phone_links('https://10.2.3.4:8080/phone https://10.2.3.4:8080/phone '
                          'https://evil.example:8080/phone https://user:pass@10.2.3.4:8080/phone '
                          'javascript:alert(1) https://localhost:8080/phone https://10.2.3.4:99999/phone')
        self.assertEqual(links,['https://10.2.3.4:8080/phone','https://localhost:8080/phone'])

    def test_multipart_and_observation_use_separate_images(self):
        (self.directory/'camera.jpg').write_bytes(b'raw-jpeg')
        (self.directory/'preview.jpg').write_bytes(b'annotated-stationary-jpeg')
        (self.directory/'runtime.json').write_text(json.dumps(dict(phase='MOVE',commands=[],phone_urls=[])))
        detail=self.client.get(f'/api/jobs/{self.id}').json
        self.assertTrue(detail['mission']['camera_available'])
        self.assertEqual(detail['mission']['phase'],'MOVE')
        response=self.client.get(f'/api/jobs/{self.id}/camera')
        self.assertIn('multipart/x-mixed-replace',response.content_type)
        self.assertIn(b'Content-Type: image/jpeg\r\n',response.data)
        self.assertIn(b'raw-jpeg',response.data)
        self.assertNotIn(b'annotated',response.data)
        response.close()
        response=self.client.get(f'/api/jobs/{self.id}/observation')
        self.assertEqual(response.data,b'annotated-stationary-jpeg')
        self.assertEqual(response.headers['Cache-Control'],'no-store')
        response.close()

    def test_camera_viewer_slots_released_on_disconnect_and_head(self):
        (self.directory/'camera.jpg').write_bytes(b'jpeg')
        streams=[self.client.get(f'/api/jobs/{self.id}/camera',buffered=False) for _ in range(3)]
        try:
            self.assertEqual(self.client.get(f'/api/jobs/{self.id}/camera').status_code,429)
        finally:
            for response in streams:
                response.close()
        for _ in range(5):
            response=self.client.head(f'/api/jobs/{self.id}/camera')
            self.assertEqual(response.status_code,200)
            response.close()
        response=self.client.get(f'/api/jobs/{self.id}/camera')
        self.assertEqual(response.status_code,200)
        response.close()


if __name__=='__main__':
    unittest.main()
