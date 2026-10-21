"""Color selection, interchangeable control and estimation; never connects motors."""

import asyncio
import json
import tempfile
import time
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np

from missions.stop_and_stare.camera import Packet
from missions.stop_and_stare.color import HSVObjects, sample_hsv
from missions.stop_and_stare.components import STEERING
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Planner, Sighting, Step
from missions.stop_and_stare.estimation import KalmanPosition
from missions.stop_and_stare.motion import PulseDriver
from missions.stop_and_stare.perception import Perception
from missions.stop_and_stare.run import Runner, StopRequested
from missions.stop_and_stare.steering import PIDSteering
from robotics_lab.snapshots import publish_snapshot
from webapp.app import create_app
from webapp.help import CONFIG_HELP
from webapp.storage import write_json


def jpeg(frame):
    ok, encoded = cv2.imencode('.jpg', frame)
    assert ok
    return encoded.tobytes()


def solid(hue=0):
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:] = (hue, 255, 255)
    return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)


class ComponentsTests(unittest.TestCase):
    def test_color_wrap_area_and_noise_rejection(self):
        cfg = replace(Config(), object_provider='hsv', color_h=0, color_s=255, color_v=255)
        detector = HSVObjects(cfg)
        hsv = np.zeros((120, 160, 3), np.uint8)
        hsv[30:70,20:60] = (179,255,255)  # red wraps across hue 0
        hsv[1:3,1:3] = (0,255,255)  # speckle
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        sighting, _ = detector.detect(image)
        self.assertAlmostEqual(sighting.size, 1600/(120*160))
        self.assertAlmostEqual(sighting.x, -.5)
        detector.set_color([90,255,255])
        self.assertIsNone(detector.detect(image)[0])

    def test_hsv_provider_needs_no_weights_or_apriltag_in_object_only_mode(self):
        cfg = replace(Config(), object_provider='hsv', mission_mode='object_only', weights=Path('missing.pt'))
        with patch.dict('sys.modules', {'ultralytics': None, 'pupil_apriltags': None}):
            perception = Perception(cfg)
            perception.warmup()
        self.assertFalse(perception.ready('object'))
        perception.set_color(sample_hsv(jpeg(solid()), .5, .5))
        self.assertTrue(perception.ready('object'))
        self.assertIsNotNone(perception.detect(jpeg(solid()), 'object')[0])

    def test_blob_selection_uses_remembered_side(self):
        detector = HSVObjects(replace(Config(), color_h=0, color_s=255, color_v=255))
        image = np.zeros((120,160,3),np.uint8)
        image[40:65,10:35] = (0,0,255)
        image[20:80,100:150] = (0,0,255)
        detector.begin_burst(-.8)
        self.assertLess(detector.detect(image)[0].x, 0)
        detector.begin_burst()
        self.assertGreater(detector.detect(image)[0].x, 0)

    def test_pid_bounds_resets_windup_and_follows_fresh_error(self):
        cfg = replace(Config(), steering_controller='pid', pid_ki=2.)
        pid = PIDSteering(cfg)
        for now in range(1, 30):
            duration = pid.duration(.9, now)
            self.assertLessEqual(duration,cfg.turn_max_seconds)
            self.assertGreaterEqual(duration,cfg.turn_min_seconds)
        self.assertLessEqual(abs(pid.integral),cfg.pid_integral_limit)
        self.assertEqual(pid.integral,0)  # saturation prevented windup
        planner = Planner(cfg)
        self.assertEqual(planner.plan(Sighting('object','bottle',.8,.01),1).name,'turn_right')
        self.assertEqual(planner.plan(Sighting('object','bottle',-.2,.01),2).name,'turn_left')
        planner.plan(None,3)
        self.assertIsNone(planner.steering.previous)

    def test_pid_uses_actual_observation_intervals(self):
        cfg = replace(Config(), pid_kp=.1, pid_kd=0., pid_ki=.1, turn_min_seconds=.001)
        short, long = PIDSteering(cfg), PIDSteering(cfg)
        short.duration(.2,1); long.duration(.2,1)
        self.assertGreater(long.duration(.2,3),short.duration(.2,1.1))

    def test_kalman_smooths_noise_and_expires_uncertain_predictions(self):
        cfg = replace(Config(), position_estimator='kalman')
        estimator = KalmanPosition(cfg)
        values = [estimator.update(.4+(i%2)*.04,1+i*.1) for i in range(20)]
        self.assertLess(np.std(values[10:]), .02)
        self.assertIsNotNone(estimator.predict(2.95))
        self.assertIsNone(estimator.predict(100))
        estimator.moved(Step('turn',.2,-.2,.1),2,3)
        self.assertEqual(estimator.state[1],0)
        self.assertIsNone(estimator.predict(3))  # motion covariance too large
        self.assertTrue(np.all(np.linalg.eigvalsh(estimator.covariance)>0))

    def test_predictions_never_authorize_forward_or_arrival(self):
        cfg = replace(Config(), position_estimator='kalman', steering_controller='pid')
        planner = Planner(cfg)
        planner.plan(Sighting('object','bottle',-.5,.05),1)
        for now in (1.1,1.2,1.3):
            step=planner.plan(None,now)
            self.assertIn(step.name,{'look_again','search_left'})
            self.assertEqual(step.left,-step.right)
        # A fresh off-center detection cannot be hidden by a centered filter.
        planner.reset_memory()
        planner.plan(Sighting('object','bottle',0,.05),2)
        self.assertEqual(planner.plan(Sighting('object','bottle',.3,.9),2.01).name,'turn_right')
        self.assertEqual(planner.leg,'object')

    def test_components_schema_validation_and_complete_help(self):
        self.assertEqual({f.name for f in fields(Config)},set(CONFIG_HELP))
        for key,value in [('object_provider','arbitrary.module'),('color_h',180),('color_min_area',0),
                          ('pid_kp',-1.),('filter_measurement_noise',0.),('position_estimator','bad'),
                          ('motion_driver','shell'),('color_v_tolerance',float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(Config(),**{key:value}).validate()
        class CustomSteering:
            def __init__(self,cfg): pass
            def reset(self): pass
            def duration(self,error,now): return .123
        with patch.dict(STEERING, {'test_adapter':CustomSteering}):
            cfg=replace(Config(),steering_controller='test_adapter')
            cfg.validate()
            self.assertEqual(Planner(cfg).plan(Sighting('object','bottle',.5,.01),1).hold_seconds,.123)


class ColorAPITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.app=create_app(self.root,testing=True)
        self.client=self.app.test_client()
        self.headers={'X-CSRF-Token':self.client.get('/api/session').json['csrf']}
        self.jobs=self.app.extensions['jobs']
        self.ticket='a'*32
        self.directory=self.root/'.control/jobs'/self.ticket
        self.directory.mkdir()
        self.jobs.records[self.ticket]=dict(id=self.ticket,kind='mission',status='running',
            params=dict(config=dict(object_provider='hsv',mission_mode='object_only')))
        publish_snapshot(self.directory,'camera.jpg',jpeg(solid(0)))

    def tearDown(self):
        self.jobs.records[self.ticket]['status']='succeeded'
        self.jobs.close()
        self.temp.cleanup()

    def post(self,path,data=None):
        return self.client.post(f'/api/jobs/{self.ticket}/{path}',json=data or {},headers=self.headers)

    def test_exact_frozen_frame_is_sampled_after_live_image_changes(self):
        frame=self.post('color-frame').json
        publish_snapshot(self.directory,'camera.jpg',jpeg(solid(60)))
        self.assertEqual(self.client.get(frame['image']).status_code,200)
        result=self.post('color',dict(token=frame['token'],x=.5,y=.5))
        self.assertEqual(result.status_code,200,result.json)
        self.assertTrue(result.json['hsv'][0]<3 or result.json['hsv'][0]>177)
        command=json.loads(next(self.directory.glob('.color-request-*.json')).read_text())
        self.assertEqual(command['hsv'],result.json['hsv'])
        self.assertEqual(len(list((self.directory/'.color-picker').glob('*.jpg'))),1)

    def test_reject_invalid_coordinates_tokens_modes_and_csrf(self):
        frame=self.post('color-frame').json
        for coords in [(True,.5),(float('nan'),.5),(-.1,.5),(1.1,.5),(.5,None)]:
            self.assertEqual(self.post('color',dict(token=frame['token'],x=coords[0],y=coords[1])).status_code,400)
        self.assertEqual(self.post('color',dict(token='../job.json',x=.5,y=.5)).status_code,400)
        self.assertEqual(self.client.post(f'/api/jobs/{self.ticket}/color-frame',json={}).status_code,403)
        self.jobs.records[self.ticket]['params']['config']['object_provider']='yolo'
        self.assertEqual(self.post('color-frame').status_code,400)
        self.jobs.records[self.ticket]['params']['config']['object_provider']='hsv'
        publish_snapshot(self.directory,'runtime.json',b'{"leg":"tag"}')
        self.assertEqual(self.post('color-frame').status_code,400)

    def test_pinned_frames_are_bounded_and_color_missions_skip_model_validation(self):
        for _ in range(7): self.assertEqual(self.post('color-frame').status_code,200)
        self.assertLessEqual(len(list((self.directory/'.color-picker').glob('*.jpg'))),4)
        prepared=self.jobs.prepare('mission',dict(config=dict(object_provider='hsv',mission_mode='object_only')),self.directory)
        self.assertFalse(prepared['drive'])
        self.assertEqual(prepared['config']['color_h'],-1)


class ColorRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_unset_color_never_searches_and_stop_is_responsive(self):
        cfg=replace(Config(),object_provider='hsv',mission_mode='object_only')
        perception=Perception(cfg)
        class Source:
            latest=None; sequence=0; eof=False
            def poll(self,now): pass
            def healthy(self,now): return True
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            stop=directory/'stop'
            driver=PulseDriver(cfg);driver.pulse=AsyncMock()
            runner=Runner(cfg,Source(),perception,driver,directory,headless=True,stop_file=stop)
            async def request_stop():
                await asyncio.sleep(.04);stop.touch()
            task=asyncio.create_task(request_stop())
            try:
                with self.assertRaises(StopRequested): await runner.run_batch()
                self.assertEqual(runner.phase,'WAIT_COLOR')
                driver.pulse.assert_not_awaited()
            finally:
                await task;runner.file.close()

    async def test_new_color_discards_old_observation_before_planning(self):
        cfg=replace(Config(),object_provider='hsv',mission_mode='object_only',color_h=0)
        perception=Perception(cfg)
        class Source:
            latest=None; sequence=0; eof=False
            def poll(self,now): pass
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            driver=PulseDriver(cfg);driver.pulse=AsyncMock()
            runner=Runner(cfg,Source(),perception,driver,directory,headless=True,stop_file=directory/'stop')
            async def observe():
                write_json(directory/'.color-request-0001.json',dict(id='test',hsv=[60,255,255]))
                return Sighting('object','bottle',0,.01)
            runner.observe=observe
            try:
                await runner.run_batch()
                driver.pulse.assert_not_awaited()
                self.assertEqual(perception.objects.hsv,(60,255,255))
                self.assertEqual(runner.planner.last_seen,-float('inf'))
            finally: runner.file.close()

    async def test_real_opencv_color_then_real_apriltag_mission(self):
        cfg=replace(Config(),object_provider='hsv',color_h=0,color_s=255,color_v=255,
                    steering_controller='pid',position_estimator='kalman',settle_seconds=.005)
        color=np.zeros((240,320,3),np.uint8);color[70:170,85:235]=(0,0,255)
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            tag=cv2.aruco.generateImageMarker(dictionary,0,120)
            tag=cv2.copyMakeBorder(tag,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
            tag=cv2.cvtColor(tag,cv2.COLOR_GRAY2BGR)
            tag=cv2.resize(tag,(160,160),interpolation=cv2.INTER_NEAREST)
            image=np.full((240,320,3),255,np.uint8);image[40:200,80:240]=tag
            images={'object':jpeg(color),'tag':jpeg(image)}
            class Source:
                latest=None; sequence=0; eof=False
                def poll(self,now):
                    self.sequence+=1
                    self.latest=Packet(self.sequence,now,images.get(runner.planner.leg,images['tag']))
                def healthy(self,now): return True
            driver=PulseDriver(cfg);driver.pulse=AsyncMock()
            runner=Runner(cfg,Source(),Perception(cfg),driver,directory,headless=True)
            result=await runner.run(max_batches=3)
            self.assertEqual(runner.planner.leg,'done',result)
            driver.pulse.assert_not_awaited()


if __name__=='__main__': unittest.main()
