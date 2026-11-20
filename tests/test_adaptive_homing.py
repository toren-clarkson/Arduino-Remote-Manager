"""Body identification, visual feedback and asynchronous mission tests; no hardware."""

import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np

from robotics_lab import adaptive_control as adaptive

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('adaptive_homing', ROOT/'adaptive-homing.py')
mission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mission)


def body(matrix=None):
    # Right wheel reversed, slower and with different forward/turn response.
    j = np.array([[-.9,-.5],[.55,-.32]]) if matrix is None else np.array(matrix)
    probes = np.array([[.13,0],[-.13,0],[0,.16],[0,-.16]])
    return adaptive.BodyModel.fit(probes, probes @ j.T)


class ModelTests(unittest.TestCase):
    def test_calibration_learns_flipped_unequal_motors_without_direction_flags(self):
        model = body()
        target = np.array([-.2,.15])
        demand = model.command(target,220)
        np.testing.assert_allclose(model.jacobian @ demand,target,atol=.004)
        self.assertLess(demand[1],0, 'reversed right wheel must receive negative effort')

    def test_calibration_rejects_stall_single_axis_and_inconsistent_reversal(self):
        probes = np.array([[.1,0],[-.1,0],[0,.1],[0,-.1]])
        for changes in (np.zeros((4,2)), np.array([[.1,0],[-.1,0],[.1,0],[-.1,0]])):
            with self.assertRaises(ValueError):
                adaptive.BodyModel.fit(probes,changes)
        with self.assertRaises(ValueError):
            adaptive.BodyModel.fit(np.array([[.1,0]]*4), np.ones((4,2)))

    def test_rls_adapts_to_surface_gain_change_and_rejects_visual_jump(self):
        model = body()
        truth = model.jacobian * np.array([[1.06,.93],[1.06,.93]])
        original_error = np.linalg.norm(model.jacobian-truth)
        for i in range(250):
            effort = np.array([.08,0]) if i%2 else np.array([0,-.08])
            model.update(effort, truth @ effort)
        self.assertLess(np.linalg.norm(model.jacobian-truth),original_error*.1)
        snapshot = model.jacobian.copy()
        self.assertFalse(model.update([.1,0],[1.,1.]))
        np.testing.assert_equal(snapshot,model.jacobian)

    def test_integrates_applied_pwm_through_ramps_and_stops(self):
        history = adaptive.EffortHistory()
        history.record(1,[.2,-.3]);history.record(2,[.6,-.7]);history.record(3,[0,0])
        np.testing.assert_allclose(history.integral(1.5,4),[.7,-.85])
        np.testing.assert_equal(history.integral(4,5),[0,0])

    def test_profile_roundtrip_and_rejects_old_schema_nan_and_pin_collisions(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(adaptive,'PROFILE_DIR',Path(directory)):
            profile = adaptive.RobotProfile(name='robot-carpet',surface='carpet',models={'tag':body()},
                                            signatures={'tag':'apriltag36h11:0'})
            profile.save()
            restored = adaptive.RobotProfile.load(profile.path)
            np.testing.assert_equal(restored.models['tag'].jacobian,profile.models['tag'].jacobian)
            self.assertEqual(restored.surface,'carpet')
            data = json.loads(profile.path.read_text())
            data['models']['tag']['jacobian'][0][0] = float('nan')
            profile.path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):adaptive.RobotProfile.load(profile.path)
            profile.path.write_text('{"schema":0}')
            with self.assertRaises(ValueError):adaptive.RobotProfile.load(profile.path)
        profile = adaptive.RobotProfile(left=(7,4,5),right=(7,8,6))
        with self.assertRaises(ValueError):profile.validate()
        with self.assertRaises(ValueError):adaptive.RobotProfile(name='../escape').validate()


class ServoTests(unittest.TestCase):
    def test_prediction_bridges_short_gap_then_stops_without_refreshing_memory(self):
        servo = adaptive.VisualServo(body(),.3,165,delay=0)
        servo.observe(('ball:1',.1,.1),10.,10.)
        self.assertEqual(servo.command(10.01)[1],'reacquire')
        servo.observe(('ball:1',.1,.1),10.1,10.1)
        demand,status = servo.command(10.12)
        self.assertEqual(status,'track');self.assertGreater(np.linalg.norm(demand),0)
        servo.observe(None,10.2,10.2)
        demand,status = servo.command(10.3)
        self.assertEqual(status,'predict');self.assertGreater(np.linalg.norm(demand),0)
        np.testing.assert_equal(servo.command(10.6)[0],[0,0])
        self.assertEqual(servo.stamp,10.1)
        self.assertFalse(servo.done)

    def test_arrival_needs_three_real_distinct_same_identity_sightings(self):
        servo = adaptive.VisualServo(body(),.3,165)
        servo.observe(('ball:1',0,.35),10,10)
        servo.observe(('ball:1',0,.35),10,10.1)
        self.assertFalse(servo.done)
        servo.observe(('ball:2',0,.35),10.1,10.1)
        self.assertFalse(servo.done)
        servo.observe(('ball:2',0,.35),10.2,10.2)
        self.assertFalse(servo.done)
        servo.observe(('ball:2',0,.35),10.3,10.3)
        self.assertTrue(servo.done)
        np.testing.assert_equal(servo.command(10.4)[0],[0,0])

    def test_stale_outlier_and_long_gap_cannot_authorize_forward_or_arrival(self):
        servo = adaptive.VisualServo(body(),.3,165)
        self.assertFalse(servo.observe(('tag:0',0,.4),10,11))
        servo.observe(('tag:0',-.3,.1),11,11)
        servo.observe(('tag:0',-.3,.1),11.1,11.1)
        self.assertFalse(servo.observe(('tag:0',.8,.6),11.2,11.2))
        self.assertFalse(servo.done)
        servo.observe(('tag:0',-.3,.1),12,12)
        np.testing.assert_equal(servo.command(12.01)[0],[0,0])

    def test_search_uses_learned_turn_and_remembered_left_side(self):
        servo = adaptive.VisualServo(body(),.3,165)
        servo.observe(('ball:1',-.4,.1),10,10)
        demand,status = servo.command(12.1)
        self.assertEqual(status,'search')
        predicted = servo.model.jacobian @ demand
        self.assertGreater(predicted[0],0)  # left turn shifts target right
        self.assertAlmostEqual(predicted[1],0,delta=.003)

    def test_simulated_closed_loop_converges_with_reversed_right_motor(self):
        servo = adaptive.VisualServo(body(),.3,165,delay=0)
        feature = np.array([.45,np.log(.08)])
        dt = .05
        for step in range(800):
            now = 10+step*dt
            servo.observe(('ball:1',float(feature[0]),float(np.exp(feature[1]))),now,now)
            demand,_ = servo.command(now)
            servo.history.record(now,demand)
            feature += servo.model.jacobian @ demand*dt
        self.assertLess(abs(feature[0]),.01)
        self.assertGreater(np.exp(feature[1]),.295)
        self.assertTrue(servo.done)


class PerceptionTests(unittest.TestCase):
    def test_real_apriltag_detector_filters_selected_id(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image = np.full((240,320,3),255,np.uint8)
        marker = cv2.aruco.generateImageMarker(dictionary,4,140)
        image[50:190,90:230] = cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
        jpeg = cv2.imencode('.jpg',image)[1].tobytes()
        perception = mission.Perception(None,None,4,'bytetrack',.5,416,'cpu')
        target,_ = perception.detect(jpeg,'tag')
        self.assertEqual(target[0],'tag:4');self.assertGreater(target[2],.4)
        perception.tag_id = 0
        self.assertIsNone(perception.detect(jpeg,'tag')[0])

    def test_bytetrack_lock_does_not_jump_to_larger_same_class_object(self):
        boxes = MagicMock()
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10,10,30,30],[60,10,80,30]])
        boxes.id.cpu.return_value.numpy.return_value = np.array([1,2])
        model = MagicMock();model.track.return_value = [SimpleNamespace(boxes=boxes)]
        perception = mission.Perception(model,0,0,'bytetrack',.5,416,'cpu')
        jpeg = cv2.imencode('.jpg',np.zeros((100,100,3),np.uint8))[1].tobytes()
        self.assertEqual(perception.detect(jpeg,'object')[0][0],'object:1')
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10,10,30,30],[40,0,100,80]])
        self.assertEqual(perception.detect(jpeg,'object')[0][0],'object:1')
        self.assertTrue(model.track.call_args.kwargs['persist'])


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_pwm_floor_preserves_learned_wheel_ratio_when_feasible(self):
        profile=adaptive.RobotProfile(minimum_pwm=(100,100))
        session=mission.Session(SimpleNamespace(),profile,None)
        session.control=SimpleNamespace(set_speed=AsyncMock())
        await session.motors_to([.25,.4])
        values=[call.args[1] for call in session.control.set_speed.await_args_list]
        self.assertAlmostEqual(values[0]/values[1],.25/.4)
        self.assertGreaterEqual(min(values)*255,100)
        self.assertLessEqual(max(values)*255,profile.max_pwm)

    async def test_user_cancellation_closes_session_and_retains_bounded_model_updates(self):
        session=SimpleNamespace(open=AsyncMock(),close=AsyncMock())
        args=SimpleNamespace(video=None,mode='drive')
        with tempfile.TemporaryDirectory() as directory,patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             patch.object(mission,'Session',return_value=session), \
             patch.object(mission,'mission',new=AsyncMock(side_effect=asyncio.CancelledError)):
            profile=adaptive.RobotProfile(models={'object':body(),'tag':body()})
            with self.assertRaises(asyncio.CancelledError):await mission.main(args,profile,None,{})
            self.assertTrue(profile.path.is_file())
        session.close.assert_awaited_once()

    async def test_continuous_drive_processes_moving_frames_switches_stages_and_stops(self):
        pwm = {};writes = []
        board = SimpleNamespace(start_aio=AsyncMock(),disable_all_reporting=AsyncMock(),
            set_pin_mode_digital_output=AsyncMock(),set_pin_mode_analog_output=AsyncMock(),
            digital_write=AsyncMock(),shutdown=AsyncMock(),sock=None)
        async def analog(pin,value):
            pwm[pin] = value;writes.append((time.monotonic(),pin,value))
        board.analog_write = analog
        jpeg = cv2.imencode('.jpg',np.zeros((60,80,3),np.uint8))[1].tobytes()
        class Camera:
            phone_urls = ['simulation'];is_streaming = True
            def __init__(self):self.stop=MagicMock();self.at=0;self.packet=None
            @property
            def jpeg(self):
                now=time.monotonic()
                if now-self.at>.025:
                    self.packet=bytes(bytearray(jpeg));self.at=now
                return self.packet
        counts = {'object':0,'tag':0};moving_detections = [];calls=[]
        def detect(packet,leg):
            calls.append(leg);counts[leg]+=1
            if any(pwm.values()):moving_detections.append(leg)
            time.sleep(.08)  # inference must not block the 25Hz control loop
            if leg == 'object':
                return ('ball:1',0,.1 if counts[leg]<5 else .4),np.zeros((60,80,3),np.uint8)
            return ('tag:0',0,.3),np.zeros((60,80,3),np.uint8)
        camera = Camera();ticks=[]
        original = mission.Session.motors_to
        async def motors_to(session,*args,**kwargs):
            ticks.append(time.monotonic())
            await original(session,*args,**kwargs)
        # A long detector gap may produce a legitimate large size change; use a smooth test approach.
        def smooth_detect(packet,leg):
            target,frame = detect(packet,leg)
            if leg=='object':target=(target[0],0,min(.35,.1*1.18**counts[leg]))
            return target,frame
        from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio import telemetrix_uno_r4_wifi_aio as library
        with tempfile.TemporaryDirectory() as directory, patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             patch.object(mission.phonesense,'start',return_value=camera), \
             patch.object(library,'TelemetrixUnoR4WiFiAio',return_value=board), \
             patch.object(mission.Session,'motors_to',motors_to),contextlib.redirect_stdout(io.StringIO()):
            args = mission.arguments(['--mode','drive','--headless','--log',str(Path(directory)/'run.jsonl')])
            profile = adaptive.RobotProfile(models={'object':body(),'tag':body()})
            await asyncio.wait_for(mission.main(args,profile,SimpleNamespace(detect=smooth_detect),{}),8)
            self.assertTrue((Path(directory)/'robot-floor.json').is_file())
        self.assertIn('tag',calls);self.assertTrue(moving_detections)
        self.assertGreater(len(ticks),len(calls)*2)
        self.assertFalse(any(pwm.values()))
        camera.stop.assert_called_once();board.shutdown.assert_awaited_once()

    async def test_replay_cannot_drive_even_when_main_called_directly(self):
        args = SimpleNamespace(video=Path('clip.avi'),mode='drive')
        with self.assertRaises(ValueError):await mission.main(args,None,None,None)

    async def test_camera_watchdog_rejects_stale_frames(self):
        profile=adaptive.RobotProfile()
        args=SimpleNamespace(headless=True,max_seconds=0)
        session=mission.Session(args,profile,None)
        session.jpeg=b'old';session.received=time.monotonic()-2
        session.control=SimpleNamespace(stop=AsyncMock())
        with self.assertRaises(ConnectionError):session.pump('test')

    async def test_camera_disconnect_stops_real_motor_controller_and_closes_session(self):
        board = SimpleNamespace(start_aio=AsyncMock(),disable_all_reporting=AsyncMock(),
            set_pin_mode_digital_output=AsyncMock(),set_pin_mode_analog_output=AsyncMock(),
            digital_write=AsyncMock(),shutdown=AsyncMock(),sock=None)
        pwm={};seen_motion=[]
        async def analog(pin,value):
            pwm[pin]=value
            if value:seen_motion.append(value)
        board.analog_write=analog
        jpeg=cv2.imencode('.jpg',np.zeros((40,40,3),np.uint8))[1].tobytes()
        began=time.monotonic()
        class Camera:
            phone_urls=['simulation']
            stop=MagicMock()
            @property
            def is_streaming(self):return time.monotonic()-began<.45
            @property
            def jpeg(self):return bytes(bytearray(jpeg))
        def detect(*args):
            time.sleep(.04)
            return ('ball:1',0,.1),np.zeros((40,40,3),np.uint8)
        from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio import telemetrix_uno_r4_wifi_aio as library
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(mission.phonesense,'start',return_value=Camera()), \
             patch.object(library,'TelemetrixUnoR4WiFiAio',return_value=board), \
             contextlib.redirect_stdout(io.StringIO()):
            args=mission.arguments(['--mode','drive','--headless','--log',str(Path(directory)/'run.jsonl')])
            profile=adaptive.RobotProfile(models={'object':body(),'tag':body()})
            with self.assertRaises(ConnectionError):
                await mission.main(args,profile,SimpleNamespace(detect=detect),{})
        self.assertTrue(seen_motion)
        self.assertFalse(any(pwm.values()))
        board.shutdown.assert_awaited_once()

    async def test_calibration_probes_both_legs_learns_direction_and_saves_profiles(self):
        matrix=body().jacobian
        profile=adaptive.RobotProfile()
        class CalibrationSession:
            args=SimpleNamespace(tag_only=False)
            history=None
            stable_key='ball:1'
            def __init__(self):self.profile=profile;self.stamp=10.;self.feature=np.array([0.,-2.])
            async def prompt(self,text):pass
            async def stable(self,leg):
                self.stable_key=leg+':1';self.stamp+=.3
                return self.feature.copy(),self.stamp
            def applied(self):return np.zeros(2)
            async def wait(self,seconds,status,demand):
                self.history.record(self.stamp,demand)
                # Below PWM 100, the wheel is stalled despite a command being applied.
                if np.max(np.abs(demand))*255>=100:
                    self.feature+=matrix @ demand*seconds
                self.stamp+=seconds
            async def stop(self):self.history.record(self.stamp,[0,0])
            def log(self,*args,**kwargs):pass
        with tempfile.TemporaryDirectory() as directory,patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             contextlib.redirect_stdout(io.StringIO()):
            await mission.calibrate(CalibrationSession(),{'object':'weights:0','tag':'apriltag36h11:0'})
            saved=adaptive.RobotProfile.load(profile.path)
        self.assertEqual(saved.minimum_pwm,[110,110])
        for leg in ('object','tag'):
            np.testing.assert_allclose(saved.models[leg].jacobian,matrix)


if __name__=='__main__':unittest.main()
