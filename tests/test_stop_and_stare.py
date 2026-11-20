"""Standalone Red-Ball -> AprilTag behavior; all motor hardware is simulated."""

import asyncio
import contextlib
import importlib.util
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,MagicMock,patch

import cv2
import numpy as np
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio import telemetrix_uno_r4_wifi_aio as library

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('standalone_stop',ROOT/'stop-and-stare.py')
mission=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mission)


class EvidenceTests(unittest.TestCase):
    def test_confirmation_rejects_flicker_wrong_id_and_jumps(self):
        ball=mission.Sighting('Red-Ball',-.3,.05)
        self.assertIsNone(mission.confirm([ball,None,None]))
        self.assertEqual(mission.confirm([ball,ball,None]),ball)
        self.assertIsNone(mission.confirm([ball,mission.Sighting('tag:0',-.3,.05),None]))
        self.assertIsNone(mission.confirm([ball,mission.Sighting('Red-Ball',.7,.05),None]))
        self.assertIsNone(mission.confirm([ball,mission.Sighting('Red-Ball',-.3,float('nan')),None]))

    def test_memory_retains_stationary_position_and_expires_by_leg(self):
        memory=mission.TargetMemory()
        target=mission.Sighting('Red-Ball',-.5,.02)
        memory.remember('object',target,b'stationary-frame',10.)
        self.assertEqual(memory.search_side('object',11.),-1)
        self.assertEqual(memory.sighting,target)
        self.assertEqual(memory.jpeg,b'stationary-frame')
        self.assertEqual(memory.search_side('tag',11.),1)
        self.assertEqual(memory.search_side('object',10+mission.TARGET_MEMORY_SECONDS+1),1)
        memory.clear()
        self.assertIsNone(memory.jpeg)

    def test_real_apriltag_detection_filters_id_and_reports_relative_size(self):
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag=cv2.aruco.generateImageMarker(dictionary,0,160)
        image=np.full((240,320,3),255,np.uint8)
        image[40:200,80:240]=cv2.cvtColor(tag,cv2.COLOR_GRAY2BGR)
        jpeg=cv2.imencode('.jpg',image)[1].tobytes()
        detector=cv2.aruco.ArucoDetector(dictionary,cv2.aruco.DetectorParameters())
        sighting,_=mission.detect(jpeg,'tag',None,None,None,detector,0)
        self.assertEqual(sighting.identity,'tag:0')
        self.assertAlmostEqual(sighting.x,0,delta=.02)
        self.assertGreater(sighting.size,mission.TAG_STOP_WIDTH)
        self.assertIsNone(mission.detect(jpeg,'tag',None,None,None,detector,1)[0])


class MissionTests(unittest.IsolatedAsyncioTestCase):
    async def stationary_case(self, delay, timeout=8., freeze=False, expected_error=None):
        """Live camera and real PWM ramps, with a slow CPU detector and mocked board."""
        pwm={};moves=[];detected=[]
        board=SimpleNamespace(start_aio=AsyncMock(),disable_all_reporting=AsyncMock(),
            set_pin_mode_digital_output=AsyncMock(),set_pin_mode_analog_output=AsyncMock(),
            digital_write=AsyncMock(),shutdown=AsyncMock(),sock=None)
        async def write(pin,value):
            pwm[pin]=value
            if value:moves.append(value)
        board.analog_write=write
        class Camera:
            phone_urls=['simulation'];is_streaming=True
            def __init__(self):
                self.sequence=0;self.stop=MagicMock();self.detector_started=None;self.packet=None
            @property
            def jpeg(self):
                frozen=(freeze and self.detector_started is not None
                        and time.monotonic()-self.detector_started>.02)
                if not frozen:
                    self.sequence+=1
                    self.packet=f'frame-{self.sequence}'.encode()
                return self.packet
        camera=Camera()
        def detect(jpeg,leg,*args):
            self.assertFalse(any(pwm.values()),'CPU inference must keep motors stopped')
            if camera.detector_started is None:camera.detector_started=time.monotonic()
            detected.append(jpeg)
            time.sleep(delay)
            self.assertFalse(any(pwm.values()),'motor output changed while CPU detector ran')
            return mission.Sighting('Red-Ball',0,.02),np.zeros((20,20,3),np.uint8)
        model=SimpleNamespace(task='detect',names={0:'Red-Ball'},predict=MagicMock())
        with tempfile.TemporaryDirectory() as folder:
            weights=Path(folder)/'ball.pt';weights.write_bytes(b'fake')
            args=SimpleNamespace(tag_only=False,color=False,drive=True,video=None,headless=True,
                weights=weights,target_class='Red-Ball',tag_id=0,arduino_ip='127.0.0.1',
                phone_ip=None,max_steps=1,inference_timeout=timeout,image_size=320,confidence=.6,device='cpu')
            with patch.object(mission.phonesense,'start',return_value=camera), \
                 patch.object(library,'TelemetrixUnoR4WiFiAio',return_value=board), \
                 patch.dict(sys.modules,{'ultralytics':SimpleNamespace(YOLO=MagicMock(return_value=model))}), \
                 patch.object(mission,'detect',side_effect=detect), \
                 patch.multiple(mission,SETTLE_SECONDS=.005,DISCARD_FRAMES=1,FORWARD_HOLD=.005,
                    CAMERA_TIMEOUT=.10 if freeze else 1.5,MIN_PWM=0,ACCELERATION=10000,DECELERATION=10000), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                if expected_error:
                    with self.assertRaises(expected_error):await mission.main(args)
                else:
                    await asyncio.wait_for(mission.main(args),7)
        self.assertFalse(any(pwm.values()),'shutdown must clear PWM')
        board.shutdown.assert_awaited_once();camera.stop.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs['imgsz'],320)
        self.assertEqual(model.predict.call_args.kwargs['conf'],.6)
        return moves,detected,output.getvalue()

    async def test_slow_stationary_inference_over_one_second_can_finish_and_move(self):
        moves,detected,output=await self.stationary_case(1.05)
        self.assertEqual(len(detected),mission.OBSERVE_FRAMES)
        self.assertEqual(len(set(detected)),len(detected))
        self.assertTrue(moves,'valid slow stationary results should still permit one movement')
        self.assertIn('stationary frame accepted',output)

    async def test_stationary_inference_still_has_a_hard_processing_timeout(self):
        moves,_,_=await self.stationary_case(.15,timeout=.04,expected_error=TimeoutError)
        self.assertFalse(moves,'timed-out inference must never authorize movement')

    async def test_frozen_camera_during_inference_stops_without_moving(self):
        moves,_,_=await self.stationary_case(.25,freeze=True,expected_error=ConnectionError)
        self.assertFalse(moves,'camera watchdog must remain active during slow inference')

    async def test_live_sequence_uses_no_moving_frames_for_detection_or_memory(self):
        board=SimpleNamespace(start_aio=AsyncMock(),disable_all_reporting=AsyncMock(),
            set_pin_mode_digital_output=AsyncMock(),set_pin_mode_analog_output=AsyncMock(),
            digital_write=AsyncMock(),shutdown=AsyncMock(),sock=None)
        pwm={}
        stopped_at=-float('inf')
        async def write(pin,value):
            nonlocal stopped_at
            pwm[pin]=value
            if not any(pwm.values()):
                stopped_at=time.monotonic()
        board.analog_write=write
        moving_frames=set()
        timestamps={}
        class Camera:
            phone_urls=['https://127.0.0.1:8080/phone'];is_streaming=True
            def __init__(self):self.sequence=0;self.stop=MagicMock()
            @property
            def jpeg(self):
                self.sequence+=1
                token=f'frame-{self.sequence}'.encode()
                timestamps[token]=time.monotonic()
                if any(pwm.values()):moving_frames.add(token)
                return token
        camera=Camera()
        ball=mission.Sighting
        # Turn, short dropout, remembered-side search, approach, arrive, turn, arrive.
        samples=iter([ball('Red-Ball',.6,.02)]*2+[None]*4+
                     [ball('Red-Ball',0,.02)]*2+[ball('Red-Ball',0,.2)]*2+
                     [ball('tag:0',-.5,.1)]*2+[ball('tag:0',0,.3)]*2)
        detected=[];remembered=[]
        def detect(jpeg,leg,*args):
            self.assertFalse(any(pwm.values()),'detection during nonzero motor output')
            self.assertNotIn(jpeg,moving_frames)
            self.assertGreaterEqual(timestamps[jpeg]+.001,stopped_at+mission.SETTLE_SECONDS,
                                    'accepted frame arrived before the post-stop settle interval')
            detected.append((leg,jpeg))
            return next(samples),np.zeros((20,20,3),np.uint8)
        original=mission.TargetMemory.remember
        def remember(memory,leg,target,jpeg,now):
            self.assertFalse(any(pwm.values()),'memory update during movement')
            self.assertNotIn(jpeg,moving_frames)
            remembered.append((leg,target,jpeg))
            original(memory,leg,target,jpeg,now)
        model=SimpleNamespace(task='detect',names={0:'Red-Ball'},predict=MagicMock())
        with tempfile.TemporaryDirectory() as folder:
            weights=Path(folder)/'ball.pt';weights.write_bytes(b'fake')
            args=SimpleNamespace(tag_only=False,color=False,drive=True,video=None,headless=True,
                weights=weights,target_class='Red-Ball',tag_id=0,arduino_ip='127.0.0.1',phone_ip=None,max_steps=0)
            output=io.StringIO()
            with patch.object(mission.phonesense,'start',return_value=camera), \
                 patch.object(library,'TelemetrixUnoR4WiFiAio',return_value=board), \
                 patch.dict(sys.modules,{'ultralytics':SimpleNamespace(YOLO=MagicMock(return_value=model))}), \
                 patch.object(mission,'detect',side_effect=detect), \
                 patch.object(mission.TargetMemory,'remember',remember), \
                 patch.multiple(mission,SETTLE_SECONDS=.01,OBSERVE_FRAMES=2,CONFIRMATIONS=2,
                    DISCARD_FRAMES=2,FORWARD_HOLD=.01,TURN_HOLD=.01,SEARCH_HOLD=.01,
                    MIN_PWM=0,ACCELERATION=10000,DECELERATION=10000), \
                 contextlib.redirect_stdout(output):
                await asyncio.wait_for(mission.main(args),8)
        self.assertEqual(len(detected),14)
        self.assertTrue(moving_frames,'simulation did not exercise a real nonzero PWM ramp')
        self.assertTrue(all(jpeg not in moving_frames for _,jpeg in detected))
        self.assertEqual([leg for leg,_,_ in remembered],['object','object','object','tag','tag'])
        self.assertIn('search right',output.getvalue())
        self.assertLess(output.getvalue().index('object reached'),output.getvalue().index('tag reached'))
        self.assertFalse(any(pwm.values()))
        board.shutdown.assert_awaited_once()
        camera.stop.assert_called_once()

    async def test_replay_cannot_drive_even_when_calling_main_directly(self):
        with patch.object(library,'TelemetrixUnoR4WiFiAio') as board:
            with self.assertRaisesRegex(ValueError,'replay'):
                await mission.main(SimpleNamespace(video=Path('clip.avi'),drive=True))
            board.assert_not_called()

    async def test_real_tag_replay_finishes_without_phone_or_arduino(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tag.avi'
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            tag=cv2.aruco.generateImageMarker(dictionary,0,160)
            image=np.full((240,320,3),255,np.uint8)
            image[40:200,80:240]=cv2.cvtColor(tag,cv2.COLOR_GRAY2BGR)
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240))
            self.assertTrue(writer.isOpened())
            for _ in range(45):writer.write(image)
            writer.release()
            args=SimpleNamespace(tag_only=True,color=False,drive=False,video=path,headless=True,
                tag_id=0,phone_ip=None,max_steps=2)
            with patch.object(mission.phonesense,'start',side_effect=AssertionError('PhoneSense opened in replay')), \
                 patch.object(library,'TelemetrixUnoR4WiFiAio',side_effect=AssertionError('Motor connection')), \
                 patch.object(mission,'SETTLE_SECONDS',.005),contextlib.redirect_stdout(io.StringIO()) as output:
                await mission.main(args)
            self.assertIn('tag reached',output.getvalue())


if __name__=='__main__':unittest.main()
