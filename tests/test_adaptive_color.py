"""Click-selected OpenCV adaptive homing; no phone, YOLO or physical motors."""

import asyncio
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np

from tests.test_adaptive_homing import mission, body
from robotics_lab import adaptive_control as adaptive


class ColorPerceptionTests(unittest.TestCase):
    def test_hue_wraparound_rejects_larger_blue_distractor(self):
        image=np.zeros((120,200,3),np.uint8)
        image[:]=(50,0,50)
        cv2.circle(image,(40,60),18,(179,255,255),-1)  # HSV red near the wrap boundary
        cv2.rectangle(image,(110,10),(190,110),(110,255,255),-1)
        jpeg=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_HSV2BGR))[1].tobytes()
        perception=mission.Perception(None,None,0,'bytetrack',.5,416,'cpu',color=True,hsv=(0,255,255))
        target,_=perception.detect(jpeg,'object')
        self.assertTrue(target[0].startswith('hsv:'))
        self.assertAlmostEqual(target[1],-.6,delta=.04)
        self.assertGreater(target[2],.2)
        self.assertLess(target[2],.3)

    def test_color_mode_without_selection_returns_no_target(self):
        jpeg=cv2.imencode('.jpg',np.zeros((120,160,3),np.uint8))[1].tobytes()
        perception=mission.Perception(None,None,0,'bytetrack',.5,416,'cpu',color=True)
        self.assertIsNone(perception.detect(jpeg,'object')[0])

    def test_color_mode_still_detects_only_the_selected_apriltag(self):
        image=np.full((240,320,3),255,np.uint8)
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),3,140)
        image[50:190,90:230]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
        jpeg=cv2.imencode('.jpg',image)[1].tobytes()
        perception=mission.Perception(None,None,3,'bytetrack',.5,416,'cpu',color=True,hsv=(0,255,255))
        self.assertEqual(perception.detect(jpeg,'tag')[0][0],'tag:3')
        perception.tag_id=0
        self.assertIsNone(perception.detect(jpeg,'tag')[0])


class ColorSetupTests(unittest.TestCase):
    def test_color_preview_never_imports_yolo_or_requires_weights(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             patch.object(mission,'PROFILE_DIR',Path(directory)),patch.object(sys,'argv',['adaptive-homing.py','--color']), \
             patch.dict(sys.modules,{'ultralytics':None}):
            args=mission.arguments(['--color','--mode','preview','--headless','--hsv','0','255','255',
                                    '--weights','does-not-exist.pt'])
            _,_,perception,signatures=mission.setup(args)
            self.assertIsNone(perception.model)
            self.assertEqual(perception.color_hsv,(0,255,255))
            self.assertTrue(signatures['object'].startswith('hsv:'))

    def test_headless_color_without_hsv_or_saved_color_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             patch.object(mission,'PROFILE_DIR',Path(directory)),patch.object(sys,'argv',['adaptive-homing.py','--color']):
            with self.assertRaisesRegex(ValueError,'Headless color'):
                mission.setup(mission.arguments(['--color','--headless']))

    def test_drive_restores_calibrated_hsv_and_rejects_changed_color_or_yolo_profile(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             patch.object(mission,'PROFILE_DIR',Path(directory)),patch.object(sys,'argv',['adaptive-homing.py','--color']), \
             patch.dict(sys.modules,{'ultralytics':None}):
            args=mission.arguments(['--color','--mode','drive'])
            profile=adaptive.RobotProfile(color_hsv=(0,255,255),models={'object':body(),'tag':body()},
                signatures={'object':mission.color_signature(args,(0,255,255)),'tag':'apriltag36h11:0'})
            profile.save()
            _,restored,perception,_=mission.setup(args)
            self.assertEqual(tuple(restored.color_hsv),(0,255,255))
            self.assertEqual(perception.color_hsv,(0,255,255))
            changed=mission.arguments(['--color','--mode','drive','--hsv','60','255','255'])
            with self.assertRaisesRegex(ValueError,'Calibrate object'):mission.setup(changed)
            profile.signatures['object']='yolo-weights:0';profile.save()
            with self.assertRaisesRegex(ValueError,'Calibrate object'):mission.setup(args)


class ColorSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_click_selects_original_hsv_and_waits_with_motors_stopped(self):
        args=mission.arguments(['--color'])
        perception=mission.Perception(None,None,0,'bytetrack',.5,416,'cpu',color=True)
        session=mission.Session(args,adaptive.RobotProfile(),perception)
        session.stop=AsyncMock()
        session.display_hsv=np.full((10,10,3),(178,240,220),np.uint8)
        def pump(status):
            self.assertIsNone(perception.color_hsv)
            session.pick_color(cv2.EVENT_LBUTTONDOWN,-1,3,None,None)
            self.assertIsNone(perception.color_hsv)
            session.pick_color(cv2.EVENT_LBUTTONDOWN,3,3,None,None)
        session.pump=MagicMock(side_effect=pump)
        signatures={}
        with contextlib.redirect_stdout(io.StringIO()):await session.prepare_color(signatures)
        session.stop.assert_awaited_once()
        self.assertEqual(perception.color_hsv,(178,240,220))
        session.display_hsv[:]=(60,255,255)
        session.pick_color(cv2.EVENT_LBUTTONDOWN,3,3,None,None)
        self.assertEqual(perception.color_hsv,(178,240,220),'mid-run clicks cannot silently change calibrated target')
        self.assertTrue(signatures['object'].startswith('hsv:'))

    async def test_real_color_then_apriltag_replay_never_loads_yolo_or_connects_robot(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(adaptive,'PROFILE_DIR',Path(directory)), \
             patch.object(mission,'PROFILE_DIR',Path(directory)),patch.object(sys,'argv',['adaptive-homing.py','--color']), \
             patch.dict(sys.modules,{'ultralytics':None}), \
             patch.object(mission.phonesense,'start',side_effect=AssertionError('Phone opened in replay')):
            path=Path(directory)/'color-to-tag.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240))
            self.assertTrue(writer.isOpened())
            for radius in list(range(15,56))+[55]*15:
                image=np.full((240,320,3),255,np.uint8)
                cv2.circle(image,(160,120),radius,(0,0,255),-1)
                writer.write(image)
            marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),0,140)
            image=np.full((240,320,3),255,np.uint8)
            image[50:190,90:230]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
            for _ in range(45):writer.write(image)
            writer.release()
            args=mission.arguments(['--color','--mode','preview','--hsv','0','255','255','--headless',
                '--video',str(path),'--log',str(Path(directory)/'run.jsonl')])
            from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio import telemetrix_uno_r4_wifi_aio as library
            with patch.object(library,'TelemetrixUnoR4WiFiAio',side_effect=AssertionError('Arduino connection')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                await asyncio.wait_for(mission.main(*mission.setup(args)),6)
            self.assertIn('object reached',output.getvalue())
            self.assertIn('tag reached',output.getvalue())


if __name__=='__main__':unittest.main()
