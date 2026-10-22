"""Camera-based response identification and control without a physical Arduino."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np

from missions.stop_and_stare.calibration import calibrate
from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Sighting, Step
from missions.stop_and_stare.run import Runner
from missions.stop_and_stare.motion import PulseDriver
from robotics_lab.motor_control import SmoothMotorController
from robotics_lab.body_model import VisualBodyModel, expected_impulse, fingerprint


def observations(matrix, cfg):
    samples=[]
    for _ in range(2):
        for axis in (0,1):
            for sign in (1,-1):
                command=[0.,0.];command[axis]=sign*cfg.body_probe_speed
                impulse=expected_impulse(*command,cfg.body_probe_seconds,cfg)
                samples.append(dict(impulse=impulse.tolist(),delta=(matrix@impulse).tolist()))
    return samples


class BodyTests(unittest.TestCase):
    def test_identifies_reversed_left_motor_and_unequal_wheel_gain(self):
        cfg=Config()
        # Positive left PWM is wired backwards; right has 1.5x visual response.
        actual=np.array([[1.,1.5],[-.4,.6]])
        model=VisualBodyModel.fit(observations(actual,cfg),cfg)
        np.testing.assert_allclose(model.matrix,actual,atol=1e-9)
        target=Sighting('tag','tag:0',0,.1)
        step=model.choose(target,Step('forward',.18,.18,.25),cfg)
        self.assertLess(step.left,0)
        self.assertGreater(step.right,0)
        response=actual@expected_impulse(step.left,step.right,step.hold_seconds,cfg)
        self.assertGreater(response[1],0)
        self.assertLessEqual(abs(step.left),cfg.body_probe_speed)
        self.assertLessEqual(step.hold_seconds,.25)
        turn=model.choose(Sighting('tag','tag:0',.4,.1),Step('turn_right',.18,-.18,.2),cfg)
        self.assertLess((actual@expected_impulse(turn.left,turn.right,turn.hold_seconds,cfg))[0],0)

    def test_rejects_stalled_unobservable_and_inconsistent_responses(self):
        cfg=Config();actual=np.array([[-1.,1.],[.4,.4]])
        samples=observations(actual,cfg)
        for sample in samples:sample['delta']=[0.,0.]
        with self.assertRaises(ValueError):VisualBodyModel.fit(samples,cfg)
        with self.assertRaises(ValueError):VisualBodyModel.fit(observations(np.ones((2,2)),cfg),cfg)
        samples=observations(actual,cfg)
        samples[0]['delta']=[.7,-.7]
        with self.assertRaises(ValueError):VisualBodyModel.fit(samples,cfg)

    def test_profile_preserves_hardware_mapping_and_never_overwrites(self):
        cfg=Config();model=VisualBodyModel.fit(observations(np.array([[-1.,1.],[.4,.4]]),cfg),cfg)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'robot.json'
            model.save_new(path)
            loaded=VisualBodyModel.load(path,cfg)
            np.testing.assert_allclose(loaded.matrix,model.matrix)
            with self.assertRaises(FileExistsError):model.save_new(path)
            with self.assertRaises(ValueError):VisualBodyModel.load(path,replace(cfg,min_pwm=100))
            data=json.loads(path.read_text());data['matrix']=[[float('nan'),1],[1,1]]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):VisualBodyModel.load(path,cfg)

    def test_online_update_is_small_and_rejects_target_jump(self):
        matrix=np.array([[-1.,1.],[.4,.4]])
        model=VisualBodyModel(matrix,metadata=fingerprint(Config()),max_demand=.12)
        u=np.array([.15,.18])
        original=model.matrix.copy()
        self.assertTrue(model.adapt(u,1.02*matrix@u))
        self.assertLess(np.linalg.norm(model.matrix-original),.03*np.linalg.norm(original))
        saved=model.matrix.copy()
        self.assertFalse(model.adapt(u,[.9,1.]))
        np.testing.assert_array_equal(saved,model.matrix)

    def test_impulse_accounts_for_ramps_and_zero_motor(self):
        cfg=Config()
        short=expected_impulse(.12,0,.02,cfg)
        longer=expected_impulse(.12,0,.2,cfg)
        self.assertEqual(short[1],0)
        self.assertGreater(longer[0],short[0])
        np.testing.assert_allclose(expected_impulse(-.12,0,.02,cfg),-short)


class CalibrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ramp_only_learned_pulse_is_executed_even_with_zero_hold(self):
        cfg=replace(Config(),mission_mode='tag_only')
        class Source:
            latest=None;sequence=0
            def poll(self,now):pass
        class Body:
            def choose(self,sighting,step,cfg):return Step('body_forward',.1,.1,0.,(0.,0.))
        with tempfile.TemporaryDirectory() as folder:
            driver=PulseDriver(cfg);driver.pulse=AsyncMock()
            runner=Runner(cfg,Source(),object(),driver,Path(folder),headless=True,body=Body())
            runner.observe=AsyncMock(return_value=Sighting('tag','tag:0',0,.1))
            try:
                await runner.run_batch()
                driver.pulse.assert_awaited_once()
                self.assertEqual(driver.pulse.await_args.args[0].hold_seconds,0.)
            finally:runner.file.close()

    async def test_independent_wheel_braking_integrates_actual_commands_and_reaches_zero(self):
        now=0.
        async def sleep(dt):
            nonlocal now
            now+=dt
            await asyncio.sleep(0)
        cfg=Config()
        driver=PulseDriver(cfg,drive=True,clock=lambda:now,sleep=sleep)
        driver.control=SmoothMotorController(AsyncMock(),driver.motors,min_pwm=cfg.min_pwm,
            acceleration=cfg.acceleration,deceleration=cfg.deceleration,clock=lambda:now)
        await driver.control.stop()
        await driver.pulse(Step('body_forward',.12,.12,.2,(.2,0.)),AsyncMock())
        self.assertTrue(driver.control.is_stopped)
        self.assertGreater(driver.last_impulse[0],driver.last_impulse[1])
        expected=expected_impulse(.12,.12,np.array([.2,0.]),cfg)
        np.testing.assert_allclose(driver.last_impulse,expected,atol=.03)

    async def exercise(self, folder, lose_after=None):
        cfg=replace(Config(),mission_mode='tag_only',body_model=str(Path(folder)/'learned.json'))
        state=np.array([0.,np.log(.2)])
        actual=np.array([[1.,1.5],[-.4,.6]])
        class Driver:
            drive=True;control=None;last_impulse=None
            def __init__(self):self.count=0;self.stop=AsyncMock()
            async def pulse(self,step,hook):
                self.count+=1
                self.last_impulse=expected_impulse(step.left,step.right,step.hold_seconds,cfg)
                state[:]+=actual@self.last_impulse
        class Source:
            latest=None;sequence=0;eof=False;video=None
            def healthy(self,now):return True
            def poll(self,now):pass
        driver=Driver()
        runner=Runner(cfg,Source(),object(),driver,Path(folder),headless=True)
        async def observe():
            if lose_after is not None and driver.count>=lose_after:return None
            return Sighting('tag','tag:0',state[0],np.exp(state[1]))
        runner.observe=observe
        return runner,driver,actual

    async def test_eight_probes_fit_simulated_robot_and_save(self):
        with tempfile.TemporaryDirectory() as folder:
            runner,driver,actual=await self.exercise(folder)
            await calibrate(runner)
            self.assertEqual(driver.count,8)
            driver.stop.assert_awaited()
            fitted=VisualBodyModel.load(Path(folder)/'learned.json',runner.cfg)
            np.testing.assert_allclose(fitted.matrix,actual,atol=1e-9)

    async def test_target_loss_aborts_without_search_or_partial_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            runner,driver,_=await self.exercise(folder,lose_after=1)
            with self.assertRaisesRegex(ValueError,'target lost'):await calibrate(runner)
            self.assertEqual(driver.count,1)
            driver.stop.assert_awaited()
            self.assertFalse((Path(folder)/'learned.json').exists())


if __name__=='__main__':unittest.main()
