"""Explicit user-started body identification using a fixed visible AprilTag."""

import time
import asyncio

from .controller import Step
from robotics_lab.body_model import VisualBodyModel, body_path, feature


async def calibrate(runner):
    cfg = runner.cfg
    if not runner.driver.drive or getattr(runner.source,'video',None):
        raise ValueError('Body learning requires explicit driving with the camera attached to the robot')
    output = body_path(cfg)
    if output.exists():
        raise ValueError('Body profile already exists. Choose a new body_model filename before calibrating')
    runner.planner.leg = 'tag'
    samples = []
    try:
        print('BODY CALIBRATION: keep the robot camera attached and the AprilTag fixed. '
              'Eight short single-motor probes; target loss aborts instead of searching.',flush=True)
        while not runner.source.healthy(time.monotonic()):
            runner.poll('WAIT_CAMERA')
            if runner.source.eof:
                raise ValueError('Calibration camera ended before starting')
            await asyncio.sleep(.02)
        baseline = await runner.observe()
        if baseline is None or abs(baseline.x)>.4 or not .08 <= baseline.size <= .5:
            raise ValueError('Before calibrating, center a stationary AprilTag with width 8–50% of the image')
        for repetition in range(2):
            for axis in (0,1):
                for sign in (1,-1):
                    demand = [0.,0.]
                    demand[axis] = sign*cfg.body_probe_speed
                    step = Step(f'calibrate_{axis}_{sign:+}',*demand,cfg.body_probe_seconds)
                    runner.event('decision',step=step.name,left=step.left,right=step.right,hold_seconds=step.hold_seconds)
                    print(f'Body probe {len(samples)+1}/8: motor {axis+1}, direction {sign:+}',flush=True)
                    started = time.monotonic()
                    await runner.driver.pulse(step,runner.motion_tick)
                    runner.event('motors_stopped')
                    after = await runner.observe()
                    if after is None or after.identity != baseline.identity:
                        raise ValueError('Calibration target lost; motors stopped. No automatic search and no model saved')
                    delta = feature(after)-feature(baseline)
                    if abs(delta[0])>.45 or abs(delta[1])>.4 or not -.85<after.x<.85:
                        raise ValueError('Calibration movement was too large; reduce probe/ramp motion before retrying')
                    impulse = getattr(runner.driver,'last_impulse',None)
                    if impulse is None:
                        raise ValueError('Motion driver does not report motor command integrals')
                    samples.append(dict(impulse=list(impulse),delta=delta.tolist(),
                                        elapsed=time.monotonic()-started,axis=axis,sign=sign))
                    baseline = after
                    if runner.telemetry:
                        runner.telemetry.update(calibration_progress=len(samples),calibration_total=8)
        model = VisualBodyModel.fit(samples,cfg)
        model.save_new(output)
        print(f'Body model saved: {output}. Select body_control=learned with this profile.',flush=True)
        if runner.telemetry:
            runner.telemetry.update(body_model=str(output),calibration_progress=8,calibration_total=8)
        return f'Body response learned and saved to {output}'
    finally:
        await runner.driver.stop()
        runner.file.close()
