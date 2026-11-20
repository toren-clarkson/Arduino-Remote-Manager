"""One-file object -> AprilTag mission. Edit settings here; no dashboard or jobs.

python stop-and-stare.py --tag-only              # camera preview, no Arduino
python stop-and-stare.py --color                 # click a color, then approach tag
python stop-and-stare.py --drive --weights models/red-ball-v1.pt --target-class Red-Ball
"""

import argparse
import asyncio
import time
from pathlib import Path
from statistics import median
from typing import NamedTuple

import cv2
import numpy as np
import phonesense
from robotics_lab.network import ARDUINO_IP, TELEMETRIX_PORT, configure_phone_camera

from robotics_lab.motor_control import SmoothMotorController

# Edit these hardware and movement settings directly in this file.
PHONE_IP = None  # optional override; the PC AP address is selected automatically
LEFT_MOTOR = (7, 4, 5)       # IN1, IN2, PWM
RIGHT_MOTOR = (9, 8, 6)
MIN_PWM = 150              # same starting floor as robust-april-tag.py
ACCELERATION = 300
DECELERATION = 400
FORWARD_SPEED = .18
TURN_SPEED = .18
FORWARD_HOLD = .20          # seconds at target PWM; ramps add movement time
TURN_HOLD = .15
SEARCH_HOLD = .20
SETTLE_SECONDS = .6
DISCARD_FRAMES = 2
OBSERVE_FRAMES = 3
CONFIRMATIONS = 2
CAMERA_TIMEOUT = 1.5
FRAME_WAIT_TIMEOUT = 6.
INFERENCE_TIMEOUT = 8.
SLOW_INFERENCE_SECONDS = 1.  # warning only while the robot is stationary
IMAGE_SIZE = 320
CONFIDENCE = .5
DEVICE = 'cpu'
CENTER_TOLERANCE = .12
OBJECT_STOP_AREA = .12      # object bounding box / image area
TAG_STOP_WIDTH = .24        # mean tag edge / image width
TARGET_MEMORY_SECONDS = 8.
WEIGHTS = Path(__file__).resolve().parent/'models/red-ball-v1.pt'
TARGET_CLASS = 'Red-Ball'
TAG_ID = 0


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int


class Sighting(NamedTuple):
    identity: str
    x: float
    size: float


class TargetMemory:
    """Only confirmed stationary evidence enters this memory; motion never updates it."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.leg = None
        self.sighting = self.jpeg = None
        self.observed_at = -float('inf')
        self.side = 1

    def remember(self,leg,sighting,jpeg,now):
        self.leg,self.sighting,self.jpeg,self.observed_at = leg,sighting,jpeg,now
        if abs(sighting.x)>CENTER_TOLERANCE:
            self.side = 1 if sighting.x>0 else -1

    def search_side(self,leg,now):
        return self.side if self.leg==leg and now-self.observed_at<TARGET_MEMORY_SECONDS else 1


def confirm(samples):
    groups = {}
    for item in samples:
        if item is not None and np.isfinite([item.x,item.size]).all() and -1 <= item.x <= 1 and item.size > 0:
            groups.setdefault(item.identity,[]).append(item)
    group = max(groups.values(),key=len,default=[])
    if len(group) < CONFIRMATIONS:
        return None
    if max(s.x for s in group)-min(s.x for s in group) > .25 or max(s.size for s in group)/min(s.size for s in group)>2:
        return None
    return Sighting(group[0].identity,median(s.x for s in group),min(s.size for s in group))


def detect(jpeg, leg, model, class_id, selected_color, detector, tag_id,
           image_size=IMAGE_SIZE, confidence=CONFIDENCE, device=DEVICE):
    frame = cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Camera returned an invalid JPEG')
    height,width = frame.shape[:2]
    candidates = []
    if leg == 'tag':
        corners,ids,_ = detector.detectMarkers(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        for polygon,identity in zip(corners,ids.flatten() if ids is not None else []):
            if int(identity) != tag_id:
                continue
            points = polygon.reshape(4,2)
            center = points.mean(axis=0)
            size = float(np.linalg.norm(points-np.roll(points,1,axis=0),axis=1).mean())/width
            candidates.append((Sighting(f'tag:{tag_id}',2*float(center[0])/width-1,size),points))
    elif selected_color is not None:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v = selected_color
        delta = np.abs(hsv[:,:,0].astype(np.int16)-h)
        hue = (np.minimum(delta,180-delta)<=10).astype(np.uint8)*255
        mask = cv2.inRange(hsv,(0,max(0,s-80),max(0,v-80)),(179,min(255,s+80),min(255,v+80)))
        mask = cv2.morphologyEx(cv2.bitwise_and(mask,hue),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            candidates.append((Sighting('color',(2*x+w)/width-1,w*h/(width*height)),
                               np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])))
    else:
        result = model.predict(frame,classes=[class_id],conf=confidence,imgsz=image_size,device=device,verbose=False)[0]
        if result.boxes is not None:
            for x1,y1,x2,y2 in result.boxes.xyxy.cpu().numpy():
                if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:
                    continue
                candidates.append((Sighting(str(result.names[class_id]),float((x1+x2)/width-1),
                                             float((x2-x1)*(y2-y1)/(width*height))),
                                   np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])))
    if not candidates:
        return None,frame
    sighting,points = max(candidates,key=lambda item:item[0].size)
    cv2.polylines(frame,[points.astype(np.int32)],True,(0,255,0),2)
    cv2.putText(frame,f'{sighting.identity}: x={sighting.x:+.2f}, size={sighting.size:.1%}',
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
    return sighting,frame


async def main(args):
    if args.video and args.drive:
        raise ValueError('Video replay cannot connect physical motors')
    model = class_id = None
    image_size = getattr(args,'image_size',IMAGE_SIZE)
    confidence = getattr(args,'confidence',CONFIDENCE)
    device = getattr(args,'device',DEVICE)
    inference_timeout = getattr(args,'inference_timeout',INFERENCE_TIMEOUT)
    if not args.tag_only and not args.color:
        print(f'Loading {args.weights}; camera URLs will appear after model warmup...',flush=True)
        from ultralytics import YOLO
        if not args.weights.is_file():
            raise FileNotFoundError(f'Model not found: {args.weights}')
        model = YOLO(str(args.weights))
        if model.task != 'detect':
            raise ValueError('Choose YOLO object-detection weights')
        print('Model classes:',model.names,flush=True)
        class_id = next((i for i,name in model.names.items() if name.casefold()==args.target_class.casefold()),None)
        if class_id is None:
            raise ValueError(f'{args.target_class!r} is not a model class')
        # Use the same class filter, device and image size as actual camera inference.
        await asyncio.to_thread(model.predict,np.zeros((image_size,image_size,3),np.uint8),
                              classes=[class_id],conf=confidence,imgsz=image_size,device=device,verbose=False)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                      cv2.aruco.DetectorParameters())
    cam = capture = board = control = None
    connected = False
    motors = (MotorPins(*LEFT_MOTOR),MotorPins(*RIGHT_MOTOR))
    latest = last_jpeg = None
    received = -float('inf')
    sequence = 0
    next_read = last_display = 0.
    selected_color = None
    display_hsv = None
    leg = 'tag' if args.tag_only else 'object'
    window = 'Robot camera'

    def click(event,x,y,flags,param):
        nonlocal selected_color
        if event==cv2.EVENT_LBUTTONDOWN and display_hsv is not None and 0<=y<display_hsv.shape[0] and 0<=x<display_hsv.shape[1]:
            selected_color = tuple(map(int,display_hsv[y,x]))
            print('Selected HSV:',selected_color,flush=True)

    def pump(phase):
        nonlocal latest,last_jpeg,received,sequence,next_read,last_display,display_hsv
        now = time.monotonic()
        jpeg = None
        if capture is not None and now>=next_read:
            next_read = now+1/fps
            ok,frame = capture.read()
            if not ok:
                raise EOFError('Replay ended')
            ok,encoded = cv2.imencode('.jpg',frame)
            if ok:
                jpeg = encoded.tobytes()
        elif cam is not None:
            packet = cam.jpeg
            if packet is not None and packet is not last_jpeg:
                jpeg = packet
        if jpeg is not None:
            latest = last_jpeg = jpeg
            received,sequence = now,sequence+1
            if not args.headless and now-last_display>=1/15:
                frame = cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                if frame is not None:
                    if args.color:
                        display_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                    cv2.putText(frame,f'{leg.upper()} | {phase} | {"DRIVE" if args.drive else "PREVIEW"}',
                                (10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),2)
                    cv2.imshow(window,frame)
                last_display = now
        if not args.headless and cv2.waitKey(1)&255 in (27,ord('q'),ord('Q')):
            raise KeyboardInterrupt
        if latest is not None and (now-received>CAMERA_TIMEOUT or (cam is not None and not cam.is_streaming)):
            raise ConnectionError('Camera stopped sending fresh frames. Robot stopping; reconnect the phone and rerun.')

    async def wait(seconds,phase):
        end = time.monotonic()+seconds
        while time.monotonic()<end:
            pump(phase)
            await asyncio.sleep(.01)

    async def fresh(after):
        end = time.monotonic()+FRAME_WAIT_TIMEOUT
        while time.monotonic()<end:
            pump('OBSERVE')
            if sequence>after:
                return sequence,latest,received
            await asyncio.sleep(.01)
        raise TimeoutError('No fresh camera frame arrived')

    async def ramp(speeds,phase):
        if control is None:
            demand = speeds if any(speeds) else (FORWARD_SPEED,FORWARD_SPEED)
            pwm = max(MIN_PWM+abs(s)*(255-MIN_PWM) if s else 0 for s in demand)
            await wait(pwm/(ACCELERATION if any(speeds) else DECELERATION),phase)
            return
        end = time.monotonic()+4
        while True:
            pump(phase)
            for motor,speed in zip(motors,speeds):
                await asyncio.wait_for(control.set_speed(motor,speed),.5)
            targets = [(MIN_PWM+abs(s)*(255-MIN_PWM))*(1 if s>0 else -1) if s else 0 for s in speeds]
            if all(abs(control.states[m].pwm-target)<.01 for m,target in zip(motors,targets)):
                return
            if time.monotonic()>end:
                raise TimeoutError('Motor ramp did not finish')
            await asyncio.sleep(.02)

    try:
        if args.video:
            capture = cv2.VideoCapture(str(args.video))
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {args.video}')
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and fps>0 else 20
        else:
            cam = phonesense.start()
            configure_phone_camera(cam, args.phone_ip)
        if not args.headless:
            cv2.namedWindow(window)
            if args.color:
                cv2.setMouseCallback(window,click)
        if args.drive:
            from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio
            board = TelemetrixUnoR4WiFiAio(transport_address=args.arduino_ip,ip_port=TELEMETRIX_PORT,autostart=False,
                loop=asyncio.get_running_loop(),shutdown_on_exception=False,
                close_loop_on_shutdown=False,hard_reset_on_shutdown=False)
            await asyncio.wait_for(board.start_aio(),15)
            connected = True
            await board.disable_all_reporting()
            for m in motors:
                await board.set_pin_mode_digital_output(m.in1)
                await board.set_pin_mode_digital_output(m.in2)
                await board.set_pin_mode_analog_output(m.enable)
            control = SmoothMotorController(board,motors,min_pwm=MIN_PWM,acceleration=ACCELERATION,deceleration=DECELERATION)
            await control.stop()
        print('DRIVE enabled' if args.drive else 'PREVIEW: no Arduino connection',flush=True)
        while latest is None or (args.color and selected_color is None):
            pump('WAIT_CAMERA / CLICK_COLOR')
            await asyncio.sleep(.01)
        memory = TargetMemory()
        empty = steps = 0
        while leg!='done':
            if control is not None and not control.is_stopped:
                raise RuntimeError('Refusing to observe with nonzero commanded motor PWM')
            await wait(SETTLE_SECONDS,'SETTLE')
            gate = sequence
            for _ in range(DISCARD_FRAMES):
                gate,_,_ = await fresh(gate)
            color = selected_color
            samples = []
            evidence = []
            for _ in range(OBSERVE_FRAMES):
                gate,jpeg,stamp = await fresh(gate)
                if control is not None and not control.is_stopped:
                    raise RuntimeError('Refusing detection with nonzero commanded motor PWM')
                started = time.monotonic()
                task = asyncio.create_task(asyncio.to_thread(detect,jpeg,leg,model,class_id,color,detector,args.tag_id,
                                                             image_size,confidence,device))
                warned = False
                try:
                    end = started+inference_timeout
                    while not task.done():
                        if time.monotonic()>end:
                            raise TimeoutError(f'Object detection exceeded {inference_timeout:g}s while stopped. '
                                               'Try a smaller YOLO image size in setup.')
                        if not warned and time.monotonic()-started>SLOW_INFERENCE_SECONDS:
                            print(f'{leg} detection is taking over {SLOW_INFERENCE_SECONDS:g}s; '
                                  'keeping motors stopped while it finishes.',flush=True)
                            warned = True
                        await wait(.01,'OBSERVE')
                    sighting,annotated = task.result()
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task,return_exceptions=True)
                elapsed = time.monotonic()-started
                if elapsed>inference_timeout:
                    raise TimeoutError(f'Object detection exceeded {inference_timeout:g}s while stopped')
                # This image was acquired after settling and the robot remained
                # stopped throughout inference. CPU processing time alone does not
                # invalidate it as it would in a continuously moving controller.
                # Continue enforcing the independent live-camera and hard timeout checks.
                pump('OBSERVE')
                if control is not None and not control.is_stopped:
                    raise RuntimeError('Robot output changed during stationary detection')
                if warned:
                    print(f'{leg} detection finished in {elapsed:.2f}s; stationary frame accepted.',flush=True)
                if not args.headless:
                    cv2.imshow('Last stationary detection',annotated)
                samples.append(sighting)
                evidence.append((sighting,jpeg,stamp,annotated))
            if leg=='object' and color!=selected_color:
                memory.clear()
                continue  # color changes discard the old observation
            target = confirm(samples)
            left = right = hold = 0.
            if target is None:
                empty += 1
                if empty>=2:
                    direction = memory.search_side(leg,time.monotonic())
                    left,right,hold = direction*TURN_SPEED,-direction*TURN_SPEED,SEARCH_HOLD
                    action = 'search left' if direction<0 else 'search right'
                else:
                    action = 'look again'
            else:
                empty = 0
                _,stable_jpeg,stamp,stable_image = next(item for item in reversed(evidence)
                    if item[0] is not None and item[0].identity==target.identity)
                memory.remember(leg,target,stable_jpeg,stamp)
                target = memory.sighting
                if not args.headless:
                    cv2.imshow('Last stationary detection',stable_image)
                if abs(target.x)>CENTER_TOLERANCE:
                    side = 1 if target.x>0 else -1
                    left,right,hold = side*TURN_SPEED,-side*TURN_SPEED,TURN_HOLD
                    action = 'turn right' if side>0 else 'turn left'
                elif target.size >= (TAG_STOP_WIDTH if leg=='tag' else OBJECT_STOP_AREA):
                    action = f'{leg} reached'
                    leg = 'done' if leg=='tag' else 'tag'
                    memory.clear()
                    empty = 0
                else:
                    left = right = FORWARD_SPEED
                    hold,action = FORWARD_HOLD,'forward'
            print(f'{action}: left={left:+.2f}, right={right:+.2f}, hold={hold:.2f}s',flush=True)
            if hold:
                try:
                    await ramp((left,right),'RAMP_UP')
                    await wait(hold,'MOVE')
                    await ramp((0,0),'BRAKE')
                finally:
                    if control:
                        await asyncio.wait_for(control.stop(),1)
            steps += 1
            if args.max_steps and steps>=args.max_steps:
                break
    except EOFError:
        print('Replay ended; no physical motors were connected.',flush=True)
    finally:
        try:
            if control:
                await asyncio.wait_for(control.stop(),1)
        finally:
            if cam:
                cam.stop()
            if capture:
                capture.release()
            cv2.destroyAllWindows()
            if board is not None:
                try:
                    if connected:
                        await asyncio.wait_for(board.shutdown(),1)
                finally:
                    sock = getattr(board,'sock',None)
                    if sock is not None and sock.writer is not None:
                        sock.writer.close()
                        await sock.writer.wait_closed()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive',action='store_true',help='Enable physical motors; default is camera preview')
    parser.add_argument('--tag-only',action='store_true')
    parser.add_argument('--color',action='store_true',help='Click a color instead of using YOLO')
    parser.add_argument('--weights',type=Path,default=WEIGHTS)
    parser.add_argument('--target-class',default=TARGET_CLASS)
    parser.add_argument('--tag-id',type=int,default=TAG_ID)
    parser.add_argument('--image-size',type=int,default=IMAGE_SIZE,help='YOLO inference image size; smaller can reduce CPU time')
    parser.add_argument('--confidence',type=float,default=CONFIDENCE)
    parser.add_argument('--device',default=DEVICE,help='YOLO device, e.g. cpu or 0 for CUDA')
    parser.add_argument('--inference-timeout',type=float,default=INFERENCE_TIMEOUT,
                        help='Maximum processing seconds per stationary image (default: 8)')
    parser.add_argument('--arduino-ip',default=ARDUINO_IP,help='Optional override; defaults to the fixed robot AP address')
    parser.add_argument('--phone-ip',default=PHONE_IP,help='Optional PC interface override; AP address is selected automatically')
    parser.add_argument('--video',type=Path,help='Replay footage; cannot drive')
    parser.add_argument('--headless',action='store_true',help='Replay without native windows')
    parser.add_argument('--max-steps',type=int,default=0)
    args = parser.parse_args()
    if args.video and args.drive:
        parser.error('Replay cannot drive the robot')
    if args.headless and (not args.video or args.color):
        parser.error('Headless mode needs a video and cannot pick a color')
    if args.color and args.tag_only:
        parser.error('Choose color-to-tag or tag-only')
    if not 0<=args.tag_id<587 or args.max_steps<0:
        parser.error('Tag ID must be 0..586 and max steps must be nonnegative')
    if not 160<=args.image_size<=1280 or not .1<=args.confidence<=.95 or not .1<=args.inference_timeout<=60:
        parser.error('Image size must be 160..1280, confidence .1...95, and inference timeout .1..60 seconds')
    return args


if __name__=='__main__':
    from robotics_lab.console import configure_script
    configure_script(globals())
    try:
        asyncio.run(main(arguments()))
    except KeyboardInterrupt:
        print('Stopped.')
    except (ConnectionError,TimeoutError) as exc:
        raise SystemExit(f'Mission stopped: {exc}') from exc
