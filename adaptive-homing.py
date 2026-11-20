"""Continuous adaptive YOLO or clicked HSV color -> AprilTag homing.

python adaptive-homing.py --setup
python adaptive-homing.py --color --setup
python adaptive-homing.py --mode preview --profile robot-floor
python adaptive-homing.py --mode calibrate --profile robot-floor
python adaptive-homing.py --mode drive --profile robot-floor
"""

import argparse
import asyncio
import hashlib
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import phonesense

from robotics_lab.adaptive_control import BodyModel, RobotProfile, VisualServo, PROFILE_DIR
from robotics_lab.console import ask, choose, pins, weights_menu, class_menu
from robotics_lab.motor_control import SmoothMotorController
from robotics_lab.network import ARDUINO_IP, TELEMETRIX_PORT, configure_phone_camera
from robotics_lab.paths import MODELS, RUNS


class MotorPins(NamedTuple):
    in1: int
    in2: int
    enable: int


class Perception:
    """One worker owns YOLO/ByteTrack state. No queued inference frames."""
    def __init__(self, model, class_id, tag_id, tracker, confidence, image_size, device,
                 *, color=False, hsv=None, hue_tolerance=10, saturation_tolerance=80, value_tolerance=80):
        self.model, self.class_id, self.tag_id = model, class_id, tag_id
        self.tracker, self.confidence, self.image_size, self.device = tracker, confidence, image_size, device
        self.tag_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv2.aruco.DetectorParameters())
        self.locked_id = None
        self.locked_at = -math.inf
        self.color = color
        self.color_hsv = tuple(hsv) if hsv is not None else None
        self.hue_tolerance = hue_tolerance
        self.saturation_tolerance = saturation_tolerance
        self.value_tolerance = value_tolerance

    def detect(self, jpeg, leg):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Invalid camera JPEG')
        h, w = frame.shape[:2]
        candidates = []
        if leg == 'tag':
            corners, ids, _ = self.tag_detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            for polygon, identity in zip(corners, ids.flatten() if ids is not None else []):
                if int(identity) != self.tag_id:
                    continue
                points = polygon.reshape(4, 2)
                extent = float(np.linalg.norm(points-np.roll(points, 1, axis=0), axis=1).mean())/w
                candidates.append((f'tag:{self.tag_id}', 2*float(points[:, 0].mean())/w-1, extent, points))
        elif self.color:
            selected = self.color_hsv
            if selected is None:
                return None,frame
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            hue,saturation,value = selected
            difference = np.abs(hsv[:,:,0].astype(np.int16)-hue)
            mask = ((np.minimum(difference,180-difference) <= self.hue_tolerance)
                    & (np.abs(hsv[:,:,1].astype(np.int16)-saturation) <= self.saturation_tolerance)
                    & (np.abs(hsv[:,:,2].astype(np.int16)-value) <= self.value_tolerance)).astype(np.uint8)*255
            mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                x,y,width,height = cv2.boundingRect(contour)
                candidates.append((f'hsv:{hue},{saturation},{value}',(2*x+width)/w-1,
                                   math.sqrt(width*height/(w*h)),
                                   np.array([[x,y],[x+width,y],[x+width,y+height],[x,y+height]])))
        else:
            options = dict(classes=[self.class_id], conf=self.confidence, imgsz=self.image_size,
                           device=self.device, verbose=False)
            result = (self.model.predict(frame, **options)[0] if self.tracker == 'detect' else
                      self.model.track(frame, persist=True, tracker=self.tracker+'.yaml', **options)[0])
            boxes = result.boxes
            if boxes is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                identities = boxes.id.cpu().numpy() if boxes.id is not None else [None]*len(coordinates)
                for box, identity in zip(coordinates, identities):
                    x1, y1, x2, y2 = map(float, box)
                    if (not np.isfinite(box).all() or x2 <= x1 or y2 <= y1
                            or (self.tracker != 'detect' and identity is None)):
                        continue
                    key = f'object:{int(identity)}' if identity is not None else 'object:single'
                    extent = math.sqrt((x2-x1)*(y2-y1)/(w*h))
                    candidates.append((key, (x1+x2)/w-1, extent,
                                       np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])))
            # Lock one track instead of jumping between same-class balls as their areas change.
            if self.locked_id is not None and time.monotonic()-self.locked_at < 1.5:
                candidates = [item for item in candidates if item[0] == self.locked_id]
        if not candidates:
            return None, frame
        target = max(candidates, key=lambda item: item[2])
        if leg == 'object':
            self.locked_id, self.locked_at = target[0], time.monotonic()
        cv2.polylines(frame, [target[3].astype(np.int32)], True, (0,255,0), 2)
        return target[:3], frame


def color_signature(args, hsv):
    return 'hsv:v1:'+','.join(map(str,hsv))+f':{args.hue_tolerance},{args.saturation_tolerance},{args.value_tolerance}'


def profile_menu():
    paths = sorted(PROFILE_DIR.glob('*.json'))
    index = choose('Robot / surface profile', [p.stem for p in paths]+['Create new profile'])
    if index < len(paths):
        profile = RobotProfile.load(paths[index])
        print(f'{profile.name}: {profile.surface}; pins {profile.left}/{profile.right}; '
              f'max PWM {profile.max_pwm}; calibrated {", ".join(profile.models) or "none"}')
        if choose('Profile settings', ['Use saved settings', 'Edit hardware and initial settings']) == 0:
            return profile
    else:
        name = ask('Profile name, e.g. robot1-carpet', 'robot-floor', str,
                   lambda s: bool(s) and not (PROFILE_DIR/f'{s}.json').exists())
        profile = RobotProfile(name=name)
        profile.validate()
    old_context = (profile.surface, profile.left, profile.right)
    profile.surface = ask('Robot, surface and camera mounting notes', profile.surface)
    for key in ('left', 'right'):
        current = ','.join(map(str, getattr(profile, key)))
        setattr(profile, key, pins(ask(f'{key} motor IN1,IN2,PWM', current, str, lambda s: bool(pins(s)))))
    profile.max_pwm = ask('Maximum PWM (40..220)', profile.max_pwm, int, lambda n: 40 <= n <= 220)
    profile.probe_seconds = ask('Calibration pulse seconds (.1..1)', profile.probe_seconds, float, lambda n: .1 <= n <= 1)
    profile.settle = ask('Calibration camera settling seconds (.2..3)', profile.settle, float, lambda n: .2 <= n <= 3)
    profile.slew = ask('Motor acceleration/deceleration PWM per second (50..1500)', profile.slew,
                       float, lambda n: 50 <= n <= 1500)
    profile.camera_delay = ask('Estimated camera/network delay seconds (0..0.5)', profile.camera_delay,
                               float, lambda n: 0 <= n <= .5)
    if old_context != (profile.surface, profile.left, profile.right) or max(profile.minimum_pwm) > profile.max_pwm:
        profile.models.clear()
        profile.signatures.clear()
        profile.minimum_pwm = (0,0)
        print('Robot/surface settings changed: this profile needs fresh calibration.')
    profile.validate()
    return profile


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', action='store_true', help='Console choices, also used when run without arguments')
    parser.add_argument('--mode', choices=['preview', 'calibrate', 'drive'], default='preview')
    parser.add_argument('--profile', default='robot-floor', help='Saved robot/surface profile name')
    parser.add_argument('--weights', type=Path, default=MODELS/'red-ball-v1.pt')
    parser.add_argument('--target-class', default='Red-Ball')
    parser.add_argument('--tag-id', type=int, default=0)
    parser.add_argument('--tag-only', action='store_true', help='Use/calibrate just the tag stage')
    parser.add_argument('--color', action='store_true', help='Click HSV color in the live window instead of loading YOLO')
    parser.add_argument('--hsv',type=int,nargs=3,metavar=('H','S','V'),help='Optional explicit HSV; otherwise click or use the calibrated profile')
    parser.add_argument('--hue-tolerance',type=int,default=10)
    parser.add_argument('--saturation-tolerance',type=int,default=80)
    parser.add_argument('--value-tolerance',type=int,default=80)
    parser.add_argument('--tracker', choices=['bytetrack', 'botsort', 'detect'], default='bytetrack')
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--object-area', type=float, default=.12, help='Arrival: detected object box area / image area')
    parser.add_argument('--tag-width', type=float, default=.24, help='Arrival: mean tag edge / image width')
    parser.add_argument('--phone-ip', default=None, help='Optional PC interface override')
    parser.add_argument('--video', type=Path, help='Offline replay, preview only')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--log', type=Path, help='New JSONL log file; default runs/adaptive/<timestamp>.jsonl')
    args = parser.parse_args(argv)
    if (not 0 <= args.tag_id <= 586 or not .001 <= args.object_area <= .8
            or not .01 <= args.tag_width <= .9 or not .1 <= args.confidence <= .95
            or not 160 <= args.image_size <= 1280 or not math.isfinite(args.max_seconds) or args.max_seconds < 0):
        parser.error('Invalid tag, arrival, confidence, image size or duration setting')
    if args.video and args.mode != 'preview':
        parser.error('Offline replay cannot drive or calibrate physical motors')
    if (not 0 <= args.hue_tolerance <= 90 or not 0 <= args.saturation_tolerance <= 255
            or not 0 <= args.value_tolerance <= 255):
        parser.error('HSV tolerances: H 0..90, S/V 0..255')
    if args.hsv is not None and (not args.color or any(not 0 <= n <= limit for n,limit in zip(args.hsv,(179,255,255)))):
        parser.error('--hsv requires --color; H must be 0..179, S/V 0..255')
    return args


def setup(args):
    interactive = args.setup or len(sys.argv) == 1
    settings_path = PROFILE_DIR.parent/'console'/('adaptive-color-homing.json' if args.color else 'adaptive-homing.json')
    if interactive:
        try:
            saved = json.loads(settings_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            saved = {}
        # The wizard revalidates selections; mode always starts at preview.
        if isinstance(saved,dict):
            for key in ('target_class','tag_id','object_area','tag_width','image_size','confidence','tracker',
                        'hue_tolerance','saturation_tolerance','value_tolerance'):
                if key in saved:
                    setattr(args,key,saved[key])
            if isinstance(saved.get('weights'),str):
                args.weights = Path(saved['weights'])
        args.mode = ['preview', 'calibrate', 'drive'][choose('Adaptive homing mode',
                      ['Camera preview (no Arduino)', 'Learn robot/surface profile (moves wheels)',
                       'Drive using a learned profile'])]
        args.tag_only = choose('Mission', ['Clicked HSV color -> AprilTag' if args.color else 'YOLO object -> AprilTag',
                                          'AprilTag only']) == 1
        profile = profile_menu()
        args.profile = profile.name
        if not args.tag_only and not args.color:
            args.weights = weights_menu(args.weights)
        args.tag_id = ask('AprilTag 36h11 ID (0..586)', args.tag_id, int, lambda n: 0 <= n <= 586)
        args.object_area = ask('Ball arrival: box area fraction', args.object_area, float, lambda n: .001 <= n <= .8)
        args.tag_width = ask('Tag arrival: edge/image width', args.tag_width, float, lambda n: .01 <= n <= .9)
        if args.color and not args.tag_only:
            args.hue_tolerance = ask('HSV hue tolerance (0..90)',args.hue_tolerance,int,lambda n: 0 <= n <= 90)
            args.saturation_tolerance = ask('HSV saturation tolerance (0..255)',args.saturation_tolerance,int,lambda n: 0 <= n <= 255)
            args.value_tolerance = ask('HSV brightness tolerance (0..255)',args.value_tolerance,int,lambda n: 0 <= n <= 255)
        if not args.tag_only and not args.color:
            args.image_size = ask('YOLO image size', args.image_size, int, lambda n: 160 <= n <= 1280)
            args.confidence = ask('YOLO confidence', args.confidence, float, lambda n: .1 <= n <= .95)
            args.tracker = ['bytetrack','botsort','detect'][choose('YOLO tracker', ['ByteTrack', 'BoT-SORT', 'Detector only (single object)'])]
    else:
        profile = RobotProfile(name=args.profile)
        path = profile.path
        if path.is_file():
            profile = RobotProfile.load(path)
    model = class_id = None
    signatures = {'tag': f'apriltag36h11:{args.tag_id}'}
    if not args.tag_only and not args.color:
        if not args.weights.is_file():
            raise ValueError(f'Weights not found: {args.weights}; train/download while Internet is available')
        print('Loading and warming up local YOLO weights...', flush=True)
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
        if model.task != 'detect':
            raise ValueError('Choose YOLO detection weights')
        if interactive:
            args.target_class = class_menu(model, args.target_class)
        class_id = next((i for i, name in model.names.items() if name.casefold() == args.target_class.casefold()), None)
        if class_id is None:
            raise ValueError(f'Class {args.target_class!r} not in {model.names}')
        model.predict(np.zeros((args.image_size,args.image_size,3),np.uint8), imgsz=args.image_size,
                      device=args.device, verbose=False)
        signatures['object'] = hashlib.sha256(args.weights.read_bytes()).hexdigest()+f':{class_id}'
    hsv = args.hsv
    if args.color and not args.tag_only:
        if hsv is None and (args.mode == 'drive' or args.headless):
            hsv = profile.color_hsv
        if hsv is not None:
            signatures['object'] = color_signature(args,hsv)
        elif args.headless:
            raise ValueError('Headless color preview needs --hsv H S V or a profile with a saved color')
    # Preview never needs a body model. Driving requires real measured calibration.
    legs = ['tag'] if args.tag_only else ['object', 'tag']
    if args.mode == 'drive':
        for leg in legs:
            if leg not in profile.models or leg not in signatures or profile.signatures.get(leg) != signatures[leg]:
                raise ValueError(f'Calibrate {leg} for this profile and detector/target first (--mode calibrate)')
    if interactive and args.mode != 'preview':
        profile.save()
    if interactive:
        settings_path.parent.mkdir(parents=True,exist_ok=True)
        settings_path.write_text(json.dumps({key: str(getattr(args,key)) if key=='weights' else getattr(args,key)
            for key in ('weights','target_class','tag_id','object_area','tag_width','image_size','confidence','tracker',
                        'hue_tolerance','saturation_tolerance','value_tolerance')},
            indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return args, profile, Perception(model, class_id, args.tag_id, args.tracker, args.confidence, args.image_size, args.device,
        color=args.color,hsv=hsv,hue_tolerance=args.hue_tolerance,
        saturation_tolerance=args.saturation_tolerance,value_tolerance=args.value_tolerance), signatures


class Session:
    """Camera polling, motor ticks and UI stay independent of a single inference worker."""
    def __init__(self, args, profile, perception):
        self.args, self.profile, self.perception = args, profile, perception
        self.cam = self.capture = self.board = self.control = self.log_file = None
        self.connected = False
        self.motors = (MotorPins(*profile.left), MotorPins(*profile.right))
        self.jpeg = self.packet = None
        self.sequence = 0
        self.received = -math.inf
        self.frame = self.annotated = None
        self.last_display = self.last_control = self.next_read = 0.
        self.task = None
        self.history = None
        self.started = time.monotonic()
        self.display_hsv = None
        self.color_locked = False

    def log(self, event, **values):
        if self.log_file:
            self.log_file.write(json.dumps(dict(t=time.monotonic()-self.started,event=event,**values), allow_nan=False)+'\n')

    async def open(self):
        path = self.args.log or RUNS/'adaptive'/f'{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = path.open('x', encoding='utf-8', buffering=1)
        self.log('start', mode=self.args.mode, profile=self.profile.name, surface=self.profile.surface,
                 weights=str(self.args.weights), target_class=self.args.target_class, tag_id=self.args.tag_id)
        print(f'Log: {path}\nEsc/Q in camera window or Ctrl+C stops.', flush=True)
        if self.args.video:
            self.capture = cv2.VideoCapture(str(self.args.video))
            if not self.capture.isOpened():
                raise ValueError('Could not open replay video')
            rate = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = rate if math.isfinite(rate) and rate > 0 else 20.
        else:
            self.cam = phonesense.start()
            configure_phone_camera(self.cam, self.args.phone_ip)
        if not self.args.headless and getattr(self.perception,'color',False) and not self.args.tag_only:
            cv2.namedWindow('Adaptive robot - live')
            cv2.setMouseCallback('Adaptive robot - live',self.pick_color)
        if self.args.mode != 'preview':
            from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio
            self.board = TelemetrixUnoR4WiFiAio(transport_address=ARDUINO_IP, ip_port=TELEMETRIX_PORT,
                autostart=False, loop=asyncio.get_running_loop(), shutdown_on_exception=False,
                close_loop_on_shutdown=False, hard_reset_on_shutdown=False)
            await asyncio.wait_for(self.board.start_aio(), 15)
            self.connected = True
            await self.board.disable_all_reporting()
            for m in self.motors:
                await self.board.set_pin_mode_digital_output(m.in1)
                await self.board.set_pin_mode_digital_output(m.in2)
                await self.board.set_pin_mode_analog_output(m.enable)
            # Raw signed duty; do not add another hidden PWM floor to the learned model.
            self.control = SmoothMotorController(self.board, self.motors, min_pwm=0,
                                                  acceleration=self.profile.slew, deceleration=self.profile.slew)
            await self.control.stop()

    def pick_color(self,event,x,y,flags,param):
        if (event != cv2.EVENT_LBUTTONDOWN or self.color_locked or self.display_hsv is None
                or not 0 <= y < self.display_hsv.shape[0] or not 0 <= x < self.display_hsv.shape[1]):
            return
        self.perception.color_hsv = tuple(map(int,self.display_hsv[y,x]))
        # Keep this exact segmentation definition throughout calibration/driving.
        self.color_locked = True
        print(f'Selected HSV {self.perception.color_hsv}. Color locked for this run.',flush=True)

    async def prepare_color(self,signatures):
        await self.stop()
        if self.perception.color_hsv is None:
            print('Click the ball/object in the "Adaptive robot - live" window to select its HSV. '
                  'Motors stay stopped until selection is complete.',flush=True)
            while self.perception.color_hsv is None:
                self.pump('CLICK THE OBJECT to choose HSV | motors stopped')
                await asyncio.sleep(.01)
        self.color_locked = True
        hsv = tuple(self.perception.color_hsv)
        signature = color_signature(self.args,hsv)
        if self.args.mode == 'drive' and self.profile.signatures.get('object') != signature:
            raise ValueError('This HSV selection needs its own object calibration. Choose Learn profile first.')
        signatures['object'] = signature
        if self.args.mode == 'calibrate':
            self.profile.color_hsv = hsv
        self.log('color_selected',hsv=hsv)
        print(f'Using OpenCV HSV {hsv}; YOLO is not loaded.',flush=True)

    def pump(self, status):
        now = time.monotonic()
        jpeg = None
        if self.capture is not None and now >= self.next_read:
            ok, image = self.capture.read()
            if not ok:
                raise EOFError('Replay ended')
            self.next_read = now + 1/self.fps
            ok, encoded = cv2.imencode('.jpg', image)
            if ok:
                jpeg = encoded.tobytes()
        elif self.cam is not None:
            packet = self.cam.jpeg
            if packet is not None and packet is not self.packet:
                jpeg = packet
        if jpeg is not None:
            self.jpeg = self.packet = jpeg
            self.received, self.sequence = now, self.sequence+1
        if self.jpeg is not None and (now-self.received > .75 or (self.cam is not None and not self.cam.is_streaming)):
            raise ConnectionError('Camera disconnected/stale; stopping. Reconnect and rerun.')
        if not self.args.headless and self.jpeg is not None and now-self.last_display >= 1/25:
            image = cv2.imdecode(np.frombuffer(self.jpeg,np.uint8),cv2.IMREAD_COLOR)
            if image is not None:
                if getattr(self.perception,'color',False) and not self.color_locked:
                    # Sample the original pixels, before status text is drawn.
                    self.display_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                cv2.putText(image, status, (10,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),2)
                cv2.imshow('Adaptive robot - live', image)
                if self.annotated is not None:
                    cv2.imshow('Latest detector result', self.annotated)
            self.last_display = now
        if not self.args.headless and cv2.waitKey(1)&255 in (27,ord('q'),ord('Q')):
            raise KeyboardInterrupt
        if self.args.max_seconds and now-self.started >= self.args.max_seconds:
            raise TimeoutError('Requested run duration reached')

    async def motors_to(self, demand, *, immediate=False, floor=True):
        now = time.monotonic()
        if not immediate and now-self.last_control < .04:
            return
        if self.history:
            # Record the previous applied PWM until writes finish; commands have bounded I/O time.
            self.history.record(now, self.applied())
        self.last_control = now
        demand = np.asarray(demand, float).copy()
        # Raise tiny commands together where possible, preserving the learned
        # left/right ratio instead of flooring both wheels to the same PWM.
        active = np.abs(demand) > .02
        if floor and np.any(active):
            minimum = np.array(self.profile.minimum_pwm)/255
            required = max(1., float(np.max(minimum[active]/np.abs(demand[active]))))
            available = (self.profile.max_pwm/255)/float(np.max(np.abs(demand)))
            demand *= min(required, available)
        for index in range(2):
            if floor and abs(demand[index]) > .02:
                demand[index] = math.copysign(max(abs(demand[index]), self.profile.minimum_pwm[index]/255), demand[index])
            elif abs(demand[index]) <= .02:
                demand[index] = 0
        demand = np.clip(demand, -self.profile.max_pwm/255, self.profile.max_pwm/255)
        if self.control:
            for motor, value in zip(self.motors, demand):
                await asyncio.wait_for(self.control.set_speed(motor, float(value), immediate=immediate), .3)
        if self.history:
            self.history.record(time.monotonic(), self.applied())

    def applied(self):
        return (np.array([self.control.states[m].pwm/255 for m in self.motors])
                if self.control else np.zeros(2))

    async def wait(self, seconds, status, demand=(0.,0.), floor=False):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.pump(status)
            await self.motors_to(demand, floor=floor)
            await asyncio.sleep(.01)

    async def stop(self):
        await self.motors_to((0.,0.), immediate=True)

    async def prompt(self, text):
        await self.stop()
        # Keep camera/UI pumping during a console prompt; physical output remains zero.
        # A daemon reader does not keep Python alive if Q stops a console prompt.
        loop = asyncio.get_running_loop()
        task = loop.create_future()
        def finish(value=None, error=None):
            if not task.done():
                task.set_exception(error) if error else task.set_result(value)
        def reader():
            try:
                result = input(text+' [Enter when ready]: ')
                loop.call_soon_threadsafe(finish, result)
            except (EOFError, OSError) as exc:
                try:
                    loop.call_soon_threadsafe(finish, None, exc)
                except RuntimeError:
                    pass
            except RuntimeError:
                pass  # event loop was already closed after camera-window quit
        threading.Thread(target=reader, daemon=True).start()
        while not task.done():
            self.pump('CALIBRATION: position stationary target, then press Enter in terminal')
            await asyncio.sleep(.02)
        task.result()

    async def detect_new(self, leg, after):
        start = time.monotonic()
        while self.sequence <= after:
            self.pump('Waiting for new camera frame')
            if time.monotonic()-start > 8:
                raise TimeoutError('No fresh camera image')
            await asyncio.sleep(.01)
        stamp, sequence = self.received, self.sequence
        self.task = asyncio.create_task(asyncio.to_thread(self.perception.detect, self.jpeg, leg))
        while not self.task.done():
            self.pump(f'CALIBRATING {leg}')
            if time.monotonic()-stamp > 5:
                raise TimeoutError('Detector too slow for calibration')
            await asyncio.sleep(.01)
        target, self.annotated = self.task.result()
        self.task = None
        if time.monotonic()-stamp > .55:
            raise TimeoutError('Detection older than .55s; reduce image size or use faster local weights')
        return target, stamp, sequence

    async def stable(self, leg):
        await self.stop()
        await self.wait(self.profile.settle+self.profile.camera_delay, 'Calibration: settling')
        sequence = self.sequence
        samples = []
        for _ in range(5):
            target, stamp, sequence = await self.detect_new(leg, sequence)
            if target:
                samples.append(target)
        if len(samples) < 3 or len({t[0] for t in samples}) != 1:
            raise ValueError(f'Cannot see a stable {leg}. Keep exactly one stationary target in view and recalibrate.')
        features = np.array([[t[1],math.log(t[2])] for t in samples])
        if np.max(np.ptp(features, axis=0)) > .08:
            raise ValueError('Calibration image is not stable; increase settling time/check camera and target')
        self.stable_key = samples[0][0]
        return np.median(features, axis=0), stamp

    async def close(self):
        try:
            if self.control:
                await asyncio.wait_for(self.stop(), 1)
        finally:
            if self.task:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            if self.cam:
                self.cam.stop()
            if self.capture:
                self.capture.release()
            cv2.destroyAllWindows()
            try:
                if self.connected:
                    await asyncio.wait_for(self.board.shutdown(), 1)
            finally:
                sock = getattr(self.board, 'sock', None)
                if sock is not None and sock.writer is not None:
                    sock.writer.close()
                    await sock.writer.wait_closed()
                if self.log_file:
                    self.log_file.close()


async def calibrate(session, signatures):
    from robotics_lab.adaptive_control import EffortHistory
    profile = session.profile
    legs = ['tag'] if session.args.tag_only else ['object','tag']
    for leg in legs:
        await session.prompt(f'Calibrating {leg}: place the robot on {profile.surface}. '
                             'Fix the camera to the robot; keep the target stationary, centered and well in view. '
                             'The wheels will pulse individually in BOTH directions. Leave clear space.')
        changes, efforts, floors = [], [], [0,0]
        for axis in range(2):
            # Find the first useful PWM rather than requiring a hand-tuned friction floor.
            levels = sorted(set([min(p,profile.max_pwm) for p in (80,110,140,170,profile.max_pwm)]))
            for pwm in levels:
                pair = []
                for direction in (1,-1):
                    before, stamp = await session.stable(leg)
                    before_key = session.stable_key
                    session.history = EffortHistory()
                    session.history.record(stamp, session.applied())
                    demand = np.zeros(2)
                    demand[axis] = direction*pwm/255
                    print(f'{leg}: wheel {axis+1}, PWM {direction*pwm:+}, {profile.probe_seconds:.2f}s', flush=True)
                    try:
                        # Includes ramp-up; all actual host-applied PWM is integrated.
                        await session.wait(profile.probe_seconds+pwm/profile.slew, 'Wheel probe', demand)
                    finally:
                        await session.stop()
                    after, end_stamp = await session.stable(leg)
                    if session.stable_key != before_key:
                        raise ValueError('Calibration target identity changed during the probe; keep one target in view and retry')
                    effort = session.history.integral(stamp, end_stamp)
                    delta = after-before
                    pair.append((effort,delta))
                    session.log('probe', leg=leg, pwm=direction*pwm, wheel=axis,
                                effort=effort.tolist(), change=delta.tolist())
                if all(np.linalg.norm(delta) >= .008 for _,delta in pair):
                    floors[axis] = pwm
                    efforts.extend(u for u,_ in pair)
                    changes.extend(delta for _,delta in pair)
                    break
            if not floors[axis]:
                raise ValueError('No measurable wheel response within the PWM limit. Check wiring/grip or increase the limit in setup.')
        body = BodyModel.fit(efforts, changes)
        profile.models[leg], profile.signatures[leg] = body, signatures[leg]
        profile.minimum_pwm = tuple(max(old,new) for old,new in zip(profile.minimum_pwm,floors))
        profile.save()
        print(f'Saved {leg} model to {profile.path}; minimum observed PWM {floors}.', flush=True)
    session.history = None
    print('Calibration complete. Run again in Drive mode using this robot/surface profile.', flush=True)


async def mission(session):
    args, profile = session.args, session.profile
    leg = 'tag' if args.tag_only else 'object'
    def new_servo():
        # Preview can inspect detections/arrival, but no invented model is ever used to drive.
        model = profile.models.get(leg, BodyModel())
        return VisualServo(model, args.tag_width if leg == 'tag' else math.sqrt(args.object_area),
                           profile.max_pwm, profile.camera_delay)
    servo = new_servo()
    session.history = servo.history
    submitted = 0
    task_stamp = None
    last_log = next_report = 0.
    target_label = f'HSV {session.perception.color_hsv}' if getattr(args,'color',False) else args.target_class
    print(f'{args.mode.upper()}: {target_label} -> AprilTag {args.tag_id}; profile {profile.name}', flush=True)
    while True:
        now = time.monotonic()
        session.pump(f'{args.mode.upper()} | {leg} | profile {profile.name}')
        if session.task is not None and session.task.done():
            target, session.annotated = session.task.result()
            session.task = None
            accepted = servo.observe(target, task_stamp, now)
            session.log('observation', leg=leg, target=target, age=now-task_stamp,
                        accepted=accepted, samples=servo.model.samples)
            if servo.done:
                await session.stop()
                print(f'{leg} reached from 3 fresh detections.', flush=True)
                if leg == 'tag':
                    return
                leg = 'tag'
                servo = new_servo()
                session.history = servo.history
                await session.wait(.4, 'Switching to AprilTag')
        if session.task is None and session.sequence > submitted:
            task_stamp, submitted = session.received, session.sequence
            session.task = asyncio.create_task(asyncio.to_thread(session.perception.detect, session.jpeg, leg))
        if session.task is not None and task_stamp is not None and now-task_stamp > 5:
            raise TimeoutError('Inference stalled; robot stopping')
        if args.mode == 'drive' and session.jpeg is not None:
            demand, status = servo.command(now)
            await session.motors_to(demand)
        else:
            demand, status = np.zeros(2), 'preview: no motors'
        if now-last_log >= .2:
            session.log('control', leg=leg, status=status, requested=demand.tolist(),
                        applied=session.applied().tolist(), samples=servo.model.samples)
            last_log = now
        if now-next_report >= 1:
            age = None if servo.stamp is None else round(now-servo.stamp,2)
            print(f'{leg}: {status}; target age={age}; PWM={np.rint(session.applied()*255).astype(int).tolist()}',flush=True)
            next_report = now
        await asyncio.sleep(.01)


async def main(args, profile, perception, signatures):
    if args.video and args.mode != 'preview':
        raise ValueError('Video replay cannot connect physical motors')
    session = Session(args, profile, perception)
    normal = False
    try:
        await session.open()
        if getattr(args,'color',False) and not args.tag_only:
            await session.prepare_color(signatures)
        if args.mode == 'calibrate':
            await calibrate(session, signatures)
        else:
            await mission(session)
        normal = True
    except (KeyboardInterrupt, asyncio.CancelledError):
        normal = True  # retain bounded online updates on an ordinary user stop
        raise
    except EOFError:
        print('Replay finished; no physical motors connected.', flush=True)
    finally:
        await session.close()
        if normal and args.mode == 'drive':
            profile.save()


if __name__ == '__main__':
    try:
        asyncio.run(main(*setup(arguments())))
    except KeyboardInterrupt:
        print('\nStopped.')
    except (ValueError, OSError, TimeoutError, ConnectionError) as exc:
        raise SystemExit(str(exc)) from exc
