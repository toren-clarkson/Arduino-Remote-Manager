"""Latest-frame loopback IPC; streaming JPEGs never requires per-frame disk I/O."""

import hmac
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class LiveCamera:
    def __init__(self):
        self.token = secrets.token_hex(32)
        self.condition = threading.Condition()
        self.frames = {}
        self.closed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if not hmac.compare_digest(self.headers.get('X-Preview-Token', ''), owner.token):
                    self.send_error(403)
                    return
                parsed = urlsplit(self.path)
                channel = parse_qs(parsed.query).get('channel', ['camera'])[0]
                if parsed.path not in {'/frame', '/stream'} or channel not in {'camera', 'observation'}:
                    self.send_error(404)
                    return
                if parsed.path == '/frame':
                    with owner.condition:
                        item = owner.frames.get(channel)
                    if item is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(item[1])))
                    self.end_headers()
                    self.wfile.write(item[1])
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                sequence = None
                try:
                    while True:
                        with owner.condition:
                            owner.condition.wait_for(lambda: owner.closed or (
                                channel in owner.frames and owner.frames[channel][0] != sequence), timeout=1)
                            if owner.closed:
                                break
                            item = owner.frames.get(channel)
                        if item is None or item[0] == sequence:
                            self.wfile.write(b'\r\n')  # keep a camera-gap proxy connection alive
                        else:
                            sequence, data = item
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+
                                             str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n')
                        self.wfile.flush()
                    self.wfile.write(b'--frame--\r\n')
                except (OSError, ConnectionError):
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True)
        self.thread.start()

    @property
    def credentials(self):
        return dict(port=self.server.server_port, token=self.token)

    def publish(self, jpeg, sequence, channel='camera'):
        with self.condition:
            if not self.closed:
                self.frames[channel] = (sequence, jpeg)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
