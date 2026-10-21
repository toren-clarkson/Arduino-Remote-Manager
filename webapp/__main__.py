"""python -m webapp [--port 5000]"""

import argparse
from waitress import serve
from .app import create_app
from .server_lock import ServerAlreadyRunning, server_lock
from robotics_lab.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description='Main robotics project control room (localhost only)')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port from 1024 to 65535.')
    control = PROJECT_ROOT/'.control'
    control.mkdir(exist_ok=True)
    try:
        # Acquire before creating the app: a duplicate must not interrupt jobs
        # or update the heartbeat belonging to the existing server.
        with server_lock(control/'server.lock'):
            application = create_app()
            store = application.extensions['store']
            print(f'Control room: http://127.0.0.1:{args.port}', flush=True)
            print(f'Managing: {store.root}', flush=True)
            try:
                serve(application, host='127.0.0.1', port=args.port, threads=8,
                      max_request_body_size=2*1024**3)
            finally:
                application.extensions['jobs'].close()
    except ServerAlreadyRunning:
        parser.error('A control-room server is already managing Main. Use its existing localhost URL.')


if __name__ == '__main__':
    main()
