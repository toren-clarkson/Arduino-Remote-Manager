"""Real subprocess contention tests; no server, camera or robot connection."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from webapp.server_lock import server_lock


class ServerLockTests(unittest.TestCase):
    def contender(self, path):
        source = '''
import sys
from pathlib import Path
from webapp.server_lock import ServerAlreadyRunning, server_lock
try:
    with server_lock(Path(sys.argv[1])):
        print('acquired')
except ServerAlreadyRunning:
    print('already running')
    sys.exit(2)
'''
        return subprocess.run([sys.executable, '-c', source, str(path)],
                              capture_output=True, text=True, timeout=10)

    def test_empty_lock_contention_and_release(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'server.lock'
            with server_lock(path):
                blocked = self.contender(path)
                self.assertEqual(blocked.returncode, 2, blocked.stderr)
                self.assertEqual(blocked.stdout.strip(), 'already running')
                self.assertNotIn('Traceback', blocked.stderr)
            acquired = self.contender(path)
            self.assertEqual(acquired.returncode, 0, acquired.stderr)
            self.assertEqual(acquired.stdout.strip(), 'acquired')

    def test_startup_exception_releases_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'server.lock'
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                with server_lock(path):
                    raise RuntimeError('startup failed')
            self.assertEqual(self.contender(path).returncode, 0)

    def test_duplicate_cli_never_initializes_application(self):
        from webapp.__main__ import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'.control').mkdir()
            with server_lock(root/'.control/server.lock'):
                with patch('webapp.__main__.PROJECT_ROOT', root), patch('sys.argv', ['webapp']), \
                     patch('webapp.__main__.create_app') as create, \
                     patch('sys.stderr'):
                    with self.assertRaises(SystemExit) as stopped:
                        main()
                    self.assertEqual(stopped.exception.code, 2)
                    create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
