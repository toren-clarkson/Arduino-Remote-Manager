"""One server per workspace, including Windows byte-range lock handling."""

import os
from contextlib import contextmanager


class ServerAlreadyRunning(Exception):
    pass


@contextmanager
def server_lock(path):
    # Do not read or initialize the byte before acquiring it: on Windows even
    # reading a byte locked by another process raises PermissionError.
    with path.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ServerAlreadyRunning from exc
        # Locking beyond EOF is supported; only the owner initializes an empty file.
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b'0')
            handle.flush()
        yield
        # Closing the handle releases the lock, including on startup exceptions.
