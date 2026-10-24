"""Serialize destructive maintenance. Rename also excludes managed hardware sessions."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import fcntl
import os


@contextmanager
def maintenance_guard(root, *, exclude_managed=False):
    root = Path(root).resolve()
    directory = root / 'extra_tools/maintenance/run_rename'
    current = root
    for part in Path('extra_tools/maintenance/run_rename').parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Maintenance path is a symlink: {current}')
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / '.maintenance.lock']
    if exclude_managed:
        paths.append(Path('/tmp') / f'ppo-managed-hardware-{os.getuid()}.lock')
    with ExitStack() as stack:
        for path in paths:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stream = stack.enter_context(os.fdopen(fd, 'a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another maintenance operation or managed queue is active') from None
        yield


@contextmanager
def activity_guard(root, *, exclusive=False):
    """Queues share this lock; rename takes it exclusively across DB/file recovery."""
    root = Path(root).resolve()
    directory = root / 'extra_tools/maintenance/run_rename'
    current = root
    for part in Path('extra_tools/maintenance/run_rename').parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Activity path is a symlink: {current}')
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / '.activity.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Queue/rename activity conflict; no operation started') from None
        if not exclusive and (directory / 'pending.json').exists():
            raise RuntimeError('Pending rename: run rename_runs.py --recover before starting queues')
        yield
