"""Optional local-process evaluation gate. Training and RNG state are untouched."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import fcntl
import json
import os
import time

_path = ContextVar('queue_evaluation_lock', default=None)
_inside = ContextVar('queue_inside_evaluation', default=False)


@contextmanager
def evaluation_lock_scope(path):
    token = _path.set(Path(path) if path else None)
    try:
        yield
    finally:
        _path.reset(token)


def serialize_evaluation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        path = _path.get()
        if path is None or _inside.get():
            return function(*args, **kwargs)
        def event(phase):
            from train_and_eval.queue_execution.runtime import phase as record_phase
            record_phase(phase, evaluation=function.__name__)
            if phase == 'evaluation_end':
                record_phase('training')
        # The scheduler creates the shared parent. Never delete the lock file:
        # unlinking a live flock would allow another worker to lock a new inode.
        with path.open('a') as handle:
            event('evaluation_wait')
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.1)
            token = _inside.set(True)
            try:
                event('evaluation_start')
                return function(*args, **kwargs)
            finally:
                _inside.reset(token)
                fcntl.flock(handle, fcntl.LOCK_UN)
                event('evaluation_end')
    return wrapped
