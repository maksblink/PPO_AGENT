"""Device requirements and model initialization shared by claimed workers."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def phase(name, **details):
    event = dict(phase=name, time=time.time(), worker_pid=os.getpid(), **details)
    print(json.dumps(event), flush=True)
    destination = os.environ.get('PPO_QUEUE_PHASE_PATH')
    if destination:
        path = Path(destination)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(event))
        temporary.replace(path)


def require_training_config(config):
    if str(config.ppo.device) != 'cuda':
        raise RuntimeError('Training requires ppo.device=cuda; refusing an implicit device change')
    if str(config.evaluation.device) != 'cpu':
        raise RuntimeError('Evaluation requires evaluation.device=cpu')


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is disabled')


def check_model_device(model, expected):
    devices = {p.device.type for p in model.policy.parameters()}
    if devices != {expected} or model.device.type != expected:
        raise RuntimeError(f'Expected {expected} model, got {sorted(devices)} / {model.device}')


@contextmanager
def model_initialization():
    """Keep thread policy local to opted-in queue workers; always restore it."""
    import torch
    enabled = os.environ.get('PPO_QUEUE_THREAD_POLICY') == '16-to-1'
    previous = torch.get_num_threads()
    if enabled:
        torch.set_num_threads(16)
        phase('initialization', threads=16, cpus=sorted(os.sched_getaffinity(0)))
    try:
        yield
    finally:
        if enabled:
            torch.set_num_threads(previous)
            phase('training', threads=torch.get_num_threads(), cpus=sorted(os.sched_getaffinity(0)))
