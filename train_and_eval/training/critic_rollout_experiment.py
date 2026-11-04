"""Temporary capture/replay of one fixed PPO rollout, independent of database."""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import platform

import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO

CAPTURE_NAME = 'diagnostic_capture_3732'
CAPTURE_STEP = 404480


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rng_state(cuda):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if cuda else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def parameter_state(model):
    optimizer = model.policy.optimizer
    groups = {id(p): {k: cpu_copy(v) for k,v in g.items() if k != 'params'}
              for g in optimizer.param_groups for p in g['params']}
    return {'weights': cpu_copy(model.policy.state_dict()),
            'optimizer_by_name': {
                name: {'state': cpu_copy(optimizer.state.get(p, {})),
                       'options': groups[id(p)]}
                for name,p in model.policy.named_parameters()},
            'n_updates': model._n_updates}


def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):
            raise RuntimeError(f'Reproduction mismatch: {path}')
    elif isinstance(a, np.ndarray):
        if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):
            raise RuntimeError(f'Reproduction mismatch: {path}')
    elif isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():raise RuntimeError(f'Keys mismatch: {path}')
        for key in a:exact(a[key],b[key],f'{path}.{key}')
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise RuntimeError(f'Sequence mismatch: {path}')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,f'{path}[{i}]')
    elif a!=b:
        raise RuntimeError(f'Reproduction mismatch: {path}: {a!r} != {b!r}')


def runtime(model):
    cuda = model.device.type == 'cuda'
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'sb3': stable_baselines3.__version__, 'numpy': np.__version__,
            'device': str(model.device), 'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(model.device) if cuda else None,
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'env': {k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'train_hash': hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest()}


def capture_before(model, diagnostic_path, local_step):
    """Only explicit capture runs call this; preserve all RNG state."""
    directory = Path(diagnostic_path).parent / f'critic_capture_{local_step}'
    directory.mkdir(exist_ok=False)
    state = rng_state(model.device.type == 'cuda')
    try:
        meta = runtime(model)
        meta.update({'format': 1, 'run_step': local_step,
                     'model_step': int(model.num_timesteps),
                     'progress_remaining': model._current_progress_remaining})
        (directory/'runtime.json').write_text(json.dumps(meta,indent=2))
        model.save(directory/'before.zip')
        torch.save({'buffer': model.rollout_buffer, 'rng': state}, directory/'rollout.pt')
    finally:
        restore_rng(state)
    return directory


def capture_after(model, directory, row):
    state = rng_state(model.device.type == 'cuda')
    try:
        torch.save({'model': parameter_state(model), 'rng': state}, directory/'expected_after.pt')
        (directory/'expected_diagnostics.json').write_text(json.dumps(row,allow_nan=False))
        names = ('runtime.json','before.zip','rollout.pt','expected_after.pt','expected_diagnostics.json')
        # This marker is written last; interrupted captures cannot be replayed.
        (directory/'complete.json').write_text(json.dumps({name:sha(directory/name) for name in names},indent=2))
    finally:
        restore_rng(state)


def verify_capture(directory):
    directory=Path(directory)
    hashes=json.loads((directory/'complete.json').read_text())
    expected={'runtime.json','before.zip','rollout.pt','expected_after.pt','expected_diagnostics.json'}
    if set(hashes)!=expected:raise RuntimeError('Invalid capture file inventory')
    for name,value in hashes.items():
        if sha(directory/name)!=value:raise RuntimeError(f'Capture checksum mismatch: {name}')
    return json.loads((directory/'runtime.json').read_text())


def split_critic_lr(model, factor):
    """Retain Adam and its per-parameter history; change critic LR only at step."""
    if not np.isfinite(factor) or factor<=0:raise ValueError('LR factor must be finite and positive')
    optimizer=model.policy.optimizer
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups)!=1:
        raise RuntimeError('Experiment requires the original single-group Adam')
    critic=[p for n,p in model.policy.named_parameters() if n.startswith(('mlp_extractor.value_net.','value_net.'))]
    actor=[p for n,p in model.policy.named_parameters() if n.startswith(('mlp_extractor.policy_net.','action_net.'))]
    if set(map(id,critic+actor)) != set(map(id,model.policy.parameters())):
        raise RuntimeError('Unexpected shared parameters')
    group=optimizer.param_groups[0]
    options={k:copy.deepcopy(v) for k,v in group.items() if k!='params'}
    ids=set(map(id,critic))
    group['params']=[p for p in group['params'] if id(p) not in ids]
    optimizer.add_param_group(dict(options,params=critic))
    # PPO sets all group LRs at train entry. Override only the critic group
    # immediately before every optimizer step, after PPO's schedule update.
    def set_lr(opt,args,kwargs):
        opt.param_groups[1]['lr']=opt.param_groups[0]['lr']*factor
    return optimizer.register_step_pre_hook(set_lr)


def run_trial(directory, output, label, factor=None):
    from stable_baselines3.common.logger import configure
    from .critic_diagnostics import train_observed
    directory,output=Path(directory),Path(output)
    meta=json.loads((directory/'runtime.json').read_text())
    # Only use captures produced locally by this tool: these contain Python objects.
    payload=torch.load(directory/'rollout.pt',map_location='cpu',weights_only=False)
    model=PPO.load(directory/'before.zip',device=meta['device'])
    model.rollout_buffer=payload['buffer']
    model.rollout_buffer.device=model.device
    model._current_progress_remaining=meta['progress_remaining']
    model.set_logger(configure(folder=str(output/('log_'+label)),format_strings=[]))
    handle=split_critic_lr(model,factor) if factor is not None else None
    trace=output/(label+'.jsonl')
    if trace.exists():raise RuntimeError(f'Refusing to append to previous trial: {trace}')
    restore_rng(payload['rng'])
    try:
        train_observed(model,trace,meta['run_step'])
        result={'model':parameter_state(model),'rng':rng_state(model.device.type=='cuda')}
        row=json.loads(trace.read_text())
    finally:
        if handle is not None:handle.remove()
    return result,row
