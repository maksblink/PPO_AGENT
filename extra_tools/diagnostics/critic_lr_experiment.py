"""Compare critic LR on a captured fixed rollout; no market simulation or DB writes."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def resolve_capture(explicit):
    if explicit:return Path(explicit).expanduser().resolve()
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    engine=create_database_engine()
    try:
        with engine.connect() as c:
            c.execute(text('SET TRANSACTION READ ONLY'))
            rows=c.execute(text("SELECT id,status FROM runs WHERE name='diagnostic_capture_3732'")).mappings().all()
            if len(rows)!=1 or rows[0]['status']!='completed':
                raise RuntimeError('Expected one completed diagnostic_capture_3732 run')
            return ROOT/'artifacts'/'runs'/f'{rows[0]["id"]:08d}'/'critic_capture_404480'
    finally:engine.dispose()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',help='Trusted local capture directory; otherwise find the completed capture run in a read-only DB query')
    parser.add_argument('--factors',type=float,nargs='+',default=[0.5,0.25,0.1])
    args=parser.parse_args()
    import math
    if any(not math.isfinite(f) or f<=0 for f in args.factors):
        raise SystemExit('STOP: factors must be finite and positive')
    directory=resolve_capture(args.capture)
    meta=json.loads((directory/'runtime.json').read_text())
    # Set process-only settings before importing torch or initializing CUDA.
    for key,value in meta['env'].items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value
    import torch
    import numpy as np
    import stable_baselines3
    import platform
    from train_and_eval.training.critic_rollout_experiment import verify_capture,run_trial,exact
    verify_capture(directory)
    for key,value in [('python',platform.python_version()),('torch',torch.__version__),
                      ('sb3',stable_baselines3.__version__),('numpy',np.__version__),('cuda_build',torch.version.cuda)]:
        if meta[key]!=value:raise RuntimeError(f'Runtime mismatch for {key}: {meta[key]} != {value}')
    torch.set_num_threads(meta['threads'])
    torch.set_num_interop_threads(meta['interop_threads'])
    torch.use_deterministic_algorithms(meta['deterministic'],warn_only=meta['deterministic_warn_only'])
    torch.set_float32_matmul_precision(meta['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=meta['matmul_tf32']
    torch.backends.cudnn.allow_tf32=meta['cudnn_tf32']
    torch.backends.cudnn.deterministic=meta['cudnn_deterministic']
    torch.backends.cudnn.benchmark=meta['cudnn_benchmark']
    if meta['device'].startswith('cuda'):
        if not torch.cuda.is_available() or torch.cuda.get_device_name(meta['device'])!=meta['gpu']:
            raise RuntimeError('Capture requires the original CUDA GPU')
    output=directory.parent/'critic_lr_experiment'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report={'capture':str(directory),'runtime':meta,'baseline_reproduced':False,'split_control_reproduced':False,'trials':[]}
    def save():
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    try:
        expected=torch.load(directory/'expected_after.pt',map_location='cpu',weights_only=False)
        expected_row=json.loads((directory/'expected_diagnostics.json').read_text())
        print('Reproducing original update with original Adam grouping...',flush=True)
        state,row=run_trial(directory,output,'original')
        exact(expected,state);exact(expected_row,row)
        report['baseline_reproduced']=True;save()
        print('PASS: captured weights, Adam state, RNG and diagnostic measurements reproduced exactly.',flush=True)
        print('Checking split parameter groups at unchanged LR...',flush=True)
        state,control=run_trial(directory,output,'split_control',1.0)
        exact(expected,state);exact(expected_row,control)
        report['split_control_reproduced']=True;save()
        print('PASS: split-group control is exactly equivalent.',flush=True)
        base_lr=next(iter(expected['model']['optimizer_by_name'].values()))['options']['lr']
        def summary(label,factor,row):
            return {'label':label,'critic_lr_factor':factor,'actor_lr':base_lr,'critic_lr':base_lr*factor,
                    'before_mse':row['before']['mse'],'after_mse':row['after']['mse'],
                    'after_nmse':row['after']['nmse'],'after_ev':row['after']['ev'],
                    'peak_mse':max(u['post']['mse'] for u in row['updates']),
                    'best_mse':min(u['post']['mse'] for u in row['updates']),
                    'clipped_updates':sum(u['critic_grad_before_clip']!=u['critic_grad_after_clip'] for u in row['updates']),
                    'epochs':[{'epoch':u['epoch'],'mse':u['post']['mse']} for i,u in enumerate(row['updates'])
                              if i+1==len(row['updates']) or row['updates'][i+1]['epoch']!=u['epoch']]}
        report['trials'].append(summary('original',1.0,row))
        for i,factor in enumerate(dict.fromkeys(args.factors)):
            if factor==1.0:continue
            print(f'Testing critic LR x {factor:g}; actor LR unchanged...',flush=True)
            state,trial=run_trial(directory,output,f'factor_{i}',factor)
            # Randomness/mini-batch draw count must be identical in every trial.
            exact(expected['rng'],state['rng'])
            item=summary(f'factor_{factor:g}',factor,trial)
            item['actor_changed_tensors']=sum(
                not torch.equal(value,state['model']['weights'][name])
                for name,value in expected['model']['weights'].items()
                if name.startswith(('mlp_extractor.policy_net.','action_net.'))
            )
            for name,param in state['model']['optimizer_by_name'].items():
                wanted=base_lr*factor if name.startswith(('mlp_extractor.value_net.','value_net.')) else base_lr
                if param['options']['lr']!=wanted:raise RuntimeError(f'Unexpected LR: {name}')
            report['trials'].append(item);save()
        report['status']='complete';save()
    except Exception as error:
        report['status']='stopped';report['error']=str(error);save()
        raise
    print('\nFactor   Critic LR       Before MSE       After MSE        Peak MSE')
    for t in report['trials']:
        print(f'{t["critic_lr_factor"]:6g}   {t["critic_lr"]:.6g}   {t["before_mse"]:.6e}   {t["after_mse"]:.6e}   {t["peak_mse"]:.6e}')
    print('Report:',output/'report.json')
    print('No market simulation, database writes or original checkpoint changes. Only experiment outputs written.')

if __name__=='__main__':main()
