"""Sweep arbitrary complete captures from an explicit critic experiment."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import math
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

def summarize(report, expected_factors):
    if (report.get('status') != 'complete' or not report.get('baseline_reproduced')
            or not report.get('split_control_reproduced')):
        raise RuntimeError('Single-rollout reproduction controls did not pass')
    trials = report['trials']
    if [t['critic_lr_factor'] for t in trials] != expected_factors:
        raise RuntimeError('Unexpected trial factors')
    original = trials[0]
    for trial in trials:
        for key in ('before_mse', 'after_mse', 'peak_mse', 'best_mse'):
            if not math.isfinite(trial[key]) or trial[key] < 0:
                raise RuntimeError('Invalid MSE in report')
        if trial['before_mse'] != original['before_mse']:
            raise RuntimeError('Trials did not start from the same error')
    result = []
    for trial in trials:
        baseline = original['after_mse']
        initial = trial['before_mse']
        values = [u['mse'] for u in trial['updates']]
        result.append({
            'factor': trial['critic_lr_factor'],
            'after_mse': trial['after_mse'],
            'after_nmse': trial['after_nmse'], 'after_ev': trial['after_ev'],
            'final_over_original': trial['after_mse'] / baseline if baseline > 0 else None,
            'final_over_initial': trial['after_mse'] / initial if initial > 0 else None,
            'peak_over_initial': trial['peak_mse'] / initial if initial > 0 else None,
            'updates_above_initial': sum(value > initial for value in values),
            'updates_increasing': sum(x > prev for x, prev in zip(values, [initial] + values[:-1])),
            'clipped_updates': trial['clipped_updates'],
            'actor_changed_tensors': trial.get('actor_changed_tensors', 0),
            'updates': trial['updates'],
        })
    return result



def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', help='Trusted local diagnostic artifact directory')
    source.add_argument('--run-id', type=int, help='Completed diagnostic run ID')
    p.add_argument('--factors', nargs='+', type=float, default=[.5,.25,.1])
    p.add_argument('--steps', nargs='+', type=int, help='Subset of planned capture steps')
    args = p.parse_args()
    if any(not math.isfinite(f) or f<=0 for f in args.factors): p.error('Factors must be positive and finite')
    factors = [1.] + [f for f in dict.fromkeys(args.factors) if f != 1.]
    if args.run_id is not None:
        from sqlalchemy import text
        from train_and_eval.database.session import create_database_engine
        engine = create_database_engine()
        try:
            with engine.connect() as c:
                c.execute(text('SET TRANSACTION READ ONLY'))
                state = c.execute(text('SELECT status FROM runs WHERE id=:id'), {'id':args.run_id}).scalar_one()
                if state != 'completed': raise ValueError('Run is not completed')
        finally: engine.dispose()
        run_dir = ROOT/'artifacts/runs'/f'{args.run_id:08d}'
    else: run_dir = Path(args.run_dir).expanduser().resolve()
    plan = json.loads((run_dir/'critic_experiment.json').read_text())
    if plan.get('format') != 1: raise ValueError('Unknown experiment format')
    if args.run_id is not None and plan['run_id'] != args.run_id: raise ValueError('Experiment run ID mismatch')
    steps = sorted(set(args.steps or plan['capture_steps']))
    if not steps or not set(steps) <= set(plan['capture_steps']): raise ValueError('Unplanned capture steps')
    from train_and_eval.training.critic_rollout_experiment import verify_capture
    captures = []
    for step in steps:
        directory = run_dir/f'critic_capture_{step}'
        meta = verify_capture(directory)
        row = json.loads((directory/'expected_diagnostics.json').read_text())
        if meta['run_step'] != step or row['run_step'] != step or not row['detailed']:
            raise ValueError('Capture step/detail mismatch')
        maximum = math.ceil(meta['rollout_samples']/meta['batch_size'])*meta['n_epochs']
        count = len(row['updates'])
        if count > maximum or (meta['target_kl'] is None and count != maximum):
            raise ValueError('Capture optimizer update count differs from configuration')
        captures.append((step,directory,row))
    output = run_dir/'critic_lr_sweep'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report = {'format':1,'status':'running','experiment':plan,'run_directory':str(run_dir),
              'factors':factors,'planned_steps':steps,'rollouts':[],
              'scope':'Independent fixed-target interventions, not continuous training or a return evaluation.'}
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    save()
    try:
        for i,(step,directory,row) in enumerate(captures,1):
            print(f'Capture {i}/{len(captures)}: step {step}',flush=True)
            target = output/str(step)
            subprocess.run([sys.executable,str(Path(__file__).with_name('_trial.py')),
                            '--capture',str(directory),'--output-dir',str(target),
                            '--factors',*map(str,args.factors)],check=True)
            result = json.loads((target/'report.json').read_text())
            trials = summarize(result,factors)
            if any(len(t['updates']) != len(row['updates']) for t in trials):
                raise ValueError('Interventions executed different numbers of optimizer steps')
            report['rollouts'].append({'run_step':step,'reward_nonzero':row['reward_nonzero'],
                'target':row['target'],'before':row['before'],'actions':row['actions'],
                'baseline_reproduced':True,'split_control_reproduced':True,'trials':trials})
            save()
        report['status']='complete';save()
    except Exception as e:
        report['status']='stopped';report['error']=str(e);save();raise
    print('Step | factor | final MSE/original | actor tensors changed')
    for row in report['rollouts']:
        for t in row['trials']:
            print(row['run_step'],t['factor'],t['final_over_original'],t['actor_changed_tensors'])
    print('Report:',output/'report.json')
    print('No database writes, market simulation or changes to original captures.')

if __name__ == '__main__': main()
