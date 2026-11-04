"""Run guarded fixed-rollout critic LR comparisons on 16 preselected captures."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def resolve_run(explicit, source_run_id=3732):
    if explicit:
        return Path(explicit).expanduser().resolve()
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    engine = create_database_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text('SET TRANSACTION READ ONLY'))
            rows = connection.execute(text(
                "SELECT id,status FROM runs WHERE name=:name"
            ), {"name": f"diagnostic_capture_multi_{source_run_id}"}).mappings().all()
            if len(rows) != 1 or rows[0]['status'] != 'completed':
                raise RuntimeError(f'Expected one completed diagnostic_capture_multi_{source_run_id} run')
            return ROOT / 'artifacts/runs' / f'{rows[0]["id"]:08d}'
    finally:
        engine.dispose()


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
        if not values:
            raise RuntimeError('Missing per-update measurements')
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', help='Trusted local multi-capture run directory; default: resolve completed run in DB')
    parser.add_argument('--factors', type=float, nargs='+', default=[0.5, 0.25, 0.1])
    parser.add_argument('--source-run-id', type=int, choices=[3732, 2143], default=3732)
    args = parser.parse_args()
    if any(not math.isfinite(f) or f <= 0 for f in args.factors):
        parser.error('Factors must be finite and positive')
    factors = [1.0] + [f for f in dict.fromkeys(args.factors) if f != 1.0]
    run_dir = (resolve_run(args.run_dir) if args.source_run_id == 3732
               else resolve_run(args.run_dir, args.source_run_id))
    from train_and_eval.training.critic_rollout_experiment import (
        MULTI_CAPTURE_STEPS, GAE85_CAPTURE_STEPS, verify_capture, step_category,
    )
    steps = GAE85_CAPTURE_STEPS if args.source_run_id == 2143 else MULTI_CAPTURE_STEPS
    captures = []
    # Validate every input before executing the first experiment.
    for step in steps:
        directory = run_dir / f'critic_capture_{step}'
        meta = verify_capture(directory)
        if args.source_run_id == 2143 and (meta.get('gae_lambda') != 0.85 or meta.get('gamma') != 0.9):
            raise RuntimeError('Capture GAE/gamma mismatch for source 2143')
        row = json.loads((directory / 'expected_diagnostics.json').read_text())
        if meta['run_step'] != step or row['run_step'] != step or not row['detailed']:
            raise RuntimeError(f'Capture identity/detail mismatch at {step}')
        if len(row['updates']) != 36:
            raise RuntimeError(f'Expected 36 optimizer steps at {step}')
        captures.append((step, directory, row))
    output = run_dir / 'critic_lr_sweep' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'run_directory': str(run_dir), 'factors': factors,
              'source_run_id': args.source_run_id, 'planned_steps': list(steps), 'rollouts': [],
              'scope': 'Independent fixed-rollout interventions; not a new training trajectory or a return evaluation.'}
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    try:
        for index, (step, directory, row) in enumerate(captures, 1):
            print(f'\n=== {index}/{len(captures)}: step {step} ({step_category(step)}) ===', flush=True)
            target = output / str(step)
            # Separate processes restore runtime/thread/env settings before CUDA init.
            subprocess.run([
                sys.executable, str(Path(__file__).with_name('critic_lr_experiment.py')),
                '--capture', str(directory), '--output-dir', str(target),
                '--factors', *map(str, args.factors),
            ], check=True)
            trial_report = json.loads((target / 'report.json').read_text())
            report['rollouts'].append({
                'run_step': step, 'category': step_category(step),
                'reward_nonzero': row['reward_nonzero'], 'actions': row['actions'],
                'target': row['target'], 'before': row['before'],
                'baseline_reproduced': True, 'split_control_reproduced': True,
                'trials': summarize(trial_report, factors),
            })
            save()
        report['status'] = 'complete'
        save()
    except Exception as error:
        report['status'] = 'stopped'
        report['error'] = str(error)
        save()
        print('Partial report:', output / 'report.json', flush=True)
        raise
    print('\nFinal MSE / original final MSE (below 1 = better fixed-target fit)')
    print(f'{"Step":>8} {"Category":>18} {"Rewards != 0":>12}' + ''.join(f' {f:>12g}' for f in factors))
    for rollout in report['rollouts']:
        print(f'{rollout["run_step"]:8d} {rollout["category"]:>18} {rollout["reward_nonzero"]:12d}' +
              ''.join(f' {t["final_over_original"]:12.5g}' if t['final_over_original'] is not None else f'{" n/a":>13}'
                      for t in rollout['trials']))
    print('Report:', output / 'report.json')
    print('No market simulation or database writes. Captures and original checkpoints unchanged.')


if __name__ == '__main__':
    main()
