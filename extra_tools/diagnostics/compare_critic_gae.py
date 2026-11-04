"""Compare two completed fixed-rollout sweeps; write no files or database records."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def load_report(path, source):
    report = json.loads(Path(path).read_text())
    if report.get('status') != 'complete':
        raise ValueError('Sweep is not complete')
    if report.get('source_run_id', 3732) != source:
        raise ValueError('Unexpected source run')
    rows = report['rollouts']
    indexed = {r['run_step']: r for r in rows}
    if len(indexed) != len(rows) or sorted(indexed) != sorted(report['planned_steps']):
        raise ValueError('Duplicate or missing rollouts')
    if report['factors'] != [1, .5, .25, .1]:
        raise ValueError('Expected factors 1, 0.5, 0.25, 0.1')
    for row in rows:
        if not row['baseline_reproduced'] or not row['split_control_reproduced']:
            raise ValueError('Reproduction control failed')
        if [t['factor'] for t in row['trials']] != report['factors']:
            raise ValueError('Inconsistent factors')
        for trial in row['trials']:
            if len(trial['updates']) != 36:
                raise ValueError('Incomplete optimizer trace')
            value = trial['final_over_original']
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError('Invalid relative MSE')
    return indexed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gae95-report', required=True)
    p.add_argument('--gae85-report', required=True)
    args = p.parse_args()
    a = load_report(args.gae95_report, 3732)
    b = load_report(args.gae85_report, 2143)
    common = sorted(a.keys() & b.keys())
    if len(a) != 16 or len(b) != 19 or len(common) != 16:
        raise ValueError('Expected 16 common steps and 3 additional GAE 0.85 steps')
    print('COMMON STEPS: independently collected rollouts at matching market positions.')
    print('Targets, policies and optimizer states differ between GAE paths.')
    print('Ratios below 1 mean better fixed-target fit than that path\'s original LR.')
    print(f'{"Step":>8} {"GAE":>5} {"Nonzero rewards":>15} {"x0.5":>12} {"x0.25":>12} {"x0.1":>12}')
    for step in common:
        for gae, rows in ((.95, a), (.85, b)):
            row = rows[step]
            values = ['undefined' if t['final_over_original'] is None else f"{t['final_over_original']:.6g}" for t in row['trials'][1:]]
            print(f'{step:8d} {gae:5.2f} {row["reward_nonzero"]:15d}' + ''.join(f' {v:>12}' for v in values))
    for label, rows, steps in [('GAE .95 common', a, common), ('GAE .85 common', b, common),
                               ('GAE .85 additional selected failures', b, sorted(b.keys()-a.keys()))]:
        print('\n' + label)
        for i, factor in enumerate((.5, .25, .1), 1):
            trials = [rows[s]['trials'][i] for s in steps]
            values = [t['final_over_original'] for t in trials if t['final_over_original'] is not None]
            med = median(values) if values else None
            print(f'x{factor}: better={sum(v<1 for v in values)}/{len(values)}; '
                  f'median ratio={med}; actor changes={sum(t["actor_changed_tensors"]>0 for t in trials)}; '
                  f'clipped updates={sum(t["clipped_updates"] for t in trials)}')
    print('\nSelected samples from two training paths; not a causal GAE isolation or return test.')
    print('Read-only. Nothing changed.')


if __name__ == '__main__':
    main()
