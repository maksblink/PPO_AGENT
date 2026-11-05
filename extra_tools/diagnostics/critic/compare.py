"""Compare completed sweep reports; no DB access or file writes."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def load_report(path):
    data = json.loads(Path(path).read_text())
    if data.get('status') != 'complete': raise ValueError('Incomplete report')
    rows = data['rollouts']; indexed = {r['run_step']:r for r in rows}
    if len(indexed)!=len(rows) or sorted(indexed)!=sorted(data['planned_steps']):
        raise ValueError('Missing/duplicate rollout records')
    for row in rows:
        if not row['baseline_reproduced'] or not row['split_control_reproduced']:
            raise ValueError('Reproduction controls failed')
        if [t['factor'] for t in row['trials']] != data['factors']: raise ValueError('Factor mismatch')
        for t in row['trials']:
            value=t['final_over_original']
            if value is not None and (not math.isfinite(value) or value<0): raise ValueError('Invalid MSE ratio')
    return data,indexed


def flatten(value, prefix=''):
    if isinstance(value,dict):
        return {k:v for key,item in value.items() for k,v in flatten(item,prefix+key+'.').items()}
    return {prefix.rstrip('.'):value}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', action='append', required=True, help='Repeat for each report')
    p.add_argument('--steps', nargs='+', type=int, help='Explicit shared subset')
    args=p.parse_args()
    loaded=[load_report(path) for path in args.report]
    common=set.intersection(*(set(rows) for _,rows in loaded))
    if args.steps:
        if not set(args.steps)<=common: raise ValueError('Requested steps are not present in every report')
        common=set(args.steps)
    if not common: raise ValueError('No common steps; do not compare unmatched summaries')
    if any(d['factors']!=loaded[0][0]['factors'] for d,_ in loaded): raise ValueError('Reports use different factors')
    configs=[flatten(d.get('experiment',{}).get('normalized_config',{})) for d,_ in loaded]
    keys=set().union(*(c.keys() for c in configs))
    print('Configuration differences (including identity and ancestry):')
    for k in sorted(keys):
        vals=[c.get(k,'<missing>') for c in configs]
        if any(v!=vals[0] for v in vals): print(k,':',vals)
    print('Matching step numbers do not imply identical observations, targets or policies.')
    for path,(data,rows) in zip(args.report,loaded):
        print('\nREPORT:',path)
        print('Shared selected steps:',len(common),'Other steps excluded:',len(rows)-len(common))
        for f in data['factors']:
            trials=[next(t for t in rows[s]['trials'] if t['factor']==f) for s in sorted(common)]
            ratios=[t['final_over_original'] for t in trials if t['final_over_original'] is not None]
            print(f'factor={f:g}; improved={sum(v<1 for v in ratios)}/{len(ratios)}; '
                  f'median ratio={median(ratios) if ratios else None}; '
                  f'actor changes={sum(t.get("actor_changed_tensors",0)>0 for t in trials)}; '
                  f'clipped updates={sum(t["clipped_updates"] for t in trials)}')
        print('Step | factor | final MSE/original')
        for step in sorted(common):
            for t in rows[step]['trials']: print(step,t['factor'],t['final_over_original'])
    print('\nSelected diagnostic samples; no causal or trading-performance conclusion is implied.')
    print('Read-only. Nothing changed.')

if __name__=='__main__': main()
