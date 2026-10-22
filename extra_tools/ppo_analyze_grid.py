#!/usr/bin/env python3
"""Grid and seed comparison analysis v3. Read-only PostgreSQL snapshot; writes only HTML and JSON reports.
Run from PPO_AGENT: python extra_tools/ppo_analyze_grid.py --all-runs
No training, model loading, hardware changes, database updates or network requests.
"""
from __future__ import annotations
import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
from itertools import combinations, product
import json
import math
from pathlib import Path
import re
import statistics as st
import sys
import zipfile

VERSION = '3.0.0'
EPS = 1e-10
METRICS = ('agent_return', 'balanced_score', 'agent_max_drawdown')
EXTRA = ('market_exposure', 'round_trips', 'profit_factor', 'win_rate', 'always_long_return')
SCOPES = {'TRAIN': 'run_training', 'VAL': 'run_validation'}


def serial(x):
    if isinstance(x, Decimal): return float(x)
    if isinstance(x, (datetime, Path)): return str(x)
    raise TypeError(type(x).__name__)


def canonical(x):
    return json.dumps(x, sort_keys=True, default=serial, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def finite(x):
    return x is not None and math.isfinite(float(x))


def med(xs):
    xs = [float(x) for x in xs if finite(x)]
    return st.median(xs) if xs else None


def quantile(xs, p):
    xs = sorted(float(x) for x in xs if finite(x))
    if not xs: return None
    k = (len(xs)-1)*p; lo = int(k); hi = math.ceil(k)
    return xs[lo] + (xs[hi]-xs[lo])*(k-lo)


def outcome(delta):
    return 'better' if delta > EPS else 'worse' if delta < -EPS else 'tie'


def delta_stats(xs):
    counts = Counter(outcome(x) for x in xs)
    return dict(n=len(xs), better=counts['better'], worse=counts['worse'], tie=counts['tie'],
                median=med(xs), mean=st.mean(xs) if xs else None)


def config_axes(c):
    p = c['ppo']
    return dict(architecture='x'.join(map(str, p['hidden_sizes'])),
                learning_rate=float(p['learning_rate']), gamma=float(p['gamma']),
                n_epochs=int(p['n_epochs']), seed=int(c['run']['seed']))


def fixed_config(c):
    # Identity, logging and output options are not scientific conditions.
    # Device is classified separately from saved checkpoint metadata.
    result = {k: c.get(k) for k in ('data', 'environment', 'training', 'ppo', 'evaluation')}
    result = json.loads(canonical(result))
    for k in ('hidden_sizes', 'learning_rate', 'gamma', 'n_epochs', 'device'):
        result['ppo'].pop(k, None)
    result['evaluation'].pop('device', None)
    return result


def safe_file(root, relative):
    p = (root / relative).resolve()
    p.relative_to(root.resolve())
    return p


def resolve_queue(root, queue):
    path = Path(queue)
    if path.suffix in ('.yml', '.yaml'):
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ValueError('Queue manifest not found: ' + str(path))
        return path.resolve()
    directory = root / 'configs' / 'search_queues'
    matches = sorted(p for p in directory.rglob('*')
                     if p.is_file() and p.suffix in ('.yml', '.yaml') and p.stem == queue)
    if len(matches) != 1:
        raise ValueError(f'Expected one manifest named {queue}; found {len(matches)}')
    return matches[0].resolve()


def load_queue(root, queue):
    import yaml
    p = resolve_queue(root, queue)
    q = yaml.safe_load(p.read_text())
    if q.get('queue_schema_version') != 1 or not isinstance(q.get('configs'), list) or not q['configs']:
        raise ValueError('Unsupported or empty queue manifest')
    configs, hashes = {}, {}
    # Use the project's validator/defaults, without hashing/reading market data.
    from train_and_eval.run_config import load_run_config
    for rel in q['configs']:
        path = safe_file(root, rel)
        loaded = load_run_config(path, verify_data=False)
        c = json.loads(loaded.normalized_json)
        name = c['run']['name']
        if name in configs: raise ValueError('Duplicate queue run name: ' + name)
        configs[name] = c
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return q, configs, {'queue_path': str(p), 'queue_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'config_sha256': hashes}


def snapshot(names):
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    engine = create_database_engine()
    try:
        with engine.connect() as con, con.begin():
            con.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
            con.execute(text("SET LOCAL statement_timeout = '120s'"))
            runs = [dict(x) for x in con.execute(text('''
                SELECT r.*, p.name AS parent_name, p.id AS parent_id
                FROM runs r LEFT JOIN checkpoints c ON c.id=r.source_checkpoint_id
                LEFT JOIN runs p ON p.id=c.run_id WHERE (:all_runs OR r.name=ANY(:names))
            '''), {'names': names or [], 'all_runs': names is None}).mappings()]
            ids = [r['id'] for r in runs]
            cps = [dict(x) for x in con.execute(text('SELECT * FROM checkpoints WHERE run_id=ANY(:ids)'), {'ids': ids}).mappings()]
            evs = [dict(x) for x in con.execute(text('''
                SELECT e.*, c.run_id, c.run_step AS checkpoint_run_step, c.model_step AS checkpoint_model_step
                FROM evaluations e JOIN checkpoints c ON c.id=e.checkpoint_id
                WHERE c.run_id=ANY(:ids) AND e.trigger='final'
                  AND e.data_scope IN ('run_training','run_validation')
                ORDER BY e.id
            '''), {'ids': ids}).mappings()]
            counts = {x[0]: x[1] for x in con.execute(text('SELECT run_id,count(*) FROM training_metrics WHERE run_id=ANY(:ids) GROUP BY run_id'), {'ids': ids})}
            timestamp = con.scalar(text('SELECT transaction_timestamp()'))
    finally: engine.dispose()
    return runs, cps, evs, counts, str(timestamp)


def device_metadata(root, cps):
    flags = set(); problems = []
    if not cps: return 'UNKNOWN', ['No checkpoint']
    for cp in cps:
        try:
            p = safe_file(root, cp['relative_path'])
            p.relative_to((root/'artifacts').resolve())
            with zipfile.ZipFile(p) as z:
                member = z.getinfo('system_info.txt')
                if member.file_size > 100000: raise ValueError('Oversized system_info.txt')
                text = z.read(member).decode('utf-8')
            m = re.search(r'^\s*-?\s*GPU Enabled:\s*(True|False)\s*$', text, re.M)
            if not m: raise ValueError('Missing GPU Enabled flag')
            flags.add(m[1])
        except Exception as e: problems.append(f"Checkpoint #{cp['id']}: {e}")
    if problems: return 'UNKNOWN', problems
    return ('GPU' if flags == {'True'} else 'CPU' if flags == {'False'} else 'MIXED'), []


def prepare(root, configs, runs, cps, evs, counts):
    issues = []; valid = []; inventory = []
    def issue(name, level, reason): issues.append(dict(name=name, level=level, reason=reason))
    by_name = defaultdict(list); by_cp = defaultdict(list); by_ev = defaultdict(list)
    for r in runs: by_name[r['name']].append(r)
    for c in cps: by_cp[c['run_id']].append(c)
    cp_by_id = {c['id']:c for c in cps}
    for e in evs: by_ev[(e['run_id'],e['data_scope'])].append(e)
    depths = {}
    def depth(name, visiting=()):
        if name in depths: return depths[name]
        if name in visiting: raise ValueError('Config dependency cycle: '+name)
        c = configs[name]['continuation']
        if c['mode'] == 'fresh': d = 0
        else:
            parent = c.get('source_run')
            if parent not in configs: raise ValueError('Source config outside queue: '+str(parent))
            d = depth(parent, visiting+(name,)) + 1
        depths[name] = d
        return d
    bad_depths = {}
    for name in configs:
        try: depth(name)
        except (ValueError, KeyError, TypeError) as error:
            bad_depths[name] = str(error)
    for name, target in configs.items():
        if name in bad_depths:
            issue(name,'EXCLUDE',bad_depths[name])
            inventory.append(dict(name=name,status='invalid ancestry'))
            continue
        matches = by_name[name]
        if len(matches) != 1:
            issue(name, 'EXCLUDE', f'Expected one DB run; found {len(matches)}')
            inventory.append(dict(name=name,status='missing/ambiguous')); continue
        r = matches[0]; c = r['normalized_config_json']; problems = []
        if isinstance(c,str): c=json.loads(c)
        row = dict(id=r['id'],name=name,status=r['status'],**config_axes(c),stage=depths[name],
                   source= r['parent_name'], git_commit=r.get('git_commit'), training_metrics=counts.get(r['id'],0),
                   stopped_early=r['stopped_early'],steps_completed=r['training_steps_completed'])
        inventory.append(row.copy())
        if r['status'] != 'completed': problems.append('Run not completed')
        if config_axes(c) != config_axes(target) or fixed_config(c) != fixed_config(target):
            problems.append('Scientific config in DB differs from queue config')
        parent = target['continuation'].get('source_run') if depths[name] else None
        if r['parent_name'] != parent or (parent is None and r['source_checkpoint_id'] is not None):
            problems.append('Source checkpoint ancestry differs from queue')
        if parent:
            source_cp = cp_by_id.get(r['source_checkpoint_id'])
            if target['continuation'].get('checkpoint') == 'final' and (not source_cp or source_cp['save_reason'] != 'final'):
                problems.append('Continuation does not use a final parent checkpoint')
            pc = configs[parent]
            if config_axes(pc) != config_axes(target) or fixed_config(pc) != fixed_config(target):
                problems.append('Chain changes scientific settings; simple stage comparison unsupported')
        if r['cycle_id'] is not None: problems.append('Walk-forward run is not a stage-one grid run')
        row['train_device_config'] = c['ppo'].get('device')
        row['eval_device_config'] = c['evaluation'].get('device')
        row['device'], notes = device_metadata(root, by_cp[r['id']])
        for note in notes: issue(name,'WARNING',note)
        for label, scope in SCOPES.items():
            completed = [e for e in by_ev[(r['id'],scope)] if e['status']=='completed']
            if not completed: problems.append('Missing completed final '+label); continue
            if len(completed)>1:
                problems.append(f'{label}: ambiguous multiple completed final evaluations')
                continue
            e = completed[-1]
            if any(not finite(e.get(k)) for k in METRICS+('market_exposure','round_trips','always_long_return')):
                problems.append('Missing/nonfinite core '+label+' metric'); continue
            if e['steps_completed'] != e['steps_expected']: problems.append(label+' evaluation incomplete')
            if e['checkpoint_run_step'] != r['training_steps_completed']:
                problems.append(label+' checkpoint is not at last completed training step')
            row[label] = {k:(float(e[k]) if finite(e.get(k)) else None) for k in METRICS+EXTRA}
            row[label].update(evaluation_id=e['id'],checkpoint_id=e['checkpoint_id'],
                              signature={k:e.get(k) for k in ('data_sha256','evaluation_start_index','evaluation_end_index',
                                  'evaluation_start_at','evaluation_end_at','lookback_rows','policy_mode','threshold_action','probability_threshold','seed')})
            row[label]['excess_return'] = row[label]['agent_return']-row[label]['always_long_return']
        if all(x in row for x in SCOPES) and row['TRAIN']['checkpoint_id'] != row['VAL']['checkpoint_id']:
            problems.append('TRAIN and VAL use different final checkpoints')
        for p in problems: issue(name,'EXCLUDE',p)
        if problems: continue
        row['seconds'] = ((r['finished_at']-r['started_at']).total_seconds()
                          if r['finished_at'] and r['started_at'] else None)
        row['cohort'] = digest(dict(fixed=fixed_config(c), data=r['data_sha256'],
                                    evals={k:row[k]['signature'] for k in SCOPES},device=(row['device'],row['train_device_config'],row['eval_device_config'])))[:16]
        # Rank each seed separately even if evaluation metadata use a constant seed.
        row['cohort'] = digest(dict(legacy_cohort=row['cohort'], seed=row['seed']))[:16]
        signatures = {
            scope: {k: v for k, v in row[scope]['signature'].items() if k != 'seed'}
            for scope in SCOPES
        }
        continuation = {k: v for k, v in c['continuation'].items()
                        if k not in ('source_run', 'mode')}
        row['seed_comparison_cohort'] = digest(dict(
            fixed=fixed_config(c), data=r['data_sha256'], evals=signatures,
            device=(row['device'], row['train_device_config'], row['eval_device_config'])
        ))[:16]
        row['continuation_options'] = continuation
        row['source_checkpoint_id'] = r['source_checkpoint_id']
        row['width_cost_proxy'] = sum(a*b for a,b in zip(c['ppo']['hidden_sizes'],c['ppo']['hidden_sizes'][1:]))
        valid.append(row)
    # Unknown/mixed metadata must be excluded before propagating ancestor exclusions.
    for r in list(valid):
        if r['device'] in ('UNKNOWN','MIXED'):
            issue(r['name'],'EXCLUDE','Unknown/mixed saved device metadata'); valid.remove(r)
    # Exclude descendants of invalid or missing ancestors; never compare broken chains.
    while True:
        names = {r['name'] for r in valid}
        bad = [r for r in valid if r['source'] and r['source'] not in names]
        if not bad: break
        for r in bad: issue(r['name'],'EXCLUDE','Ancestor excluded from analysis'); valid.remove(r)
    return valid, inventory, issues


def cell(r, omit=()):
    keys=('cohort','architecture','learning_rate','gamma','n_epochs','seed','stage')
    return tuple((k,r[k]) for k in keys if k not in omit)


def summarize(rs):
    out = dict(n=len(rs), chains=len({cell(r,('stage',)) for r in rs}))
    for scope in SCOPES:
        for key in METRICS+EXTRA+('excess_return',):
            vals=[r[scope][key] for r in rs]
            out[f'{scope}_{key}_median']=med(vals)
        vals=[r[scope]['agent_return'] for r in rs]
        out[f'{scope}_return_q25']=quantile(vals,.25);out[f'{scope}_return_q75']=quantile(vals,.75)
        out[f'{scope}_return_worst']=min(vals) if vals else None
        out[f'{scope}_positive_return']=sum(v>0 for v in vals)
        out[f'{scope}_beats_long']=sum(r[scope]['excess_return']>EPS for r in rs)
    out['median_seconds_descriptive']=med([r['seconds'] for r in rs])
    return out


def analyze(rows, score_slack, axis_values=None):
    # Duplicate scientific cells would overweight comparisons; reject them.
    cells = Counter(cell(r) for r in rows)
    if any(n>1 for n in cells.values()): raise ValueError('Duplicate scientific cells; cannot perform unweighted matched comparisons')
    groups=defaultdict(list)
    for r in rows: groups[(r['cohort'],r['n_epochs'],r['architecture'])].append(r)
    overview=[dict(cohort=k[0],n_epochs=k[1],architecture=k[2],**summarize(v)) for k,v in sorted(groups.items())]
    pairs=[]; balanced=[]
    for (cohort,ne), rr in group(rows, ('cohort','n_epochs')).items():
        archs=sorted({r['architecture'] for r in rr})
        maps={a:{cell(r,('architecture',)):r for r in rr if r['architecture']==a} for a in archs}
        common=set.intersection(*(set(v) for v in maps.values()))
        for a in archs: balanced.append(dict(cohort=cohort,n_epochs=ne,architecture=a,**summarize([maps[a][k] for k in common])))
        for a,b in combinations(archs,2):
            keys=set(maps[a])&set(maps[b])
            for scope in SCOPES:
                for metric in METRICS:
                    ds=[maps[b][k][scope][metric]-maps[a][k][scope][metric] for k in keys]
                    pairs.append(dict(cohort=cohort,n_epochs=ne,A=a,B=b,scope=scope,metric=metric,
                                      only_A=len(set(maps[a])-keys),only_B=len(set(maps[b])-keys),**delta_stats(ds)))
    transitions=[]; transition_detail=[]
    by_name={r['name']:r for r in rows}
    for r in rows:
        p=by_name.get(r['source'])
        if p is None or cell(r,('stage',)) != cell(p,('stage',)): continue
        d=dict(name=r['name'],parent=p['name'],cohort=r['cohort'],architecture=r['architecture'],
               n_epochs=r['n_epochs'],stage=r['stage'])
        for s in SCOPES:
            for m in METRICS: d[s+'_'+m]=r[s][m]-p[s][m]
        d['train_up_val_down']=d['TRAIN_agent_return']>EPS and d['VAL_agent_return'] < -EPS
        transition_detail.append(d)
    for k,rr in group(transition_detail,('cohort','n_epochs','architecture','stage')).items():
        for s in SCOPES:
            for m in METRICS:
                transitions.append(dict(cohort=k[0],n_epochs=k[1],architecture=k[2],stage=k[3],scope=s,metric=m,
                                        train_up_val_down=sum(r['train_up_val_down'] for r in rr),**delta_stats([r[s+'_'+m] for r in rr])))
    best_stages=[]
    for key,rr in group(rows,('cohort','n_epochs','architecture','learning_rate','gamma','seed')).items():
        best=max(r['VAL']['balanced_score'] for r in rr)
        best_stages.append(dict(chain=dict(zip(('cohort','n_epochs','architecture','learning_rate','gamma','seed'),key)),
                                stages=sorted(r['stage'] for r in rr),
                                best_VAL_score_stages=[r['stage'] for r in rr if abs(r['VAL']['balanced_score']-best)<=EPS]))
    # Aggregates are descriptive. Matched effects below hold all other axes fixed.
    sensitivity=[]
    for axis in ('learning_rate','gamma'):
        for key,rr in group(rows,('cohort','n_epochs','architecture',axis)).items():
            sensitivity.append(dict(cohort=key[0],n_epochs=key[1],architecture=key[2],axis=axis,value=key[3],**summarize(rr)))
    adjacent=[]; neighborhoods=[]
    for axis in ('learning_rate','gamma'):
        for _,rr in group_cells(rows, (axis,)).items():
            rr=sorted(rr,key=lambda r:r[axis])
            ordered = (axis_values or {}).get(str(rr[0]['n_epochs']), {}).get(axis, sorted({r[axis] for r in rr}))
            ranks = {v:i for i,v in enumerate(ordered)}
            for left,right in zip(rr,rr[1:]):
                if ranks[right[axis]] - ranks[left[axis]] != 1: continue
                for s in SCOPES:
                    for m in METRICS:
                        adjacent.append(dict(axis=axis,cohort=right['cohort'],architecture=right['architecture'],n_epochs=right['n_epochs'],
                                             low=left[axis],high=right[axis],scope=s,metric=m,delta=right[s][m]-left[s][m]))
            if axis=='learning_rate':
                for i,r in enumerate(rr):
                    near=[x for x in rr if abs(ranks[x[axis]]-ranks[r[axis]])<=1]
                    neighborhoods.append(dict(name=r['name'],cohort=r['cohort'],n_epochs=r['n_epochs'],architecture=r['architecture'],
                       neighbor_count=len(near)-1,neighbor_names=[x['name'] for x in near if x is not r],
                       local_worst_score=min(x['VAL']['balanced_score'] for x in near),
                       local_median_score=med([x['VAL']['balanced_score'] for x in near]),
                       local_worst_return=min(x['VAL']['agent_return'] for x in near),
                       local_worst_dd=min(x['VAL']['agent_max_drawdown'] for x in near)))
    effects=[]
    axes=('axis','cohort','n_epochs','architecture','low','high','scope','metric')
    for k,rr in group(adjacent,axes).items(): effects.append(dict(zip(axes,k),**delta_stats([x['delta'] for x in rr])))
    candidates=[]; pareto=[]
    neighborhoods_by_name={r['name']:r for r in neighborhoods}
    for key,rr in group(rows,('cohort','n_epochs')).items():
        front=[r for r in rr if not any(
            q['VAL']['agent_return']>=r['VAL']['agent_return'] and q['VAL']['agent_max_drawdown']>=r['VAL']['agent_max_drawdown']
            and (q['VAL']['agent_return']>r['VAL']['agent_return'] or q['VAL']['agent_max_drawdown']>r['VAL']['agent_max_drawdown']) for q in rr)]
        pareto += [r['name'] for r in front]
        picks=defaultdict(list)
        for metric,label in [('agent_return','Top VAL return'),('balanced_score','Top VAL balanced score')]:
            for r in sorted(rr,key=lambda r:(-r['VAL'][metric],r['name']))[:5]: picks[r['name']].append(label)
        stable=[r for r in rr if neighborhoods_by_name[r['name']]['neighbor_count']==2]
        for r in sorted(stable,key=lambda r:(-neighborhoods_by_name[r['name']]['local_worst_score'],r['name']))[:5]:
            picks[r['name']].append('Top worst-case score across LR + both neighbors')
        best=max(r['VAL']['balanced_score'] for r in rr)
        cheap=[r for r in rr if r['VAL']['balanced_score']>=best-score_slack]
        for r in sorted(cheap,key=lambda r:(r['width_cost_proxy'],-r['VAL']['balanced_score'],r['name']))[:3]:
            picks[r['name']].append(f'Smaller width proxy within {score_slack:g} of best score')
        for name,reasons in picks.items():
            r=by_name[name];n=neighborhoods_by_name[name]
            candidates.append(dict(name=name,id=r['id'],cohort=key[0],n_epochs=key[1],reasons=reasons,pareto=name in pareto,
                                   VAL=r['VAL'],TRAIN=r['TRAIN'],neighborhood=n))
    behavior=[]
    for k,rr in group(rows,('cohort','n_epochs','architecture')).items():
        behavior.append(dict(cohort=k[0],n_epochs=k[1],architecture=k[2],n=len(rr),
          high_exposure_few_trades=sum(r['VAL']['market_exposure']>=.95 and r['VAL']['round_trips']<=10 for r in rr),
          low_exposure=sum(r['VAL']['market_exposure']<=.05 for r in rr),
          beats_long=sum(r['VAL']['excess_return']>EPS for r in rr),
          positive_return=sum(r['VAL']['agent_return']>0 for r in rr),
          median_exposure=med([r['VAL']['market_exposure'] for r in rr]),
          median_round_trips=med([r['VAL']['round_trips'] for r in rr]),
          median_profit_factor=med([r['VAL']['profit_factor'] for r in rr]),
          missing_profit_factor=sum(r['VAL']['profit_factor'] is None for r in rr)))
    return dict(overview=overview,common_cell_overview=balanced,architecture_pairs=pairs,transitions=transitions,
                transition_detail=transition_detail,chain_best_stages=best_stages,sensitivity=sensitivity,
                adjacent_effects=effects,lr_neighborhoods=neighborhoods,behavior=behavior,candidates=candidates,pareto_names=pareto)


def group(rows, keys):
    out=defaultdict(list)
    for r in rows: out[tuple(r[k] for k in keys)].append(r)
    return out


def group_cells(rows, omit):
    out=defaultdict(list)
    for r in rows: out[cell(r,omit)].append(r)
    return out


def esc(x): return html.escape(str(x))


def fmt(x):
    if x is None: return '—'
    if isinstance(x,float): return f'{x:.6g}'
    if isinstance(x,(dict,list,tuple)): return esc(canonical(x))
    return esc(x)


def pct(x): return '—' if x is None else f'{100*x:+.2f}%'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def stats_table(records, fields):
    return table(fields,[[r.get(k) for k in fields] for r in records])


def render(report):
    a=report['analysis'];rows=report['runs'];issues=report['issues'];meta=report['metadata']
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPO queue analysis</title>',
      '<style>body{font:15px system-ui;margin:32px auto;max-width:1500px;padding:0 20px;color:#182838;background:#f8fafc}h1,h2{color:#123d64}section{background:white;padding:22px;margin:20px 0;border:1px solid #dce3ea;border-radius:10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px;border-bottom:1px solid #e1e7ed;white-space:nowrap}th{background:#eaf1f8;position:sticky;top:0}.scroll{overflow:auto;max-height:650px}p{line-height:1.6}.warn{color:#a03a10}summary{cursor:pointer;font-weight:600;margin:14px 0}a{color:#1260a0}.note{color:#536576}td:first-child{max-width:650px}code{overflow-wrap:anywhere}</style>',
      '<h1>PPO queue analysis</h1><p>'+esc(meta['queue'])+' · snapshot '+esc(meta['snapshot_at'])+'</p>',
      '<p>'+ ' · '.join(f'<a href="#s{i}">{i}. {t}</a>' for i,t in enumerate(['Integrity','Architectures','Continuation','LR / gamma','Strategy behavior','Candidates'],1))+'</p>',
      '<p class="note">All returns/DD/exposure in formatted tables are percentages; raw metric and delta tables use fractions (0.01 = 1 percentage point). Higher / less negative DD is better. No statistical significance is claimed.</p>']
    def section(i,title,body): parts.append(f'<section id="s{i}"><h2>{i}. {title}</h2>'+body+'</section>')
    body=f'<p>Selected: <b>{meta["selected"]}</b>; eligible complete runs: <b>{len(rows)}</b>; excluded/missing: <b>{meta["selected"]-len(rows)}</b>. Cohorts: {len(set(r["cohort"] for r in rows))}. All comparisons stay within a cohort and n_epochs.</p>'
    body+='<p>A cohort fixes dataset hash, evaluation windows/policy, non-grid scientific configuration and saved device category. Seed, stage and the other grid axes are matched in pairwise comparisons. GPU Enabled is availability metadata, not independent proof of execution device. No model weights or market data are loaded.</p>'
    body+=stats_table(report['cohorts'],['cohort','device','n_epochs','runs','seeds','architectures','learning_rates','gammas','stages','cartesian_cells','missing_cells','early_stopped','git_commits'])
    if issues: body+='<p class="warn">Review the integrity details below before using the rankings. Warnings do not automatically exclude a run.</p>'
    body+='<details><summary>Integrity details and exclusions</summary>'+stats_table(issues,['level','name','reason'])+'</details>'
    body+='<p>TRAIN and VAL must use the final training-step checkpoint. Multiple completed final evaluations exclude the run as ambiguous. Missing/invalid ancestors exclude their descendants.</p>'
    section(1,'Is the grid complete and comparable?',body)
    fields=['cohort','n_epochs','architecture','n','chains','VAL_agent_return_median','VAL_return_q25','VAL_return_q75','VAL_agent_max_drawdown_median','VAL_balanced_score_median','VAL_beats_long']
    body='<p>Use the common-cell table first: each architecture is evaluated on the same set of LR × gamma × seed × stage cells. Medians across all stages are descriptive; stages in one chain are dependent.</p>'+stats_table(a['common_cell_overview'],fields)
    for key, rr in group(a['common_cell_overview'],('cohort','n_epochs')).items():
        usable=[x for x in rr if x['n']]
        if usable:
            top=max(usable,key=lambda x:x['VAL_balanced_score_median'])
            body+='<p><b>'+esc(str(key))+':</b> highest common-cell median VAL balanced score: <b>'+esc(top['architecture'])+'</b> ('+fmt(top['VAL_balanced_score_median'])+'). This is a descriptive leader; inspect matched deltas below.</p>'
        else: body+='<p class="warn">No cells shared by all architectures in '+esc(str(key))+'. Use pairwise results only.</p>'
    body+='<p>Pairwise results: B minus A. Better/worse count uses the selected metric independently. Positive median delta favors B; counts alone do not measure effect size.</p>'
    body+=stats_table([x for x in a['architecture_pairs'] if x['scope']=='VAL'],['cohort','n_epochs','A','B','metric','n','better','worse','tie','median','mean','only_A','only_B'])
    body+='<details><summary>All architecture summaries including wall times</summary>'+stats_table(a['overview'],fields+['median_seconds_descriptive'])+'<p>Wall time includes evaluation/waiting and hardware/concurrency differences. It is not a controlled speed benchmark.</p></details>'
    section(2,'Does a larger architecture help?',body)
    td=a['transition_detail'];conflict=sum(x['train_up_val_down'] for x in td)
    body=f'<p>Matched parent → child transitions: {len(td)}. TRAIN return improved while VAL return worsened in <b>{conflict}</b>. This is a warning sign, not proof of overfitting.</p>'
    body+=stats_table([x for x in a['transitions'] if x['scope']=='VAL'],['cohort','n_epochs','architecture','stage','metric','n','better','worse','tie','median','train_up_val_down'])
    best=Counter((x['chain']['cohort'],x['chain']['n_epochs'],x['chain']['architecture'],tuple(x['best_VAL_score_stages'])) for x in a['chain_best_stages'])
    body+='<p>Best observed VAL balanced-score stage per chain (ties retained; this is post-selection on VAL).</p>'+table(['Cohort','Epochs','Architecture','Best stages','Chains'],[[*k,v] for k,v in sorted(best.items())])
    body+='<details><summary>TRAIN and VAL transition metrics</summary>'+stats_table(a['transitions'],['cohort','n_epochs','architecture','stage','scope','metric','n','better','worse','tie','median'])+'</details>'
    section(3,'When do continuations help or hurt?',body)
    body='<p>Aggregates mix stages/gammas and are descriptive. Adjacent-value comparisons hold other grid axes fixed. LR neighbors are the next lower/higher tested values, not equal numeric steps. Boundary LRs have only one neighbor.</p>'
    body+='<p>For each LR/gamma inspect both median return and lower-quartile return: a high median with a weak lower quartile indicates sensitivity to the other settings or stage. Do not rank different cohorts against each other.</p>'
    body+=stats_table(a['sensitivity'],['cohort','n_epochs','architecture','axis','value','n','VAL_agent_return_median','VAL_balanced_score_median','VAL_agent_max_drawdown_median','VAL_return_q25','VAL_return_q75'])
    body+='<details><summary>Matched adjacent LR / gamma effects</summary>'+stats_table([x for x in a['adjacent_effects'] if x['scope']=='VAL'],['axis','cohort','n_epochs','architecture','low','high','metric','n','better','worse','tie','median'])+'</details>'
    near=sorted([x for x in a['lr_neighborhoods'] if x['neighbor_count']==2],key=lambda x:-x['local_worst_score'])
    body+='<p>Local LR robustness: rank by the worst VAL balanced score across the center and its two neighbors. This tests local sensitivity, not seed or out-of-sample robustness.</p>'+stats_table(near[:30],['name','local_worst_score','local_median_score','local_worst_return','local_worst_dd','neighbor_count'])
    section(4,'Are good results stable across LR and gamma?',body)
    body='<p>High-exposure / few-trades flag: VAL exposure ≥95% and round trips ≤10. Low-exposure flag: ≤5%. These are descriptive thresholds, not rejection rules. Excess return = agent minus always-long over the same evaluation window; it is not risk-adjusted alpha.</p>'
    body+=stats_table(a['behavior'],['cohort','n_epochs','architecture','n','high_exposure_few_trades','low_exposure','beats_long','positive_return','median_exposure','median_round_trips','median_profit_factor','missing_profit_factor'])
    body+='<p>Undefined profit factor stays missing; it is never replaced by zero. Win rate and PF alone do not determine quality. TRAIN/VAL have different durations, so their cumulative returns should not be directly subtracted as an overfitting score.</p>'
    section(5,'What kind of strategy did the agent learn?',body)
    section(6,'Candidate selection',
        '<p>The primary shortlist above uses the configured top percentage, exact cutoff ties, '
        'the VAL Pareto front and illustrative drawdown budgets. No fixed top-five shortlist is used. '
        'Additional seeds test sensitivity to training randomness on the same data; untouched '
        'walk-forward data are still required to assess performance on new periods.</p>')
    parts.append('<section><h2>All eligible runs</h2><p>Use browser Find to locate a run. Full local run table; the compact JSON contains detailed candidate chains and Pareto runs.</p>'+table(['Run','Stage','Device','TRAIN return','TRAIN DD','VAL return','VAL DD','VAL score','Exposure','Trips','PF','Win rate','Long return','Seconds'],[[r['name'],r['stage'],r['device'],pct(r['TRAIN']['agent_return']),pct(r['TRAIN']['agent_max_drawdown']),pct(r['VAL']['agent_return']),pct(r['VAL']['agent_max_drawdown']),r['VAL']['balanced_score'],pct(r['VAL']['market_exposure']),r['VAL']['round_trips'],r['VAL']['profit_factor'],pct(r['VAL']['win_rate']),pct(r['VAL']['always_long_return']),r['seconds']] for r in sorted(rows,key=lambda r:-r['VAL']['balanced_score'])])+'</section>')
    parts.append('<p>Report version '+VERSION+'. Read-only database snapshot; only this report and its JSON were written.</p></html>')
    return '\n'.join(parts)


def frontier(rows, scope):
    """Exact dominance in saved floating-point metrics; ties remain separate runs."""
    return [r for r in rows if not any(
        q[scope]['agent_return'] >= r[scope]['agent_return']
        and q[scope]['agent_max_drawdown'] >= r[scope]['agent_max_drawdown']
        and (q[scope]['agent_return'] > r[scope]['agent_return']
             or q[scope]['agent_max_drawdown'] > r[scope]['agent_max_drawdown'])
        for q in rows)]


def front_geometry(front, scope):
    # x = drawdown magnitude; y = return. Less x and more y are desirable.
    points = sorted({(-r[scope]['agent_max_drawdown'], r[scope]['agent_return']) for r in front})
    if not points:
        return {}
    x0,y0 = points[0]; x1,y1 = points[-1]
    dx,dy = x1-x0,y1-y0
    normalized = [((x-x0)/dx if dx else 0, (y-y0)/dy if dy else 0) for x,y in points]
    hull = []
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
    for i,p in enumerate(normalized):
        # Keep collinear supported points, discard only inward bends.
        while len(hull)>1 and cross(normalized[hull[-2]], normalized[hull[-1]],p)>1e-12:
            hull.pop()
        hull.append(i)
    supported = {points[i] for i in hull}
    out={}
    for r in front:
        x,y=-r[scope]['agent_max_drawdown'],r[scope]['agent_return']
        gap = y-(y0+(y1-y0)*(x-x0)/dx) if dx else None
        out[r['name']]=dict(supported=(x,y) in supported,
            chord_gap_return=gap,
            chord_side=('undefined' if gap is None else 'above' if gap>EPS else 'below' if gap < -EPS else 'on'),
            normalized_chord_distance=(gap/dy/math.sqrt(2) if gap is not None and dy else None))
    return out


def select_top_percent(rows, scope, metric, percent):
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError('Top percent must be greater than 0 and at most 100')
    if not rows:
        return [], dict(target_count=0, selected_count=0, cutoff=None)
    count = int((Decimal(str(percent)) * len(rows) / 100).to_integral_value(rounding='ROUND_CEILING'))
    ordered = sorted(rows, key=lambda r: (-r[scope][metric], r['name']))
    cutoff = ordered[count-1][scope][metric]
    # Exact ties in stored metrics; no rounding by displayed percentages.
    picked = [r for r in ordered if r[scope][metric] >= cutoff]
    return picked, dict(target_count=count, selected_count=len(picked), cutoff=cutoff)


def candidate_analysis(rows, top_percent=1.0):
    results=[]
    for (cohort,),rr in sorted(group(rows,('cohort',)).items()):
        fronts={}
        for scope in SCOPES:
            ff=frontier(rr,scope)
            geometry=front_geometry(ff,scope)
            fronts[scope]=[dict(name=r['name'],id=r['id'],n_epochs=r['n_epochs'],
                                **geometry[r['name']]) for r in sorted(ff,key=lambda r:-r[scope]['agent_return'])]
        reasons=defaultdict(list)
        for r in fronts['VAL']:
            reasons[r['name']].append('VAL Pareto')
            if r['supported']: reasons[r['name']].append('Supported VAL Pareto')
        ranking_selection = {}
        for metric,title in [('agent_return','VAL return'),('balanced_score','VAL balanced score')]:
            picked, selection = select_top_percent(rr, 'VAL', metric, top_percent)
            selection['names'] = [r['name'] for r in picked]
            ranking_selection[metric] = selection
            for r in picked:
                reasons[r['name']].append(f'Top {top_percent:g}% '+title)
        # Explicit risk budgets, no claim that these are the user's preferred limits.
        budgets=[]
        for cap in (.10,.20,.30,.40,.50):
            feasible=[r for r in rr if r['VAL']['agent_max_drawdown'] >= -cap]
            if not feasible:
                budgets.append(dict(max_dd_magnitude=cap,names=[])); continue
            best=max(r['VAL']['agent_return'] for r in feasible)
            winners=[r for r in feasible if r['VAL']['agent_return']==best]
            budgets.append(dict(max_dd_magnitude=cap,names=[r['name'] for r in winners]))
            for r in winners: reasons[r['name']].append(f'Best return with DD <= {cap:.0%}')
        results.append(dict(cohort=cohort,seed=rr[0]['seed'],runs=len(rr),fronts=fronts,top_percent=top_percent,ranking_selection=ranking_selection,
            front_composition={scope:dict(Counter(str(x['n_epochs']) for x in ff)) for scope,ff in fronts.items()},
            drawdown_budgets=budgets,
            shortlist=[dict(name=n,reasons=why) for n,why in sorted(reasons.items(),
                key=lambda item:-next(r['VAL']['agent_return'] for r in rr if r['name']==item[0]))]))
    return results


def candidate_html(report):
    rows={r['name']:r for r in report['runs']}
    def candidates_table(items,scope,geometry=False):
        headers=['Run','Epochs','Return','Max DD','Score','Exposure','Round trips','PF','Win rate','Excess vs long',
                 'TRAIN return','TRAIN DD','Near-long flag']
        if geometry: headers += ['Supported','Chord side','Above chord [pp]']
        else: headers+=['Selection reasons']
        data=[]
        for item in items:
            r=rows[item['name']]; v=r[scope]
            row=[r['name'],r['n_epochs'],pct(v['agent_return']),pct(v['agent_max_drawdown']),
                 v['balanced_score'],pct(v['market_exposure']),v['round_trips'],v['profit_factor'],
                 pct(v['win_rate']),pct(v['excess_return']),pct(r['TRAIN']['agent_return']),
                 pct(r['TRAIN']['agent_max_drawdown']),
                 v['market_exposure']>=.95 and v['round_trips']<=10]
            row += ([item['supported'],item['chord_side'],
                    None if item['chord_gap_return'] is None else 100*item['chord_gap_return']]
                    if geometry else ['; '.join(item['reasons'])])
            data.append(row)
        return table(headers,data)
    parts=['<section><h2>Candidate selection across n_epochs</h2>',
        '<p>Goal: find promising individual candidates, not rank parameter families by their averages. '
        'TRAIN and VAL fronts are calculated independently across all epochs and architectures within each comparable cohort. '
        'No ranking is pooled across cohorts. Identical metric coordinates retain every run.</p>',
        '<p>Near-long flag means exposure >=95% and <=10 round trips; it is descriptive, not proof of identical decisions '
        'or grounds for automatic rejection. Trajectories and weights are not loaded.</p>',
        '<p>Chord: line joining the lowest-DD and highest-return points; signed gap is return minus that line. '
        'Supported points lie on the favorable convex hull. Coordinates use linear metric scales; numeric hull tolerance is 1e-12 '
        'after range normalization. Interpolated DD is not a forecast for a combination of strategies.</p>']
    for c in report['candidate_selection']:
        parts.append('<h3>Cohort '+esc(c['cohort'])+f' — seed {c["seed"]} — {c["runs"]} eligible runs</h3>')
        parts.append(table(['Scope','Pareto counts by n_epochs'],[[s,c['front_composition'][s]] for s in SCOPES]))
        parts.append('<p>Top '+fmt(c['top_percent'])+'% per metric; ceil(percent × eligible cohort size / 100). '
                     'Exact cutoff ties are retained. The combined pool contains each run once.</p>')
        parts.append(table(['Metric','Target count','Count including ties','Cutoff (stored units)'],
                     [[metric,info['target_count'],info['selected_count'],info['cutoff']]
                      for metric,info in c['ranking_selection'].items()]))
        parts.append('<details><summary>Combined VAL candidate pool ('+str(len(c['shortlist']))+
                     ' runs)</summary>'+candidates_table(c['shortlist'],'VAL')+'</details>')
        for scope in SCOPES:
            parts.append('<h3>'+scope+' Pareto front</h3>'+candidates_table(c['fronts'][scope],scope,True))
        parts.append('<details><summary>Best return at illustrative VAL drawdown limits</summary>'+
            table(['Maximum DD magnitude','Winners'],[[pct(b['max_dd_magnitude']),b['names']] for b in c['drawdown_budgets']])+'</details>')
    parts.append('<p>Shortlist is a screening set, not a final recommendation. Freeze selected configurations, test additional seeds, '
        'then the untouched walk-forward periods. Neighboring LR results are supplementary and do not automatically exclude an isolated winner. '
        'These metrics do not establish behavioral diversity or statistical superiority.</p></section>')
    return '\n'.join(parts)


def clean_superseded(parent, current, report):
    """Delete only recognized two-file reports whose eligible run names are covered."""
    current_names={r['name'] for r in report['runs']}
    removed=[]; retained=[]
    for directory in sorted(parent.iterdir()):
        if directory==current or not directory.is_dir() or directory.is_symlink(): continue
        if not re.fullmatch(r'\d{8}T\d{12}Z',directory.name): continue
        files=list(directory.iterdir())
        if {p.name for p in files}!={'report.json','report.html'} or any(p.is_symlink() or not p.is_file() for p in files):
            retained.append(str(directory)); continue
        try:
            old=json.loads((directory/'report.json').read_text())
            if old.get('seed_comparison'):
                raise ValueError('Retain seed comparison reports as selection/evidence history')
            if old['metadata']['version'] not in {'1.0','1.0.0','1.0.1','2.0.0','2.1.0','2.2.0'}:
                raise ValueError('Unknown report version')
            names=set(old.get('eligible_run_names') or [r['name'] for r in old['runs']])
            if not names or not names <= current_names:
                raise ValueError('Old report has runs not covered by new eligible results')
            if 'analysis' not in old or 'cohorts' not in old: raise ValueError('Unknown report format')
        except (ValueError,KeyError,TypeError):
            retained.append(str(directory));continue
        for p in files: p.unlink()
        directory.rmdir();removed.append(str(directory))
    for p in removed: print('Removed superseded grid report:',p)
    for p in retained: print('Retained (not safely superseded):',p)
    return removed,retained


def compact_seed_comparison(comparison):
    if comparison is None:
        return None
    result = dict(comparison)
    matched_names = {p[side]['name'] for p in comparison['pairs']
                     for side in ('reference', 'replay')}
    result['chains'] = []
    unmatched = {}
    for original in comparison['chains']:
        chain = {k: v for k, v in original.items()
                 if k not in ('reference_path', 'replay_path')}
        for side in ('reference', 'replay'):
            path = original[side + '_path']
            chain[side + '_run_names'] = [r['name'] for r in path]
            for r in path:
                if r['name'] not in matched_names:
                    unmatched[r['name']] = r
        result['chains'].append(chain)
    result['unmatched_path_details'] = list(unmatched.values())
    result['path_detail_lookup'] = (
        'Matched full run metrics are in pairs.reference/pairs.replay; '
        'remaining path details are in unmatched_path_details.')
    return result


def compact_export(report):
    rows=report['runs']; by_name={r['name']:r for r in rows}
    selected={x['name'] for c in report['candidate_selection'] for x in c['shortlist']}
    fronts={x['name'] for c in report['candidate_selection'] for scope in SCOPES for x in c['fronts'][scope]}
    def root_name(r):
        seen=set()
        while r['source']:
            if r['name'] in seen: raise ValueError('Cycle in eligible run ancestry')
            seen.add(r['name']);r=by_name[r['source']]
        return r['name']
    roots={root_name(by_name[n]) for n in selected}
    chain_rows=[r for r in rows if root_name(r) in roots]
    details_names={r['name'] for r in chain_rows}|fronts
    details=[]
    for r in rows:
        if r['name'] not in details_names:continue
        d={k:v for k,v in r.items() if k not in SCOPES}
        d['roles']=(['candidate'] if r['name'] in selected else [])+(['candidate_chain'] if root_name(r) in roots else [])+(['pareto'] if r['name'] in fronts else [])
        for scope in SCOPES:
            d[scope]={k:v for k,v in r[scope].items() if k!='signature'}
        details.append(d)
    chains=[]
    for root in sorted(roots):
        chain=sorted([r for r in chain_rows if root_name(r)==root],key=lambda r:(r['stage'],r['name']))
        chains.append(dict(root=root,run_names=[r['name'] for r in chain],
                           candidate_names=[r['name'] for r in chain if r['name'] in selected]))
    tails=[]
    for keys in [('cohort',),('cohort','n_epochs','architecture'),
                 ('cohort','n_epochs','architecture','learning_rate'),
                 ('cohort','n_epochs','architecture','gamma')]:
        for key,rr in sorted(group(rows,keys).items()):
            d=dict(group=dict(zip(keys,key)),n=len(rr),metrics={})
            for scope in SCOPES:
                for metric in METRICS:
                    ordered=sorted(rr,key=lambda r:(-r[scope][metric],r['name']))
                    vals=[r[scope][metric] for r in rr]
                    d['metrics'][scope+'/'+metric]=dict(
                        median=med(vals),p90=quantile(vals,.90),p95=quantile(vals,.95),
                        p99=quantile(vals,.99),maximum=max(vals),
                        leaders=[dict(name=r['name'],value=r[scope][metric]) for r in ordered[:3]])
            tails.append(d)
    profiles=defaultdict(list)
    for r in rows:
        profiles[(r['cohort'],canonical({k:r['VAL'][k] for k in METRICS+EXTRA}))].append(r['name'])
    identical=[dict(names=ns,count=len(ns)) for ns in profiles.values()
               if len(ns)>1 and set(ns)&selected]
    a=report['analysis']
    cohort_conditions=[]
    for (cohort,),rr in group(rows,('cohort',)).items():
        cohort_conditions.append(dict(cohort=cohort,evaluation_signatures={s:rr[0][s]['signature'] for s in SCOPES}))
    return dict(metadata=dict(report['metadata'],export='compact-v1',
                full_run_count=len(rows),detailed_run_count=len(details),
                note='All calculations performed locally. HTML retains all eligible runs. JSON keeps candidate chains, both Pareto fronts and group upper tails. Quantiles are descriptive, not confidence intervals.'),
        cohorts=report['cohorts'],cohort_conditions=cohort_conditions,issues=report['issues'],
        seed_comparison=compact_seed_comparison(report.get('seed_comparison')),
        inventory_summary=dict(status_counts=dict(Counter(r['status'] for r in report['inventory']))),
        eligible_run_names=sorted(by_name),candidate_selection=report['candidate_selection'],
        candidate_chains=chains,runs=details,group_upper_tails=tails,
        identical_VAL_metric_profiles=identical,
        analysis=dict(overview=a['overview'],behavior=a['behavior'],
            candidate_lr_neighborhoods=[n for n in a['lr_neighborhoods'] if n['name'] in selected],
            candidate_chain_transitions=[x for x in a['transition_detail'] if x['name'] in details_names]),
        limitations=['Identical aggregate metrics do not prove identical trajectories.',
                     'Candidate-chain details include all currently available descendants of the same fresh root.',
                     'Full per-run inventory and supplementary tables remain in HTML.'])


def comparison_key(row, include_stage=True):
    keys = ('seed_comparison_cohort', 'architecture', 'learning_rate', 'gamma', 'n_epochs')
    return tuple(row[k] for k in keys) + ((row['stage'],) if include_stage else ())


def brief_run(row):
    return dict(id=row['id'], name=row['name'], stage=row['stage'],
                seed=row['seed'], source=row['source'],
                source_checkpoint_id=row.get('source_checkpoint_id'),
                steps_completed=row['steps_completed'], stopped_early=row['stopped_early'],
                **{scope: {k: row[scope].get(k) for k in METRICS + EXTRA + ('excess_return',)}
                   for scope in SCOPES})


def compare_seeds(rows, configs, expected, base_seed, replay_seed,
                  top_percent=1.0, selection_report=None):
    """Match saved scientific conditions and ancestry; never infer identity from run names."""
    base = [r for r in rows if r['seed'] == base_seed]
    replay_all = [r for r in rows if r['seed'] == replay_seed]
    replay = [r for r in replay_all if not expected or r['name'] in expected]
    if not base:
        raise ValueError('No eligible reference seed runs')
    issues = []
    base_names = {r['name']: r for r in base}
    replay_names = {r['name']: r for r in replay}
    expected_names = set(expected)
    for name in sorted(expected_names - set(replay_names)):
        issues.append(dict(name=name, reason='Expected replay missing or excluded by integrity checks'))
    for r in replay:
        if r['name'] in expected:
            actual, planned = configs[r['name']], expected[r['name']]
            if (config_axes(actual) != config_axes(planned)
                    or fixed_config(actual) != fixed_config(planned)
                    or actual['continuation'] != planned['continuation']):
                issues.append(dict(name=r['name'], reason='Replay differs from its expected queue configuration'))
    bad_names = {i['name'] for i in issues}
    bm, tm = defaultdict(list), defaultdict(list)
    for r in base:
        bm[comparison_key(r)].append(r)
    for r in replay:
        tm[comparison_key(r)].append(r)
    matched = {}
    for key, targets in sorted(tm.items()):
        refs = bm.get(key, [])
        if len(refs) != 1 or len(targets) != 1:
            for target in targets:
                issues.append(dict(name=target['name'], reason=
                    f'Ambiguous or absent scientific match: {len(refs)} reference / {len(targets)} replay runs'))
            continue
        a, b = refs[0], targets[0]
        if b['name'] in bad_names:
            continue
        if a.get('continuation_options') != b.get('continuation_options'):
            issues.append(dict(name=b['name'], reason='Continuation options differ'))
            continue
        if any(abs(a[s]['always_long_return'] - b[s]['always_long_return']) > EPS for s in SCOPES):
            issues.append(dict(name=b['name'], reason='Always-long baseline differs'))
            continue
        matched[b['name']] = (a, b)
    # A resume must use the matched replay parent, not a reference-seed checkpoint.
    while True:
        invalid = []
        for name, (a, b) in matched.items():
            if bool(a['source']) != bool(b['source']):
                invalid.append(name)
            elif b['source']:
                parent = matched.get(b['source'])
                if parent is None or parent[0]['name'] != a['source']:
                    invalid.append(name)
        if not invalid:
            break
        for name in invalid:
            issues.append(dict(name=name, reason='Replay parent is not the matching reference parent'))
            del matched[name]

    selection_source = dict(mode='recomputed from eligible reference seed only',
                            top_percent=top_percent,
                            note='Not an immutable historical selection; use --selection-report to freeze prior names.')
    if selection_report:
        path = selection_report.expanduser().resolve()
        saved = json.loads(path.read_text())
        selected_names = {x['name'] for c in saved['candidate_selection'] for x in c['shortlist']}
        current_names = {r['name'] for r in rows}
        missing = selected_names - current_names
        if missing:
            raise ValueError('Historical candidates missing/ineligible: ' + ', '.join(sorted(missing)))
        candidates = selected_names & set(base_names)
        if not candidates:
            raise ValueError('Selection report contains no eligible reference-seed candidates')
        selection_source = dict(mode='frozen candidate names from report', path=str(path),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        candidates = {x['name'] for c in candidate_analysis(base, top_percent) for x in c['shortlist']}
    pairs = []
    for a, b in sorted(matched.values(), key=lambda ab: comparison_key(ab[0])):
        changes = {s: {m: b[s][m] - a[s][m] for m in METRICS} for s in SCOPES}
        pairs.append(dict(
            reference=brief_run(a), replay=brief_run(b),
            reference_candidate=a['name'] in candidates, deltas=changes,
            different_realized_training_length=a['steps_completed'] != b['steps_completed'],
            worst_VAL_return=min(a['VAL']['agent_return'], b['VAL']['agent_return']),
            worst_VAL_score=min(a['VAL']['balanced_score'], b['VAL']['balanced_score']),
            worst_VAL_dd=min(a['VAL']['agent_max_drawdown'], b['VAL']['agent_max_drawdown']),
            positive_VAL_return_both=all(r['VAL']['agent_return'] > EPS for r in (a, b)),
            beats_long_both=all(r['VAL']['excess_return'] > EPS for r in (a, b)),
            no_VAL_trades_either=any(r['VAL']['round_trips'] == 0 for r in (a, b)),
            near_long_either=any(r['VAL']['market_exposure'] >= .95
                                and r['VAL']['round_trips'] <= 10 for r in (a, b))))
    matched_candidates = {p['reference']['name'] for p in pairs if p['reference_candidate']}
    aggregates = []
    for label, subset in [('All matched stages', pairs),
                          ('Original candidates at the same stage',
                           [p for p in pairs if p['reference_candidate']])]:
        for scope in SCOPES:
            for metric in METRICS:
                aggregates.append(dict(population=label, scope=scope, metric=metric,
                    **delta_stats([p['deltas'][scope][metric] for p in subset])))

    chains = []
    replay_groups = group(replay, ('seed_comparison_cohort', 'architecture',
                                  'learning_rate', 'gamma', 'n_epochs'))
    base_groups = group(base, ('seed_comparison_cohort', 'architecture',
                              'learning_rate', 'gamma', 'n_epochs'))
    for key, rr in sorted(replay_groups.items()):
        aa = sorted(base_groups.get(key, []), key=lambda r: r['stage'])
        rr = sorted(rr, key=lambda r: r['stage'])
        names = {r['name'] for r in rr}
        pp = [p for p in pairs if p['replay']['name'] in names]
        base_stages = [r['stage'] for r in aa]
        target_stages = [r['stage'] for r in rr]
        complete = (bool(aa) and base_stages == list(range(max(base_stages) + 1))
                    and base_stages == target_stages and len(pp) == len(aa))
        transitions = []
        by_stage = {p['reference']['stage']: p for p in pp}
        for stage, p in sorted(by_stage.items()):
            prev = by_stage.get(stage - 1)
            if prev is None:
                continue
            transitions.append(dict(to_stage=stage,
                **{side: {scope: {metric: p[side][scope][metric] - prev[side][scope][metric]
                                 for metric in METRICS} for scope in SCOPES}
                   for side in ('reference', 'replay')}))
        best = {}
        for metric in ('agent_return', 'balanced_score'):
            if not aa:
                continue
            value = max(r['VAL'][metric] for r in aa)
            chosen = [r for r in aa if r['VAL'][metric] == value]
            # Preserve ties rather than select a tie using replay performance.
            locked = [p for p in pp if p['reference']['name'] in {r['name'] for r in chosen}]
            best[metric] = dict(
                reference_best_stages=[r['stage'] for r in chosen],
                reference_best_value=value,
                replay_at_reference_best=[dict(stage=p['replay']['stage'],
                                               value=p['replay']['VAL'][metric]) for p in locked],
                replay_best_stages=[r['stage'] for r in rr
                                   if r['VAL'][metric] == max(x['VAL'][metric] for x in rr)],
                replay_best_value=max(r['VAL'][metric] for r in rr),
                replay_best_is_post_selection=True)
        chains.append(dict(
            conditions=dict(zip(('comparison_cohort', 'architecture', 'learning_rate', 'gamma', 'n_epochs'), key)),
            reference_root=aa[0]['name'] if aa and aa[0]['stage'] == 0 else None,
            replay_root=rr[0]['name'] if rr and rr[0]['stage'] == 0 else None,
            reference_stages=base_stages, replay_stages=target_stages,
            complete_relative_to_reference=complete,
            reference_candidate_names=[r['name'] for r in aa if r['name'] in candidates],
            reference_path=[brief_run(r) for r in aa],
            replay_path=[brief_run(r) for r in rr],
            best_stage=best, transitions=transitions,
            strictly_increasing_VAL_return={
                side: (all(t[side]['VAL']['agent_return'] > EPS for t in transitions)
                       if complete and transitions else None)
                for side in ('reference', 'replay')}))
    # Keep ranking groups separate when datasets or scientific conditions differ.
    candidate_ranks = []
    for (cohort,), rr in group(base, ('seed_comparison_cohort',)).items():
        names = {r['name'] for r in rr}
        eligible_pairs = [p for p in pairs if p['reference_candidate'] and p['reference']['name'] in names]
        candidate_ranks.append(dict(comparison_cohort=cohort,
            by_worst_VAL_return=[p['reference']['name'] for p in sorted(
                eligible_pairs, key=lambda p: (-p['worst_VAL_return'], p['reference']['name']))],
            by_worst_VAL_score=[p['reference']['name'] for p in sorted(
                eligible_pairs, key=lambda p: (-p['worst_VAL_score'], p['reference']['name']))]))
    readiness = ('INCOMPLETE' if issues or any(not c['complete_relative_to_reference'] for c in chains)
                 else 'PASS' if expected else 'OBSERVED RUNS ONLY')
    return dict(base_seed=base_seed, replay_seed=replay_seed, readiness=readiness,
        counts=dict(reference_runs=len(base), observed_replay_runs=len(replay),
                    expected_replay_runs=len(expected) if expected else None,
                    matched_stages=len(pairs), observed_chains=len(chains),
                    complete_chains=sum(c['complete_relative_to_reference'] for c in chains),
                    original_candidates=len(candidates), matched_candidates=len(matched_candidates)),
        selection=selection_source, issues=issues,
        candidates_without_match=sorted(candidates - matched_candidates),
        metric_changes=aggregates, candidates_ranked=candidate_ranks,
        pairs=pairs, chains=chains,
        limitations=[
            'Seed2 paths were selected using seed1; they are not an unbiased sample of the full grid.',
            'Two seeds do not establish statistical robustness or out-of-sample performance.',
            'Stages in one chain are dependent. Better/worse counts are descriptive.',
            'Worst-of-two rankings use both seeds and are exploratory, not a new held-out test.',
            'Best replay stage uses replay outcomes; compare locked reference stages separately.',
            'This compares saved final metrics; it does not compare tensor/trajectory identity.',
            'Without expected queues, completely absent replay paths cannot be detected.'])


def seed_comparison_html(c):
    parts = ['<section id="seed-comparison"><h2>Seed comparison: ' +
             esc(c['base_seed']) + ' → ' + esc(c['replay_seed']) + '</h2>',
             '<p>Readiness: <b>' + esc(c['readiness']) + '</b></p>',
             table(['Measurement', 'Count'], list(c['counts'].items())),
             '<p>Selection: ' + esc(canonical(c['selection'])) + '</p>',
             '<p>Delta = replay minus reference, in stored units. Positive return/score '
             'or less-negative DD counts as better. Exposure and trading activity are descriptive.</p>',
             stats_table(c['metric_changes'],
                         ['population', 'scope', 'metric', 'n', 'better', 'worse', 'tie', 'median', 'mean'])]
    if c['issues']:
        parts += ['<h3>Unmatched or invalid comparisons</h3>', stats_table(c['issues'], ['name', 'reason'])]
    parts += ['<h3>Original candidates at their original stage</h3>',
              '<p>Sorted by the worse VAL return across the two seeds within each comparable group. '
              'This exploratory ranking uses both seeds. Full paths appear below.</p>']
    lookup = {p['reference']['name']: p for p in c['pairs']}
    for ranking in c['candidates_ranked']:
        values = []
        for name in ranking['by_worst_VAL_return']:
            p = lookup[name]; a, b = p['reference'], p['replay']
            values.append([name, b['name'], pct(a['VAL']['agent_return']), pct(b['VAL']['agent_return']),
                           pct(a['VAL']['agent_max_drawdown']), pct(b['VAL']['agent_max_drawdown']),
                           a['VAL']['balanced_score'], b['VAL']['balanced_score'],
                           pct(a['VAL']['market_exposure']), pct(b['VAL']['market_exposure']),
                           a['VAL']['round_trips'], b['VAL']['round_trips'],
                           p['near_long_either'], p['different_realized_training_length']])
        parts += ['<h4>Comparison cohort ' + esc(ranking['comparison_cohort']) + '</h4>',
                  table(['Reference', 'Replay', 'Return A', 'Return B', 'DD A', 'DD B',
                         'Score A', 'Score B', 'Exposure A', 'Exposure B',
                         'Trips A', 'Trips B', 'Near-long either', 'Training length differs'], values)]
    if c['candidates_without_match']:
        parts += ['<details><summary>Original candidates without a valid replay</summary>',
                  table(['Reference name'], [[n] for n in c['candidates_without_match']]), '</details>']
    for chain in c['chains']:
        parts += ['<details><summary>' + esc(chain['reference_root'] or chain['conditions']) +
                  ' — complete: ' + esc(chain['complete_relative_to_reference']) + '</summary>',
                  '<p>Strictly increasing VAL return: ' + esc(canonical(chain['strictly_increasing_VAL_return'])) + '</p>',
                  '<p>Best-stage comparison (all exact ties retained):</p>',
                  stats_table([dict(metric=k, **v) for k, v in chain['best_stage'].items()],
                    ['metric', 'reference_best_stages', 'reference_best_value',
                     'replay_at_reference_best', 'replay_best_stages', 'replay_best_value'])]
        data = []
        for side in ('reference', 'replay'):
            for r in chain[side + '_path']:
                for scope in SCOPES:
                    v = r[scope]
                    data.append([side, r['stage'], r['name'], scope, pct(v['agent_return']),
                                 pct(v['agent_max_drawdown']), v['balanced_score'],
                                 pct(v['market_exposure']), v['round_trips'],
                                 v['profit_factor'], pct(v['win_rate']), pct(v['always_long_return'])])
        parts += [table(['Seed role', 'Stage', 'Run', 'Scope', 'Return', 'DD', 'Score',
                         'Exposure', 'Trips', 'PF', 'Win', 'Always long'], data), '</details>']
    parts += ['<p>' + '</p><p>'.join(esc(x) for x in c['limitations']) + '</p></section>']
    return '\n'.join(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    scope=parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--all-runs',action='store_true',help='Inventory every DB run; analyze eligible stage-one runs')
    scope.add_argument('--queue',action='append',help='Queue name/path; may be repeated to combine queues')
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/queue_analysis'))
    parser.add_argument('--top-percent',type=float,default=1.0,help='Top percentage per metric/cohort; rounded up, including cutoff ties (default: 1)')
    parser.add_argument('--score-slack',type=float,default=.02)
    parser.add_argument('--clean-old-reports',action='store_true',
                        help='After successful save, remove recognized older two-file grid reports covered by this report')
    parser.add_argument('--compare-seeds', nargs=2, type=int, metavar=('BASE', 'REPLAY'),
                        help='Compare matching full paths, e.g. --compare-seeds 1 2')
    parser.add_argument('--comparison-queue', action='append', default=[],
                        help='Expected replay queue name/path; repeat for multiple queues')
    parser.add_argument('--selection-report', type=Path,
                        help='Optional previous JSON report to freeze the original candidate names')
    args=parser.parse_args()
    if args.compare_seeds and args.compare_seeds[0] == args.compare_seeds[1]:
        raise ValueError('Comparison requires two different seeds')
    if (args.comparison_queue or args.selection_report) and not args.compare_seeds:
        raise ValueError('Comparison options require --compare-seeds')

    if not math.isfinite(args.top_percent) or not 0<args.top_percent<=100 or not math.isfinite(args.score_slack) or args.score_slack<0: raise ValueError('Invalid options')
    root=Path.cwd().resolve()
    if not (root/'run_search_queue.py').is_file():raise ValueError('Run from PPO_AGENT repository root')
    sys.path.insert(0,str(root))
    expected_configs = {}
    comparison_sources = []
    for queue in args.comparison_queue:
        _, cc, pr = load_queue(root, queue)
        comparison_sources.append(pr)
        for name, c in cc.items():
            if int(c['run']['seed']) != args.compare_seeds[1]:
                raise ValueError('Comparison queue contains a different replay seed: ' + name)
            if name in expected_configs and canonical(expected_configs[name]) != canonical(c):
                raise ValueError('Conflicting comparison queue config: ' + name)
            expected_configs[name] = c
    configs={}; provenance={}; preissues=[]; excluded_inventory=[]
    if args.all_runs:
        print('Reading ALL runs in one consistent read-only DB snapshot...',flush=True)
        runs,cps,evs,counts,stamp=snapshot(None)
        provenance={'source':'saved normalized DB configurations; no queue membership restriction'}
        for r in runs:
            try:
                c=r['normalized_config_json']
                if isinstance(c,str):c=json.loads(c)
                config_axes(c);fixed_config(c)
                if c['run']['name']!=r['name']: raise ValueError('Saved config name differs from DB name')
                if r.get('cycle_id') is not None:raise ValueError('Walk-forward run: inventoried, not pooled with stage-one results')
                if c['continuation']['mode'] not in ('fresh','resume'):raise ValueError('Unsupported continuation mode')
                configs[r['name']]=c
            except (ValueError,KeyError,TypeError,AttributeError) as e:
                preissues.append(dict(name=r['name'],level='EXCLUDE',reason=str(e)))
                excluded_inventory.append(dict(id=r['id'],name=r['name'],status=r['status']))
        label='ALL DATABASE RUNS'
        selected=len(runs)
    else:
        sources=[]
        for queue in args.queue:
            q,cc,pr=load_queue(root,queue);sources.append(pr)
            for name,c in cc.items():
                if name in configs and canonical(configs[name])!=canonical(c):raise ValueError('Conflicting queue configs: '+name)
                configs[name]=c
        provenance={'queues':sources};label=' + '.join(args.queue);selected=len(configs)
        print('Reading consistent read-only DB snapshot...',flush=True)
        runs,cps,evs,counts,stamp=snapshot(list(configs))
    print('Checking final evaluations, ancestry and checkpoint metadata...',flush=True)
    rows,inventory,issues=prepare(root,configs,runs,cps,evs,counts)
    issues=preissues+issues;inventory=excluded_inventory+inventory
    if not rows:raise ValueError('No eligible completed runs: '+canonical(issues[:10]))
    # Avoid a pooled ranking when even the deterministic baseline is inconsistent.
    for (cohort,),rr in group(rows,('cohort',)).items():
        for scope_name in SCOPES:
            vals=[r[scope_name]['always_long_return'] for r in rr]
            if max(vals)-min(vals)>EPS:
                raise ValueError(scope_name+' always-long baseline differs within cohort '+cohort)
    cohorts=[]
    for k,rr in group(rows,('cohort','n_epochs')).items():
        axes=['architecture','learning_rate','gamma','seed','stage']
        values=[sorted({r[a] for r in rr}) for a in axes]
        expected=math.prod(len(v) for v in values)
        cohorts.append(dict(cohort=k[0],n_epochs=k[1],device=rr[0]['device'],runs=len(rr),
            architectures=values[0],learning_rates=values[1],gammas=values[2],seeds=values[3],stages=values[4],
            cartesian_cells=expected,missing_cells=expected-len(rr),early_stopped=sum(bool(r['stopped_early']) for r in rr),
            git_commits=sorted({r['git_commit'] or 'unknown' for r in rr})))
    # Candidate fronts retain duplicate configs; aggregate comparisons need one observation per cell.
    cells=Counter(cell(r) for r in rows)
    aggregate_rows=[r for r in rows if cells[cell(r)]==1]
    for r in rows:
        if cells[cell(r)]>1:issues.append(dict(name=r['name'],level='WARNING',
            reason='Repeated scientific cell: retained in rankings; excluded from supplementary matched aggregates'))
    analysis=analyze(aggregate_rows,args.score_slack) if aggregate_rows else dict(
        overview=[],common_cell_overview=[],architecture_pairs=[],transitions=[],transition_detail=[],
        chain_best_stages=[],sensitivity=[],adjacent_effects=[],lr_neighborhoods=[],behavior=[],candidates=[],pareto_names=[])
    print('Computing combined-epoch TRAIN/VAL Pareto fronts and shortlist...',flush=True)
    report=dict(metadata=dict(version=VERSION,queue=label,snapshot_at=stamp,selected=selected,score_slack=args.score_slack,top_percent=args.top_percent,
        ranking_scope='within cohort, across n_epochs and architectures',**provenance),
        cohorts=cohorts,issues=issues,inventory=inventory,runs=rows,analysis=analysis,
        candidate_selection=candidate_analysis(rows,args.top_percent))
    if args.compare_seeds:
        report['seed_comparison'] = compare_seeds(
            rows, configs, expected_configs, *args.compare_seeds,
            top_percent=args.top_percent, selection_report=args.selection_report)
        report['seed_comparison']['queue_provenance'] = comparison_sources
    legacy=render(report)
    if args.compare_seeds:
        legacy = legacy.replace('<section id="s1">',
                                seed_comparison_html(report['seed_comparison']) +
                                '<section id="s1">', 1)
    marker='<section id="s1">'
    legacy=legacy.replace(marker,candidate_html(report)+
        '<details><summary>Supplementary integrity checks and descriptive group analysis</summary>'+marker,1)
    legacy=legacy.replace('<section><h2>All eligible runs</h2>','</details><section><h2>All eligible runs</h2>',1)
    legacy=legacy.replace('All comparisons stay within a cohort and n_epochs.',
        'Primary rankings pool epochs within a cohort. Supplementary matched comparisons are separated by n_epochs.')
    parent=args.output_dir.expanduser()
    if not parent.is_absolute():parent=root/parent
    if parent.is_symlink():raise ValueError('Output directory must not be a symlink')
    parent=parent.resolve()
    if args.clean_old_reports and parent!=(root/'artifacts/queue_analysis').resolve():
        raise ValueError('Cleanup restricted to artifacts/queue_analysis')
    parent.mkdir(parents=True,exist_ok=True)
    out=parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir()
    for filename,content in [('report.json',json.dumps(compact_export(report),default=serial,indent=2,allow_nan=False)+'\n'),('report.html',legacy)]:
        with (out/filename).open('x',encoding='utf-8') as f:
            f.write(content);f.flush();os.fsync(f.fileno())
    print(f'Inventoried: {selected}; eligible: {len(rows)}; issues: {len(issues)}')
    for c in report['candidate_selection']:
        print(f"Cohort {c['cohort']}: {c['runs']} runs; VAL Pareto={len(c['fronts']['VAL'])}; "
              f"TRAIN Pareto={len(c['fronts']['TRAIN'])}; shortlist={len(c['shortlist'])}")
        print('Pareto by n_epochs:',c['front_composition'])
        for metric,selection in c['ranking_selection'].items():
            print(f"Top {c['top_percent']:g}% {metric}: target={selection['target_count']}; "
                  f"including ties={selection['selected_count']}")
    if args.compare_seeds:
        comparison = report['seed_comparison']
        print('Seed comparison:', canonical(comparison['counts']))
        print('Comparison readiness:', comparison['readiness'])
    print('Full local HTML report:',out/'report.html');print('Compact JSON for sharing:',out/'report.json')
    if args.clean_old_reports:
        clean_superseded(parent,out,report)
    print('Database, configs, checkpoints, CPU replay reports and hardware unchanged.')


if __name__=='__main__':
    try:main()
    except Exception as e:raise SystemExit(f'STOP: {type(e).__name__}: {e}')
