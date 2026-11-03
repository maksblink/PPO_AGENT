#!/usr/bin/env python3
"""Read-only stage-one training diagnostics. Run from the PPO_AGENT root."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

# Load the adjacent analysis module without importing training or PyTorch.
_spec = importlib.util.spec_from_file_location('training_grid_helpers', Path(__file__).with_name('ppo_analyze_grid.py'))
grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grid)
VERSION = '1.0.0'
IDENTITY = {'id', 'run_id', 'run_step', 'model_step', 'rollout_number', 'created_at'}
OPTIONAL = {'ep_reward', 'ep_len'}
KEYS = ('pre_nmse', 'post_nmse', 'nmse_improvement', 'explained_variance',
        'post_train_explained_variance', 'ev_improvement', 'post_bias_scaled',
        'post_prediction_scale', 'post_train_value_target_prediction_corr',
        'policy_entropy', 'approx_kl', 'clip_fraction', 'value_loss',
        'rollout_reward_mean')
NOTES = [
    'All statistics use every stored rollout. Charts use equal index bins: median, minimum and maximum; raw endpoints are retained.',
    'Pre-update predictions were collected during rollout; post-update predictions use the same observations and fixed rollout targets.',
    'Targets are bootstrapped GAE returns, not independently observed future profit. Post-update fit is in-sample, not critic validation.',
    'MSE = error_std squared + error_mean squared (population moments). NMSE = MSE / target_std squared.',
    'Positive signed error means target exceeds prediction (underestimation). EV ignores constant bias; NMSE includes it.',
    'Normalized metrics are undefined when target_std <= --min-target-std. No epsilon is added to denominators.',
    'Tail statistics use the final configured fraction of chronological rollout records, not the best window.',
    'Episode fields summarize completed-episode buffers and may retain previous episodes after resume. NULL is not zero.',
    'Policy entropy is minus entropy_loss. value_loss and loss are optimizer logs, not post-update fixed-target MSE.',
    'Correlations with final evaluation outcomes are descriptive and do not establish causality; continuations are dependent.',
    'Flags are diagnostic prompts, not pass/fail judgments of trading quality. No automatic winner or deletion is performed.',
    'Final TRAIN/VAL metrics, saved device metadata and ancestry are validated using the existing grid analyzer.',
]


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size: return dict(n=0, median=None, mean=None, p10=None, p90=None, min=None, max=None)
    return dict(n=int(a.size), median=float(np.median(a)), mean=float(a.mean()),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
                min=float(a.min()), max=float(a.max()))


def derive(arrays, n, minimum):
    def a(key): return arrays.get(key, np.full(n, np.nan))
    def div(x, y):
        out = np.full(n, np.nan)
        np.divide(x, y, out=out, where=np.isfinite(x) & np.isfinite(y) & (y > minimum))
        return out
    target = a('value_target_std')
    before = a('value_error_std') ** 2 + a('value_error_mean') ** 2
    after = a('post_train_value_mse')
    # Gate on standard deviation explicitly; the denominator is its square.
    variance = np.where(target > minimum, target ** 2, np.nan)
    def normalized(x):
        out = np.full(n, np.nan)
        np.divide(x, variance, out=out, where=np.isfinite(variance) & (variance > 0) & np.isfinite(x))
        return out
    pre, post = normalized(before), normalized(after)
    return dict(pre_mse=before, pre_nmse=pre, post_nmse=post,
                nmse_improvement=pre-post,
                ev_improvement=a('post_train_explained_variance')-a('explained_variance'),
                pre_bias_scaled=div(a('value_error_mean'), target),
                post_bias_scaled=div(a('post_train_value_error_mean'), target),
                pre_prediction_scale=div(a('value_prediction_std'), target),
                post_prediction_scale=div(a('post_train_value_prediction_std'), target),
                policy_entropy=-a('entropy_loss'))


def aggregate_curve(steps, arrays, limit):
    n = len(steps)
    bins = [x for x in np.array_split(np.arange(n), min(limit, n)) if len(x)]
    result = dict(x=[float(np.median(steps[b])) for b in bins], metrics={})
    for key, a in arrays.items():
        triplets = []
        for b in bins:
            v = a[b]; v = v[np.isfinite(v)]
            triplets.append(dict(median=float(np.median(v)) if v.size else None,
                min=float(v.min()) if v.size else None, max=float(v.max()) if v.size else None))
        result['metrics'][key] = dict(median=[s['median'] for s in triplets],
            min=[s['min'] for s in triplets], max=[s['max'] for s in triplets],
            first=float(a[0]), last=float(a[-1]))
    result['first_step'], result['last_step'] = int(steps[0]), int(steps[-1])
    return clean(result)


def analyze_metrics(records, fields, completed_steps, tail_fraction=.1, points=80, minimum=1e-12):
    n = len(records)
    if not n:
        return dict(records=0, flags=['NO_METRICS'], quality={'missing_records': True}, statistics={}, curve=None)
    steps = np.asarray([r['run_step'] for r in records], dtype=np.int64)
    if np.any(np.diff(steps) <= 0):
        raise ValueError('Duplicate or non-increasing run_step; metrics must be ordered and unique')
    arrays = {k: np.asarray([float(r[k]) if finite(r.get(k)) else np.nan for r in records]) for k in fields}
    nulls = {k: sum(r.get(k) is None for r in records) for k in fields}
    nonfinite = {k: sum(r.get(k) is not None and not finite(r[k]) for r in records) for k in fields}
    quality = dict(nulls={k:v for k,v in nulls.items() if v},
                   nonfinite={k:v for k,v in nonfinite.items() if v},
                   last_step_matches=bool(steps[-1] == completed_steps),
                   low_target_variance=int(np.sum(arrays.get('value_target_std', np.full(n,np.nan)) <= minimum)))
    rollouts = [r['rollout_number'] for r in records]
    quality['rollout_sequence_valid'] = all(v is not None for v in rollouts) and all(b>a for a,b in zip(rollouts,rollouts[1:]))
    negative_stds = {k:int(np.sum(a<0)) for k,a in arrays.items() if k.endswith('_std') and np.any(a<0)}
    quality['negative_standard_deviations'] = negative_stds
    arrays.update(derive(arrays, n, minimum))
    tail = max(1, math.ceil(n*tail_fraction))
    stats = {k: dict(all=summary(a), head=summary(a[:tail]), tail=summary(a[-tail:]),
                    first=clean(float(a[0])), last=clean(float(a[-1]))) for k,a in arrays.items()}
    paired = np.isfinite(arrays['pre_nmse']) & np.isfinite(arrays['post_nmse'])
    delta = arrays['nmse_improvement'][paired]
    quality['paired_updates'] = int(paired.sum())
    quality['updates_nmse_improved'] = int(np.sum(delta > 1e-10))
    quality['updates_nmse_worsened'] = int(np.sum(delta < -1e-10))
    quality['updates_nmse_tied'] = int(np.sum(np.abs(delta) <= 1e-10))
    quality['post_mse_moment_mismatches'] = 0
    if all(k in arrays for k in ('post_train_value_mse','post_train_value_error_std','post_train_value_error_mean')):
        expected = arrays['post_train_value_error_std']**2 + arrays['post_train_value_error_mean']**2
        actual = arrays['post_train_value_mse']
        mask = np.isfinite(expected)&np.isfinite(actual)
        quality['post_mse_moment_mismatches'] = int(np.sum(~np.isclose(expected[mask],actual[mask],rtol=1e-6,atol=1e-10)))
    flags=[]
    if any(v for k,v in nulls.items() if k not in OPTIONAL): flags.append('MISSING_METRICS')
    if any(nonfinite.values()): flags.append('NONFINITE_METRICS')
    if not quality['last_step_matches']: flags.append('LAST_METRIC_NOT_AT_RUN_END')
    if not quality['rollout_sequence_valid']: flags.append('ROLLOUT_SEQUENCE')
    if negative_stds or quality['post_mse_moment_mismatches']: flags.append('INCONSISTENT_MOMENTS')
    def tailmed(k): return stats.get(k,{}).get('tail',{}).get('median')
    if finite(tailmed('post_nmse')) and tailmed('post_nmse') > 1: flags.append('TAIL_NMSE_ABOVE_MEAN_BASELINE')
    if finite(tailmed('post_train_explained_variance')) and tailmed('post_train_explained_variance') < 0: flags.append('TAIL_NEGATIVE_EV')
    if finite(tailmed('post_bias_scaled')) and abs(tailmed('post_bias_scaled')) > 1: flags.append('TAIL_LARGE_SIGNED_BIAS')
    if finite(tailmed('post_prediction_scale')) and tailmed('post_prediction_scale') < .1: flags.append('TAIL_LOW_PREDICTION_VARIATION')
    if finite(tailmed('policy_entropy')) and tailmed('policy_entropy') < .05: flags.append('TAIL_LOW_POLICY_ENTROPY')
    if len(delta) and np.mean(delta < -1e-10) > .5: flags.append('MAJORITY_UPDATES_WORSEN_NMSE')
    return dict(records=n, tail_records=tail, quality=quality, flags=flags, statistics=stats,
                curve=aggregate_curve(steps, arrays, points))


def ancestor_names(names, by_name):
    result=set()
    def visit(name, active):
        if name in active: raise ValueError('Ancestry cycle: '+name)
        if name in result: return
        matches=by_name.get(name,[])
        if len(matches)!=1: raise ValueError(f'Expected one run named {name}; found {len(matches)}')
        r=matches[0]; result.add(name)
        cfg=r['normalized_config_json']
        parent=cfg.get('continuation',{}).get('source_run')
        if parent: visit(parent,active|{name})
    for name in names: visit(name,set())
    return result


def paired_diagnostics(comparison, metrics):
    pairs=[]
    for p in comparison['pairs']:
        aid,bid=p['A']['id'],p['B']['id']
        a,b=metrics[str(aid)],metrics[str(bid)]
        deltas={}
        for k in KEYS:
            av=a['statistics'].get(k,{}).get('tail',{}).get('median')
            bv=b['statistics'].get(k,{}).get('tail',{}).get('median')
            deltas[k]=bv-av if finite(av) and finite(bv) else None
        pairs.append(dict(A=aid,B=bid,cell=p['cell'],tail_median_delta=deltas,evaluation_delta=p['delta']))
    summaries=[]
    by_id={r['id']:r for r in comparison.get('_selected_rows',[])}
    configs=comparison.get('_configs',{})
    for coverage in comparison['coverage']:
        chosen=[p for p in pairs if not by_id or (
            {a:grid.axis_get(configs[by_id[p['A']]['name']],by_id[p['A']],a) for a in comparison['axes']}==coverage['A'] and
            {a:grid.axis_get(configs[by_id[p['B']]['name']],by_id[p['B']],a) for a in comparison['axes']}==coverage['B'])]
        if by_id:
            summaries.append(dict(A=coverage['A'],B=coverage['B'],pairs=len(chosen),
                tail_median_deltas={k:summary([p['tail_median_delta'][k] if finite(p['tail_median_delta'][k]) else np.nan for p in chosen]) for k in KEYS}))
    return dict(diagnostic_contrasts=summaries,axes=comparison['axes'],reference=comparison['reference'],variants=comparison['variants'],
                coverage=comparison['coverage'],ambiguous_cells=comparison['ambiguous_cells'],pairs=pairs)


def build_groups(rows, configs, metrics, axes):
    groups=defaultdict(list)
    for r in rows:
        values={a:grid.axis_get(configs[r['name']],r,a) for a in axes}
        groups[grid.canonical(dict(cohort=r['cohort'],values=values))].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        item=json.loads(key); item['runs']=len(rr)
        item['tail_medians']={k:summary([metrics[str(r['id'])]['statistics'].get(k,{}).get('tail',{}).get('median')
                                       if finite(metrics[str(r['id'])]['statistics'].get(k,{}).get('tail',{}).get('median')) else np.nan for r in rr]) for k in KEYS}
        item['val_return_leaders']=[dict(id=r['id'],name=r['name'],value=r['VAL']['agent_return']) for r in sorted(rr,key=lambda r:-r['VAL']['agent_return'])[:5]]
        item['associations']={}
        for k in KEYS:
            points=[(metrics[str(r['id'])]['statistics'].get(k,{}).get('tail',{}).get('median'),r['VAL']['agent_return']) for r in rr]
            points=[(a,b) for a,b in points if finite(a) and finite(b)]
            corr=None
            if len(points)>=3:
                x,y=np.asarray(points).T
                if np.std(x)>0 and np.std(y)>0: corr=float(np.corrcoef(x,y)[0,1])
            item['associations'][k]=dict(n=len(points),pearson_with_val_return=corr)
        out.append(item)
    return out


def read_report(root,args):
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    desired={}; provenance=[]
    for q in args.queue or []:
        _,configs,proof=grid.load_queue(root,q)
        for name,cfg in configs.items():
            if name in desired and grid.canonical(cfg)!=grid.canonical(desired[name]):
                raise ValueError('Conflicting queue copies: '+name)
            desired[name]=cfg
        provenance.append(proof)
    filters=grid.parse_assignments(args.where)
    reference=grid.parse_assignments(args.reference)
    engine=create_database_engine()
    try:
        with engine.connect() as con, con.begin():
            con.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
            con.execute(text("SET LOCAL statement_timeout = '300s'"))
            stamp=str(con.scalar(text('SELECT transaction_timestamp()')))
            runs=[dict(x) for x in con.execute(text('''SELECT r.*,p.name AS parent_name,p.id AS parent_id
                FROM public.runs r LEFT JOIN public.checkpoints c ON c.id=r.source_checkpoint_id
                LEFT JOIN public.runs p ON p.id=c.run_id''')).mappings()]
            by_name=defaultdict(list); by_id={}
            for r in runs:
                if isinstance(r['normalized_config_json'],str): r['normalized_config_json']=json.loads(r['normalized_config_json'])
                by_name[r['name']].append(r);by_id[r['id']]=r
            if args.run_id:
                missing=set(args.run_id)-set(by_id)
                if missing: raise ValueError('Run IDs not found: '+str(sorted(missing)))
                requested={by_id[i]['name'] for i in args.run_id}
            elif args.queue: requested=set(desired)
            else: requested={r['name'] for r in runs if r['cycle_id'] is None}
            names=ancestor_names(requested,by_name)
            chosen=[by_name[name][0] for name in sorted(names)]
            configs={r['name']:desired.get(r['name'],r['normalized_config_json']) for r in chosen}
            ids=[r['id'] for r in chosen]
            cps=[dict(x) for x in con.execute(text('SELECT * FROM public.checkpoints WHERE run_id=ANY(:ids)'),{'ids':ids}).mappings()]
            evs=[dict(x) for x in con.execute(text('''SELECT e.*,c.run_id,c.run_step AS checkpoint_run_step,
                c.model_step AS checkpoint_model_step FROM public.evaluations e JOIN public.checkpoints c ON c.id=e.checkpoint_id
                WHERE c.run_id=ANY(:ids) AND e.trigger='final' AND e.data_scope IN ('run_training','run_validation')'''),{'ids':ids}).mappings()]
            counts={x[0]:x[1] for x in con.execute(text('SELECT run_id,count(*) FROM public.training_metrics WHERE run_id=ANY(:ids) GROUP BY run_id'),{'ids':ids})}
            valid,inventory,issues=grid.prepare(root,configs,chosen,cps,evs,counts)
            selected=[r for r in valid if r['name'] in requested and all(grid.axis_get(configs[r['name']],r,k)==v for k,v in filters.items())]
            if not selected: raise ValueError('No eligible runs selected; issues: '+grid.canonical(issues[:10]))
            # Retain full ancestry for comparison, but never silently add it to the requested selection.
            selected_ids={r['id'] for r in selected}
            comparison=None
            if args.compare_by:
                if reference and not any(all(grid.axis_get(configs[r['name']],r,k)==v for k,v in reference.items()) for r in selected):
                    raise ValueError('Reference variant is absent from selected runs')
                comparison=grid.generic_comparison(valid,configs,args.compare_by,filters=filters,reference=reference,
                    rank_axes=args.compare_by,group_axes=[],top_percent=1)
                comparison['pairs']=[p for p in comparison['pairs'] if p['A']['id'] in selected_ids and p['B']['id'] in selected_ids]
                # Recompute matching within selected scope while retaining ancestors in signature lookup.
                for cov in comparison['coverage']:
                    cov['note']='Coverage below recomputed from selected run IDs; ancestors are context only.'
                comparison['coverage']=selected_coverage(comparison,selected,configs)
                comparison['ambiguous_cells']=[cell for cell in comparison['ambiguous_cells'] if any(i in selected_ids for i in cell['run_ids'])]
                comparison['_selected_rows']=selected
                comparison['_configs']=configs
            fields=con.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='training_metrics' ORDER BY ordinal_position")).scalars().all()
            fields=[k for k in fields if k not in IDENTITY]
            if not fields: raise ValueError('No training metric columns found')
            missing=set(('value_target_std','value_error_mean','value_error_std','post_train_value_mse','explained_variance','post_train_explained_variance'))-set(fields)
            if missing: issues.append(dict(level='WARNING',reason='Missing diagnostic columns: '+', '.join(sorted(missing))))
            metrics={}
            for index,r in enumerate(selected,1):
                records=[dict(x) for x in con.execute(text('SELECT * FROM public.training_metrics WHERE run_id=:id ORDER BY run_step,id'),{'id':r['id']}).mappings()]
                metrics[str(r['id'])]=analyze_metrics(records,fields,r['steps_completed'],args.tail_fraction,args.chart_points,args.min_target_std)
                if index==1 or index%50==0 or index==len(selected):
                    print(f"Training metrics: {index}/{len(selected)} runs",flush=True)
            context={r['name']:r for r in valid}
            for r in selected:
                current=r; ancestors=[]
                while current['source']:
                    current=context[current['source']];ancestors.append(current['id'])
                r['ancestor_ids']=list(reversed(ancestors))
                r['parameters']={k:grid.axis_get(configs[r['name']],r,grid.axis_path(k)) for k in grid.AXIS_ALIASES}
            groups=build_groups(selected,configs,metrics,list(map(grid.axis_path,args.group_by)))
            return dict(version=VERSION,snapshot=stamp,selection=dict(requested=len(requested),eligible=len(selected),
                excluded_requested=sorted(requested-{r['name'] for r in valid}),filters=filters,queues=provenance),
                settings=dict(tail_fraction=args.tail_fraction,chart_points=args.chart_points,min_target_std=args.min_target_std),
                issues=issues,notes=NOTES,runs=selected,metrics=metrics,groups=groups,
                comparison=paired_diagnostics(comparison,metrics) if comparison else None)
    finally:
        engine.dispose()


def selected_coverage(comparison,rows,configs):
    axes=comparison['axes']; result=[]
    def variant(r): return {a:grid.axis_get(configs[r['name']],r,a) for a in axes}
    variants_by_id={r['id']:variant(r) for r in rows}
    for old in comparison['coverage']:
        aa={i for i,v in variants_by_id.items() if v==old['A']}
        bb={i for i,v in variants_by_id.items() if v==old['B']}
        if not aa and not bb: continue
        pairs=[p for p in comparison['pairs'] if p['A']['id'] in aa and p['B']['id'] in bb]
        result.append(dict(A=old['A'],B=old['B'],matched=len(pairs),
            unmatched_A_ids=sorted(aa-{p['A']['id'] for p in pairs}),
            unmatched_B_ids=sorted(bb-{p['B']['id'] for p in pairs})))
    return result


def compact(report):
    result={k:v for k,v in report.items() if k not in ('metrics','runs')}
    result['runs']=[]
    for r in report['runs']:
        m=report['metrics'][str(r['id'])]
        result['runs'].append(dict(id=r['id'],name=r['name'],stage=r['stage'],source=r['source'],
            ancestor_ids=r['ancestor_ids'],parameters=r['parameters'],TRAIN=r['TRAIN'],VAL=r['VAL'],
            records=m['records'],flags=m['flags'],quality=m['quality'],
            tail={k:m['statistics'].get(k,{}).get('tail',{}).get('median') for k in KEYS}))
    return clean(result)


def render(report):
    from plotly.offline import get_plotlyjs
    payload=json.dumps(clean(report),default=grid.serial,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>PPO training diagnostics</title>
<style>body{font:15px system-ui;margin:24px;color:#172435;background:#f7f9fc}h1{margin-bottom:8px}button,select,input{padding:8px;margin:4px}nav{position:sticky;top:0;background:#f7f9fc;padding:10px;z-index:2}section{display:none}section.active{display:block}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:7px;border-bottom:1px solid #d7deea;text-align:left}th{background:#e8eff8} .scroll{overflow:auto;max-height:65vh} .plot{height:500px}pre{white-space:pre-wrap}code{overflow-wrap:anywhere}.note{background:#e6eef9;padding:12px}label{display:inline-block}</style>
<h1>PPO training diagnostics</h1><p id="summary"></p><p class="note">Critic fit on rollout targets is not validation performance. No automatic trading-quality verdict.</p>
<nav><button data-tab="overview">Overview</button><button data-tab="run">Run metrics</button><button data-tab="paths">Paths</button><button data-tab="pairs">Matched comparisons</button><button data-tab="groups">Groups</button><button data-tab="quality">Quality & definitions</button></nav>
<section id="overview" class="active"><label>Diagnostic <select id="scatterMetric"></select></label><div id="scatter" class="plot"></div><div id="runTable" class="scroll"></div></section>
<section id="run"><input id="search" placeholder="Filter run name or ID"><select id="runSelect"></select><select id="metricSelect"></select><div id="runChart" class="plot"></div><div id="stats" class="scroll"></div></section>
<section id="paths"><label>Path ending at <select id="pathSelect"></select></label><select id="pathMetric"></select><div id="pathChart" class="plot"></div><div id="pathInfo"></div></section>
<section id="pairs"><p>B minus A. Signs are numeric differences, not universal better/worse labels. Tail windows summarize each run separately.</p><select id="pairSelect"></select><select id="pairMetric"></select><div id="pairChart" class="plot"></div><div id="pairInfo"></div></section>
<section id="groups"><p>Descriptive groups retain the grid analyzer's fixed-condition cohorts. Correlations are exploratory; dependent stages are not independent replications.</p><select id="groupSelect"></select><pre id="groupInfo"></pre></section>
<section id="quality"><div id="notes"></div><pre id="issues"></pre></section>
<script>''' + get_plotlyjs() + '</script><script id="data" type="application/json">'+payload+'''</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id), R=new Map(D.runs.map(r=>[String(r.id),r]));
const keys=[...new Set(Object.values(D.metrics).flatMap(m=>Object.keys(m.statistics)))];
const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>typeof x==='number'?Number(x.toPrecision(6)).toString():esc(x);
const table=(headers,rows)=>'<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+fmt(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
const options=(id,items)=>{$(id).replaceChildren(...items.map(([value,label])=>{let o=document.createElement('option');o.value=value;o.textContent=label;return o}))};
const tail=(id,k)=>D.metrics[String(id)]?.statistics[k]?.tail.median??null;
$('summary').textContent=`${D.runs.length} eligible runs; ${Object.values(D.metrics).reduce((n,m)=>n+m.records,0)} metric records. Snapshot: ${D.snapshot}`;
for(let id of ['runSelect','pathSelect'])options(id,D.runs.map(r=>[r.id,`#${r.id} ${r.name}`]));
for(let id of ['metricSelect','pathMetric','pairMetric','scatterMetric']){options(id,keys.map(k=>[k,k]));$(id).value=keys.includes('post_nmse')?'post_nmse':keys[0]}
function plot(id,traces,title,xlabel='Run steps',ylabel='Value'){Plotly.react(id,traces,{title,xaxis:{title:{text:xlabel}},yaxis:{title:{text:ylabel}},margin:{t:60},hovermode:'closest'},{responsive:true,displaylogo:false})}
function traces(r,k,label){let c=D.metrics[String(r.id)].curve;if(!c||!c.metrics[k])return [];let m=c.metrics[k];return [
{x:c.x,y:m.min,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',legendgroup:label},
{x:c.x,y:m.max,mode:'lines',line:{width:0},fill:'tonexty',opacity:.15,showlegend:false,hoverinfo:'skip',legendgroup:label},
{x:c.x,y:m.median,mode:'lines',name:label,connectgaps:false,legendgroup:label},
{x:[c.first_step,c.last_step],y:[m.first,m.last],mode:'markers',name:label+' endpoints',legendgroup:label}];}
const companion={post_nmse:'pre_nmse',post_train_explained_variance:'explained_variance',post_bias_scaled:'pre_bias_scaled',post_prediction_scale:'pre_prediction_scale',post_train_value_target_prediction_corr:'value_target_prediction_corr'};
function run(){let r=R.get($('runSelect').value);if(!r)return;let k=$('metricSelect').value,m=D.metrics[String(r.id)];let ts=traces(r,k,k);if(companion[k])ts.push(...traces(r,companion[k],companion[k]));plot('runChart',ts,k+' — binned median and min/max envelope');$('stats').innerHTML=table(['Metric','All median','Head median','Tail median','Min','Max','Finite N'],Object.entries(m.statistics).map(([k,v])=>[k,v.all.median,v.head.median,v.tail.median,v.all.min,v.all.max,v.all.n]));}
function scatter(){let k=$('scatterMetric').value;plot('scatter',[{x:D.runs.map(r=>tail(r.id,k)),y:D.runs.map(r=>100*r.VAL.agent_return),text:D.runs.map(r=>'#'+r.id+' '+r.name),customdata:D.runs.map(r=>r.id),mode:'markers',type:'scattergl'}],'Final VAL return vs tail diagnostic',k,'VAL return [%]');$('scatter').removeAllListeners('plotly_click');$('scatter').on('plotly_click',e=>{let id=e.points[0].customdata;options('runSelect',D.runs.map(r=>[r.id,'#'+r.id+' '+r.name]));$('runSelect').value=id;show('run');run()})}
function paths(){let r=R.get($('pathSelect').value);if(!r)return;let ids=[...r.ancestor_ids,r.id],rr=ids.map(i=>R.get(String(i))).filter(Boolean),k=$('pathMetric').value;plot('pathChart',[{x:rr.map(x=>x.stage+1),y:rr.map(x=>tail(x.id,k)),mode:'lines+markers',name:k}],k+' by continuation stage','Stage + 1',k);$('pathInfo').innerHTML='<p>Only selected runs are plotted. Missing ancestor IDs: '+esc(ids.filter(i=>!R.has(String(i))).join(', ')||'none')+'</p>'+table(['Run','Stage','TRAIN return [%]','VAL return [%]','VAL DD [%]','Tail '+k],rr.map(x=>[x.id,x.stage,100*x.TRAIN.agent_return,100*x.VAL.agent_return,100*x.VAL.agent_max_drawdown,tail(x.id,k)]));}
const pairs=D.comparison?.pairs??[];options('pairSelect',pairs.map((p,i)=>[i,`#${p.A} → #${p.B}`]));
function pair(){let p=pairs[Number($('pairSelect').value)];if(!p){$('pairInfo').textContent='No matched pairs. Use --compare-by and optionally --reference.';return}let k=$('pairMetric').value;plot('pairChart',[...traces(R.get(String(p.A)),k,'A #'+p.A),...traces(R.get(String(p.B)),k,'B #'+p.B)],k+' — matched runs');$('pairInfo').textContent=JSON.stringify(p,null,2)}
options('groupSelect',D.groups.map((g,i)=>[i,`${g.cohort}: ${JSON.stringify(g.values)} (${g.runs} runs)`]));
function group(){$('groupInfo').textContent=JSON.stringify(D.groups[Number($('groupSelect').value)]??{},null,2)}
$('runTable').innerHTML=table(['ID','Name','Stage','VAL return [%]','VAL DD [%]','Tail post NMSE','Tail post EV','Flags'],D.runs.slice().sort((a,b)=>b.VAL.agent_return-a.VAL.agent_return).map(r=>[r.id,r.name,r.stage,100*r.VAL.agent_return,100*r.VAL.agent_max_drawdown,tail(r.id,'post_nmse'),tail(r.id,'post_train_explained_variance'),D.metrics[String(r.id)].flags.join(', ')]));
$('notes').innerHTML='<ul>'+D.notes.map(n=>'<li>'+esc(n)+'</li>').join('')+'</ul>';
$('issues').textContent=JSON.stringify({selection:D.selection,settings:D.settings,issues:D.issues,comparison_coverage:D.comparison?.coverage,ambiguous_cells:D.comparison?.ambiguous_cells,per_run:Object.fromEntries(Object.entries(D.metrics).map(([id,m])=>[id,{flags:m.flags,quality:m.quality}]))},null,2);
function show(id){document.querySelectorAll('section').forEach(s=>s.classList.toggle('active',s.id===id));if(id==='overview')scatter();if(id==='run')run();if(id==='paths')paths();if(id==='pairs')pair();if(id==='groups')group()}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.tab));
for(let id of ['runSelect','metricSelect'])$(id).onchange=run;for(let id of ['pathSelect','pathMetric'])$(id).onchange=paths;for(let id of ['pairSelect','pairMetric'])$(id).onchange=pair;$('scatterMetric').onchange=scatter;$('groupSelect').onchange=group;
$('search').oninput=()=>{let q=$('search').value.toLowerCase();options('runSelect',D.runs.filter(r=>(r.id+' '+r.name).toLowerCase().includes(q)).map(r=>[r.id,'#'+r.id+' '+r.name]));run()};show('overview');
</script></html>'''


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    scope=p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--all-runs',action='store_true',help='All stage-one runs; only validated completed runs are eligible')
    scope.add_argument('--queue',action='append',help='Queue name/path; repeat to combine queues')
    scope.add_argument('--run-id',action='append',type=int,help='Repeat to select individual runs; ancestors loaded for validation only')
    p.add_argument('--where',action='append',default=[],help='FIELD=JSON_VALUE; aliases as in grid analyzer')
    p.add_argument('--compare-by',nargs='+',help='Matched comparison axes, e.g. ent_coef gae_lambda')
    p.add_argument('--reference',action='append',default=[],help='Specify every comparison axis, or omit for all contrasts')
    p.add_argument('--group-by',nargs='+',default=['architecture','n_epochs','gamma','ent_coef','gae_lambda'])
    p.add_argument('--tail-fraction',type=float,default=.1)
    p.add_argument('--chart-points',type=int,default=80,help='Maximum number of bins per run; statistics always use all records')
    p.add_argument('--min-target-std',type=float,default=1e-12)
    p.add_argument('--json-detail',choices=['summary','full'],default='summary',help='Full exports all metric statistics, still no chart series')
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/training_analysis'))
    args=p.parse_args(argv)
    if not math.isfinite(args.tail_fraction) or not 0<args.tail_fraction<=1:p.error('--tail-fraction must be in (0,1]')
    if args.chart_points<2:p.error('--chart-points must be >= 2')
    if not math.isfinite(args.min_target_std) or args.min_target_std<0:p.error('--min-target-std must be finite and >= 0')
    if args.reference and not args.compare_by:p.error('--reference requires --compare-by')
    return args


def main(argv=None):
    args=parse_args(argv);root=Path.cwd().resolve()
    if not (root/'run_search_queue.py').is_file():raise ValueError('Run from the PPO_AGENT root')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    print('Reading one consistent read-only database snapshot; no training or model loading.',flush=True)
    report=read_report(root,args)
    exported=compact(report)
    if args.json_detail=='full':exported['metric_statistics']={k:{a:b for a,b in v.items() if a!='curve'} for k,v in report['metrics'].items()}
    # Render and serialize before creating an output directory.
    rendered=render(report)
    serialized=json.dumps(clean(exported),default=grid.serial,allow_nan=False,indent=2)+'\n'
    parent=args.output_dir.expanduser()
    if not parent.is_absolute():parent=root/parent
    out=parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    (out/'report.html').write_text(rendered,encoding='utf-8')
    (out/'report.json').write_text(serialized,encoding='utf-8')
    print(f"Eligible: {len(report['runs'])}; metric records: {sum(m['records'] for m in report['metrics'].values())}; issues: {len(report['issues'])}")
    print('HTML:',out/'report.html');print('Compact JSON:',out/'report.json')
    print('Database, configs, checkpoints and existing reports unchanged.')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,RuntimeError) as exc:raise SystemExit(f'STOP: {type(exc).__name__}: {exc}')
