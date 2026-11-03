"""Synthetic diagnostics tests. No DB, GPU or market-data access."""
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import sys
import pytest
spec=importlib.util.spec_from_file_location('training_analysis_test',Path(__file__).resolve().parents[1]/'extra_tools/analysis/ppo_analyze_training.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def records(n=10):
    return [dict(run_step=(i+1)*1024,model_step=(i+1)*1024,rollout_number=i+1,
        value_target_std=2.,value_target_mean=0.,value_error_mean=2.,value_error_std=2.,
        value_prediction_mean=-2.,value_prediction_std=1.,
        post_train_value_error_mean=1.,post_train_value_error_std=1.,
        post_train_value_prediction_mean=-1.,post_train_value_prediction_std=1.5,
        post_train_value_mse=2.,explained_variance=0.,post_train_explained_variance=.75,
        value_target_prediction_corr=.5,post_train_value_target_prediction_corr=.8,
        entropy_loss=-.4,approx_kl=.01,clip_fraction=.1,value_loss=8.,
        ep_reward=None if i<n-1 else 3.,ep_len=None if i<n-1 else n*1024,
        rollout_reward_mean=.01) for i in range(n)]

def analyze(rr=None,**kw):
    rr=records() if rr is None else rr
    return m.analyze_metrics(rr,[k for k in rr[0] if k not in m.IDENTITY] if rr else [],len(rr)*1024,**kw)

def test_population_moments_and_normalization():
    a=analyze();s=a['statistics']
    for k,v in dict(pre_mse=8,pre_nmse=2,post_nmse=.5,nmse_improvement=1.5,post_bias_scaled=.5,post_prediction_scale=.75,policy_entropy=.4).items():
        assert s[k]['all']['median']==v
    assert a['quality']['updates_nmse_improved']==10
    assert not a['flags']

def test_ev_ignores_bias_but_nmse_does_not():
    rr=records()
    for r in rr:r.update(value_error_mean=20.,value_error_std=1.,explained_variance=.75)
    s=analyze(rr)['statistics']
    assert s['explained_variance']['all']['median']==.75
    assert s['pre_nmse']['all']['median']==100.25

def test_zero_variance_is_undefined():
    rr=records();rr[0]['value_target_std']=0;a=analyze(rr)
    assert a['quality']['low_target_variance']==1
    assert a['statistics']['pre_nmse']['all']['n']==9
    assert a['statistics']['pre_nmse']['first'] is None

def test_threshold_applies_to_std_not_variance():
    rr=records(1);rr[0]['value_target_std']=1e-7
    assert analyze(rr,minimum=1e-8)['statistics']['pre_nmse']['first']==pytest.approx(8e14)

def test_expected_episode_nulls_and_nonfinite():
    a=analyze();assert a['quality']['nulls']['ep_reward']==9
    assert 'MISSING_METRICS' not in a['flags']
    rr=records();rr[0]['ep_reward']=float('inf')
    assert 'NONFINITE_METRICS' in analyze(rr)['flags']

def test_missing_critic_field():
    rr=records();rr[3]['value_error_std']=None;a=analyze(rr)
    assert 'MISSING_METRICS' in a['flags']
    assert a['statistics']['pre_nmse']['all']['n']==9

def test_tail_is_chronological_and_rounded_up():
    rr=records(11)
    for i,r in enumerate(rr):r['approx_kl']=i
    a=analyze(rr,tail_fraction=.1)
    assert a['tail_records']==2
    assert a['statistics']['approx_kl']['tail']['median']==9.5
    assert a['statistics']['approx_kl']['head']['median']==.5

def test_chart_keeps_extremes_statistics_use_all_records():
    rr=records(100);rr[43]['approx_kl']=100;a=analyze(rr,points=3)
    assert max(a['curve']['metrics']['approx_kl']['max'])==100
    assert a['statistics']['approx_kl']['all']['n']==100
    assert len(a['curve']['x'])==3

def test_moment_mismatch_and_negative_std():
    rr=records();rr[0]['post_train_value_mse']=90;rr[0]['value_error_std']=-2
    a=analyze(rr)
    assert a['quality']['post_mse_moment_mismatches']==1
    assert 'INCONSISTENT_MOMENTS' in a['flags']

def test_metric_end_mismatch():
    rr=records();a=m.analyze_metrics(rr,[k for k in rr[0] if k not in m.IDENTITY],99999)
    assert 'LAST_METRIC_NOT_AT_RUN_END' in a['flags']

@pytest.mark.parametrize('steps',[[1024,1024],[2048,1024]])
def test_invalid_steps(steps):
    rr=records(2)
    for r,s in zip(rr,steps):r['run_step']=s
    with pytest.raises(ValueError,match='non-increasing'):analyze(rr)

def test_empty_run():
    a=analyze([]);assert a['flags']==['NO_METRICS'] and a['curve'] is None

def test_ancestry():
    def row(n,p=None):return dict(name=n,normalized_config_json=dict(continuation=dict(source_run=p) if p else {}))
    by={'a':[row('a')],'b':[row('b','a')]}
    assert m.ancestor_names({'b'},by)=={'a','b'}
    with pytest.raises(ValueError,match='Expected one'):m.ancestor_names({'c'},by)
    by['a'][0]['normalized_config_json']['continuation']['source_run']='b'
    with pytest.raises(ValueError,match='cycle'):m.ancestor_names({'b'},by)

@pytest.mark.parametrize('args',[
    ['--all-runs','--tail-fraction','0'],['--all-runs','--tail-fraction','nan'],
    ['--all-runs','--chart-points','1'],['--all-runs','--min-target-std','-1'],
    ['--all-runs','--reference','seed=1'],['--all-runs','--queue','x']])
def test_invalid_cli(args):
    with pytest.raises(SystemExit):m.parse_args(args)

def sample():
    from tests.test_flexible_grid_comparison import dataset
    rows,configs=dataset()
    for r in rows:
        r['cohort']='same';r['ancestor_ids']=[];parent=r['source']
        while parent:
            pr=next(x for x in rows if x['name']==parent)
            r['ancestor_ids'].insert(0,pr['id']);parent=pr['source']
        r['parameters']={k:m.grid.axis_get(configs[r['name']],r,m.grid.axis_path(k)) for k in m.grid.AXIS_ALIASES}
    metrics={str(r['id']):analyze() for r in rows}
    comparison=m.grid.generic_comparison(rows,configs,['ent_coef','gae_lambda'],reference={'ppo.ent_coef':.0002,'ppo.gae_lambda':.85})
    return dict(version=m.VERSION,snapshot='synthetic',selection={},settings={},issues=[],notes=m.NOTES,
                runs=rows,metrics=metrics,groups=m.build_groups(rows,configs,metrics,['ppo.n_epochs']),
                comparison=m.paired_diagnostics(comparison,metrics)),configs

def test_matched_deltas():
    r,_=sample();assert len(r['comparison']['pairs'])==18
    assert all(p['tail_median_delta']['post_nmse']==0 for p in r['comparison']['pairs'])
    assert all(p['evaluation_delta']['VAL']['agent_return']>0 for p in r['comparison']['pairs'])

def test_compact_strict_json_without_curves():
    r,_=sample();c=m.compact(r)
    assert 'metrics' not in c and len(c['runs'])==24
    assert all('tail' in x for x in c['runs'])
    json.dumps(c,allow_nan=False)

def test_constant_metric_correlation_undefined():
    r,_=sample()
    assert all(g['associations']['post_nmse']['pearson_with_val_return'] is None for g in r['groups'])

def test_selected_coverage_excludes_ancestor_context():
    r,c=sample();rows=r['runs'];cmp=m.grid.generic_comparison(rows,c,['ent_coef','gae_lambda'],reference={'ppo.ent_coef':.0002,'ppo.gae_lambda':.85})
    selected=[x for x in rows if x['stage']==2];ids={x['id'] for x in selected}
    cmp['pairs']=[p for p in cmp['pairs'] if p['A']['id'] in ids and p['B']['id'] in ids]
    cov=m.selected_coverage(cmp,selected,c)
    assert all(x['matched']==2 and not x['unmatched_A_ids'] and not x['unmatched_B_ids'] for x in cov)

def test_html_offline_and_json_escaping():
    r,_=sample();r['runs'][0]['name']='</script><script>alert(1)</script>';s=m.render(r)
    assert '<script src=' not in s
    assert r['runs'][0]['name'] not in s
    assert 'Matched comparisons' in s

def test_snapshot_read_only_metrics_inside_transaction(monkeypatch,tmp_path):
    report,configs=sample();valid=report['runs'][:3]
    runs=[dict(r,normalized_config_json=configs[r['name']],cycle_id=None) for r in valid]
    rr=records();fields=list(rr[0]);queries=[]
    class Result:
        def __init__(self,data):self.data=data
        def mappings(self):return self
        def scalars(self):return self
        def all(self):return self.data
        def __iter__(self):return iter(self.data)
    class Connection:
        active=False
        def __enter__(self):return self
        def __exit__(self,*exc):self.active=False
        def begin(self):self.active=True;return self
        def scalar(self,q):assert self.active;return 'snapshot'
        def execute(self,q,params=None):
            assert self.active
            q=str(q);queries.append(q)
            if 'FROM public.runs' in q:return Result(runs)
            if 'information_schema.columns' in q:return Result(fields)
            if 'count(*)' in q:return Result([(r['id'],10) for r in runs])
            if 'SELECT * FROM public.training_metrics' in q:return Result(rr)
            return Result([])
    con=Connection()
    class Engine:
        disposed=False
        def connect(self):return con
        def dispose(self):self.disposed=True
    engine=Engine();module=ModuleType('train_and_eval.database.session');module.create_database_engine=lambda:engine
    monkeypatch.setitem(sys.modules,'train_and_eval.database.session',module)
    monkeypatch.setattr(m.grid,'prepare',lambda *a:(copy.deepcopy(valid),[],[]))
    out=m.read_report(tmp_path,m.parse_args(['--run-id',str(valid[-1]['id'])]))
    assert len(out['runs'])==1 and out['runs'][0]['ancestor_ids']==[valid[0]['id'],valid[1]['id']]
    assert engine.disposed
    assert queries[0]=='SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'
    assert not any(q.lstrip().split()[0] in {'UPDATE','INSERT','DELETE'} for q in queries)

def test_main_writes_new_reports_without_overwriting_existing(monkeypatch,tmp_path):
    report,_=sample()
    (tmp_path/'run_search_queue.py').write_text('# synthetic project marker\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m,'read_report',lambda *a:report)
    m.main(['--all-runs'])
    first=list((tmp_path/'artifacts/training_analysis').iterdir())[0]
    original=(first/'report.json').read_bytes()
    m.main(['--all-runs','--json-detail','full'])
    directories=list((tmp_path/'artifacts/training_analysis').iterdir())
    assert len(directories)==2 and (first/'report.json').read_bytes()==original
    second=next(p for p in directories if p!=first)
    full=json.loads((second/'report.json').read_text())
    assert 'metric_statistics' in full
    assert all('curve' not in v for v in full['metric_statistics'].values())

def test_contrast_summary_uses_only_its_variant_pair():
    r,c=sample();rows=r['runs']
    comparison=m.grid.generic_comparison(rows,c,['ent_coef','gae_lambda'],reference={'ppo.ent_coef':.0002,'ppo.gae_lambda':.85})
    comparison['_selected_rows']=rows;comparison['_configs']=c
    d=m.paired_diagnostics(comparison,r['metrics'])
    assert len(d['diagnostic_contrasts'])==3
    assert all(x['pairs']==6 and x['tail_median_deltas']['post_nmse']['n']==6 for x in d['diagnostic_contrasts'])
