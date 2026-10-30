"""Generic comparisons: synthetic inputs, no database or hardware access."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('flexible_grid',Path(__file__).resolve().parents[1]/'extra_tools/analysis/ppo_analyze_grid.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def dataset():
    rows=[];configs={}
    for ent in (.0002,.002):
        for gae in (.85,.95):
            for seed in (1,2):
                parent=None
                for stage in range(3):
                    rid=len(rows)+1;name=f'r{rid}'
                    c=dict(run=dict(name=name,seed=seed),ppo=dict(ent_coef=ent,gae_lambda=gae,
                        learning_rate=.00002,gamma=.9,n_epochs=3,hidden_sizes=[8,8],device='cuda',vf_coef=.5),
                        data=dict(path='data/test.parquet'),environment=dict(window=40),
                        training=dict(duration_amount=1,duration_unit='data_epochs'),evaluation=dict(device='cpu'),
                        continuation=dict(mode='resume',source_run=parent,checkpoint='final') if parent else dict(mode='fresh'))
                    m=dict(agent_return=ent*100+gae+stage*.01,balanced_score=ent*100+gae-.1,
                        agent_max_drawdown=-.1,market_exposure=.5,round_trips=50,profit_factor=1.1,
                        win_rate=.5,always_long_return=.2,signature=dict(data_sha256='same',seed=seed),excess_return=.1)
                    r=dict(id=rid,name=name,seed=seed,stage=stage,source=parent,source_checkpoint_id=rid-1 if parent else None,
                        device='GPU',steps_completed=1024,stopped_early=False,TRAIN=copy.deepcopy(m),VAL=copy.deepcopy(m))
                    rows.append(r);configs[name]=c;parent=name
    return rows,configs


def compare(**kw):
    rows,configs=dataset()
    return g.generic_comparison(rows,configs,kw.pop('axes',['ent_coef','gae_lambda']),**kw)


def test_cartesian_pairs_and_reference():
    r=compare();assert len(r['pairs'])==36
    assert len(r['coverage'])==6
    assert all(c['matched']==6 for c in r['coverage'])
    r=compare(reference={'ppo.ent_coef':.0002,'ppo.gae_lambda':.85})
    assert len(r['pairs'])==18
    assert all(p['delta']['VAL']['agent_return']>0 for p in r['pairs'])


def test_single_axis_holds_other_axis_seed_and_stage():
    r=compare(axes=['ent_coef'])
    assert len(r['pairs'])==12
    assert all(p['A']['seed']==p['B']['seed'] and p['A']['stage']==p['B']['stage'] for p in r['pairs'])


def test_seed_pairs_keep_entropy_and_gae_fixed():
    r=compare(axes=['seed']);assert len(r['pairs'])==12
    assert all(abs(p['delta']['VAL']['agent_return'])<1e-12 for p in r['pairs'])


def test_stage_progression_and_ranking_pooling():
    assert len(compare(axes=['stage'])['pairs'])==24
    r=compare(rank_axes=['ent_coef','gae_lambda','stage'])
    assert len(r['rankings'])==2  # one per seed
    assert all(x['runs']==12 for x in r['rankings'])


def test_filter_retains_full_candidate_paths():
    r=compare(filters={'run.seed':1,'stage':2})
    assert r['selected_runs']==4
    assert all(len(chain)==3 for chain in r['candidate_paths'].values())


def test_unmatched_and_duplicates_are_not_guessed():
    rows,c=dataset();rows=rows[:-1]
    r=g.generic_comparison(rows,c,['seed'])
    assert len(r['pairs'])==11
    assert r['coverage'][0]['only_A']==1
    rows,c=dataset();duplicate=copy.deepcopy(rows[0]);duplicate['id']=100;duplicate['name']='duplicate'
    c['duplicate']=copy.deepcopy(c[rows[0]['name']]);c['duplicate']['run']['name']='duplicate';rows.append(duplicate)
    r=g.generic_comparison(rows,c,['seed'])
    assert len(r['pairs'])==11 and len(r['ambiguous_cells'])==1


def test_unselected_scientific_difference_and_ancestry_block_pair():
    rows,c=dataset();c['r4']['environment']['window']=41
    r=g.generic_comparison(rows,c,['seed'])
    assert len(r['pairs'])==9  # changed root also invalidates its descendant matches


def test_arbitrary_scientific_field():
    rows,c=dataset()
    for row in rows:c[row['name']]['ppo']['vf_coef']=row['seed']*.5
    r=g.generic_comparison(rows,c,['seed','ppo.vf_coef'])
    assert len(r['pairs'])==12
    assert len(g.generic_comparison(rows,c,['ppo.vf_coef'])['pairs'])==0


def test_fixed_evaluation_conditions_cannot_be_pooled():
    rows,c=dataset()
    for r in rows:
        if r['seed']==2:
            for s in g.SCOPES:r[s]['signature']['data_sha256']='different'
    assert not g.generic_comparison(rows,c,['seed'])['pairs']


def test_baseline_mismatch_stops():
    rows,c=dataset();rows[3]['VAL']['always_long_return']=.3
    with pytest.raises(ValueError,match='baseline differs'):
        g.generic_comparison(rows,c,['seed'])


@pytest.mark.parametrize('axis',['bogus','run.name','continuation.source_run','continuation.mode'])
def test_invalid_axes(axis):
    with pytest.raises(ValueError):compare(axes=[axis])


def test_missing_field_and_reference_errors():
    with pytest.raises(ValueError,match='Missing comparison field'):compare(axes=['ppo.missing'])
    with pytest.raises(ValueError,match='every'):compare(reference={'ppo.ent_coef':.0002})
    with pytest.raises(ValueError,match='absent'):compare(reference={'ppo.ent_coef':.8,'ppo.gae_lambda':.85})
    with pytest.raises(ValueError,match='No eligible'):compare(filters={'run.seed':99})


def test_assignments_and_overlap():
    assert g.parse_assignments(['architecture=[8,8]','seed=1'])=={'ppo.hidden_sizes':[8,8],'run.seed':1}
    for values in (['seed=NaN'],['seed=1','seed=2'],['seed']):
        with pytest.raises(ValueError):g.parse_assignments(values)
    with pytest.raises(ValueError,match='Overlapping'):compare(axes=['ppo.hidden_sizes','ppo.hidden_sizes.foo'])


def test_json_export_and_html_tabs(tmp_path):
    rows,c=dataset()
    args=SimpleNamespace(compare_by=['ent_coef','gae_lambda'],where=['seed=1'],reference=[],
        rank_across=['ent_coef','gae_lambda','stage'],group_by=['n_epochs'],top_percent=1,
        trajectory_pair=[],comparison_export='summary',output_dir=tmp_path/'out')
    g.generic_write(tmp_path,args,rows,c,[], 'test',dict(source='synthetic'))
    out=next(args.output_dir.iterdir());result=json.loads((out/'report.json').read_text())
    assert result['pair_count']==18 and 'pairs' not in result
    html=(out/'report.html').read_text()
    assert 'role="tablist"' in html and 'All matched pairs' in html
    assert 'use --compare-seeds' not in html.lower()
    assert 'r21' in html and '>r24<' not in html


def test_trajectory_must_be_matched_before_any_db_access(tmp_path,monkeypatch):
    rows,c=dataset()
    monkeypatch.setattr(g,'path_read_snapshot',lambda *a:pytest.fail('Unexpected DB access'))
    args=SimpleNamespace(compare_by=['seed'],where=[],reference=[],rank_across=None,group_by=[],
        top_percent=1,trajectory_pair=['1:999'],comparison_export='summary',output_dir=tmp_path/'out')
    with pytest.raises(ValueError,match='not an unambiguous'):g.generic_write(tmp_path,args,rows,c,[],'test',{})
    assert not args.output_dir.exists()


def test_full_generic_cli_routes_before_legacy(monkeypatch,tmp_path):
    rows,c=dataset()
    (tmp_path/'run_search_queue.py').touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(g,'snapshot',lambda names: ([],[],[],{},'stamp'))
    monkeypatch.setattr(g,'load_queue',lambda root,q: ({},c,{'queue_path':q}))
    monkeypatch.setattr(g,'prepare',lambda *args: (rows,[],[]))
    called=[]
    monkeypatch.setattr(g,'generic_write',lambda *args: called.append(args))
    monkeypatch.setattr(g.sys,'argv',['tool','--queue','grid3_s1','--mode','full','--compare-by','ent_coef','--trajectory-pair','1:13'])
    g.main()
    assert len(called)==1


@pytest.mark.parametrize('options',[
    ['--where','seed=1'],
    ['--compare-by','seed','--compare-seeds','1','2'],
    ['--compare-by','seed','--mode','grid','--trajectory-pair','1:4'],
    ['--compare-by','seed','--clean-old-reports'],
])
def test_cli_rejects_ambiguous_modes_before_database(monkeypatch,options):
    monkeypatch.setattr(g.sys,'argv',['tool','--all-runs']+options)
    monkeypatch.setattr(g,'snapshot',lambda *a:pytest.fail('Unexpected DB read'))
    with pytest.raises(ValueError):g.main()


def test_duplicate_id_selectors_and_dict_axis():
    rows,c=dataset()
    r=g.generic_comparison(rows,c,['environment.window','seed'])
    assert len(r['pairs'])==12
    assert r['axes']==['environment.window','run.seed']
    with pytest.raises(ValueError):g.generic_comparison(rows,c,['ppo.hidden_sizes.foo'])


def test_complete_grid3_has_432_baseline_pairs_and_one_global_front():
    from itertools import product
    rows=[];configs={}
    base_rows,base_configs=dataset()
    for ne,arch,gamma,lr in product((3,9),([384]*3,[384]*4,[768]*3,[768]*4),(.9,.95,.99),(.00002,.00004)):
        mapping={r['name']:f"g{len(rows)}_{r['name']}" for r in base_rows if r['seed']==1}
        for original in base_rows:
            if original['seed']!=1:continue
            r=copy.deepcopy(original);c=copy.deepcopy(base_configs[original['name']])
            r['id']=len(rows)+1;r['name']=mapping[original['name']];r['source']=mapping.get(original['source'])
            c['run']['name']=r['name']
            c['ppo'].update(n_epochs=ne,hidden_sizes=arch,gamma=gamma,learning_rate=lr)
            if r['source']:c['continuation']['source_run']=r['source']
            rows.append(r);configs[r['name']]=c
    result=g.generic_comparison(rows,configs,['ent_coef','gae_lambda'],
        reference={'ppo.ent_coef':.0002,'ppo.gae_lambda':.85},
        rank_axes=['architecture','learning_rate','gamma','n_epochs','ent_coef','gae_lambda','stage'])
    assert len(rows)==576
    assert len(result['pairs'])==432
    assert [c['matched'] for c in result['coverage']]==[144]*3
    assert len(result['rankings'])==1
    assert not result['ambiguous_cells']
