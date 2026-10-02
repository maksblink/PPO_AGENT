import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from train_and_eval.dashboard.coverage import (
    coordinates, coverage_rows, family_settings, grid_cells, read_plan, decimal_text,
)
from train_and_eval.run_config import RunConfig, normalize_config


@pytest.fixture
def plan():
    config = yaml.safe_load((Path(__file__).parent/'fixtures/temporal_run.yml').read_text())
    config['run']['name'] = 'unrelated_fresh'
    config['continuation'] = {'mode': 'fresh'}
    config = json.loads(normalize_config(RunConfig.model_validate(config)))
    child = copy.deepcopy(config)
    child['run']['name'] = 'unrelated_child'
    child['continuation'] = {'mode': 'resume', 'source_run': 'unrelated_fresh', 'checkpoint': 'final'}
    return {c['run']['name']: dict(config=c, path=f'configs/{i}.yml', queues={'fixture'})
            for i,c in enumerate([config,child])}


def registry(plan):
    runs, cps, evals = [], [], []
    for i, (name, entry) in enumerate(plan.items(), start=1):
        runs.append(dict(id=i, name=name, status='completed', source_checkpoint_id=None if i==1 else 11,
                         normalized_config_json=entry['config'], training_steps_completed=100,
                         training_steps_requested=100, data_epochs_completed=1))
        cps.append(dict(id=i+10, run_id=i, save_reason='final'))
        for scope in ('run_training', 'run_validation'):
            evals.append(dict(checkpoint_id=i+10, data_scope=scope, status='completed', trigger='final',
                              agent_return=999))
    return runs,cps,evals


def cells(frame, phases=range(2)):
    return grid_cells(frame, {a:list(frame[a].unique()) for a in ('architecture','gamma','lr','n_epochs','seed')}, phases)


def test_complete_path_uses_config_and_both_scopes(plan):
    frame,_ = coverage_rows(plan,*registry(plan))
    assert frame.phase.tolist() == [0,1]
    assert frame.ready.all()
    assert cells(frame).iloc[0]['state'] == 'complete'
    assert cells(frame).iloc[0]['done'] == 2
    assert 'agent_return' not in frame.columns


def test_pending_and_missing_grid_stages(plan):
    frame,_ = coverage_rows(plan,[],[],[])
    assert cells(frame).iloc[0]['state'] == 'not_started'
    result = cells(frame, range(3)).iloc[0]
    assert result['state'] == 'missing_config'
    assert result['missing_phases'] == 'v2'
    axes={a:list(frame[a].unique()) for a in ('architecture','gamma','lr','n_epochs','seed')}
    axes['lr'] = ['0.000000001']
    assert grid_cells(frame,axes,range(2)).iloc[0]['state'] == 'no_config'


@pytest.mark.parametrize('status,state', [('running','running'),('failed','issue'),('cancelled','issue'),('pending','partial')])
def test_execution_statuses(plan,status,state):
    runs,cps,evals=registry(plan)
    runs[1]['status']=status
    frame,_=coverage_rows(plan,runs,cps,evals)
    assert cells(frame).iloc[0]['state']==state
    assert cells(frame).iloc[0]['done']==1


@pytest.mark.parametrize('fault', ['evaluation','checkpoint','lineage','config','early_stop'])
def test_incomplete_or_mismatched_completed_run_is_not_ready(plan,fault):
    runs,cps,evals=registry(plan)
    if fault=='evaluation': evals.pop()
    if fault=='checkpoint': cps.pop()
    if fault=='lineage': runs[1]['source_checkpoint_id']=12
    if fault=='config':
        runs[1]['normalized_config_json']=copy.deepcopy(runs[1]['normalized_config_json'])
        runs[1]['normalized_config_json']['ppo']['learning_rate'] *= 2
    if fault=='early_stop': runs[1]['training_steps_completed']=50
    frame,_=coverage_rows(plan,runs,cps,evals)
    assert not frame.iloc[1]['ready']
    assert frame.iloc[1]['issues']
    assert cells(frame).iloc[0]['state']=='issue'


def test_settings_families_prevent_merging_costs_and_reward(plan):
    original=next(iter(plan.values()))['config']
    same=copy.deepcopy(original)
    same['ppo']['hidden_sizes']=[768]*3
    same['ppo']['learning_rate']=.000015
    same['run']['name']='different'
    assert family_settings(original)==family_settings(same)
    same['environment']['fee_bps']+=1
    assert family_settings(original)!=family_settings(same)


def test_duplicate_paths_and_cycle_are_visible(plan):
    duplicate=copy.deepcopy(plan['unrelated_fresh'])
    duplicate['config']['run']['name']='another'
    plan['another']=duplicate
    frame,_=coverage_rows(plan,[],[],[])
    assert frame.loc[frame.phase==0,'issues'].str.contains('Duplicate grid stage').all()
    plan['unrelated_fresh']['config']['continuation']={'mode':'resume','source_run':'unrelated_child','checkpoint':'final'}
    frame,_=coverage_rows(plan,[],[],[])
    assert frame.loc[frame.name=='unrelated_child','issues'].iloc[0].startswith('Cycle')


def test_plan_missing_parent_is_reported(plan):
    del plan['unrelated_fresh']
    frame,_=coverage_rows(plan,[],[],[])
    assert frame.phase.isna().all()
    assert 'outside selected plan' in frame.iloc[0]['issues']


def test_plan_loading_deduplicates_overlapping_queues_and_reports_missing_files(plan,tmp_path):
    root=tmp_path
    config=next(iter(plan.values()))['config']
    (root/'config.yml').write_text(yaml.safe_dump(config))
    a=root/'a.yml'; b=root/'b.yaml'
    a.write_text(yaml.safe_dump(dict(queue_schema_version=1, configs=['config.yml'])))
    b.write_text(yaml.safe_dump(dict(queue_schema_version=1, configs=['config.yml','absent.yml'])))
    entries,errors=read_plan(root,[a,b])
    assert len(entries)==1
    assert next(iter(entries.values()))['queues']=={'a','b'}
    assert len(errors)==1 and 'absent.yml' in errors[0]


def test_decimal_spellings_are_equal():
    assert decimal_text('0.90')==decimal_text('.9')
    assert decimal_text('0.000025000')==decimal_text('2.5e-5')


def test_figure_keeps_numeric_ids_and_exact_lr(plan):
    from train_and_eval.dashboard.coverage import coverage_figure
    frame,_=coverage_rows(plan,[],[],[])
    figure=coverage_figure(cells(frame))
    assert figure.data[0].customdata[0][3] == frame.iloc[0]['n_epochs']
    assert figure.data[0].text[0] == '0/2'
    assert len(figure.data[0].y[0].split('.')[1]) == 9
    figure.to_json()


def test_ui_handles_empty_database_and_axis_changes(plan,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from train_and_eval.dashboard import coverage
    qdir=tmp_path/'configs/search_queues';qdir.mkdir(parents=True)
    paths=[]
    for i,entry in enumerate(plan.values()):
        relative=f'config_{i}.yml';paths.append(relative)
        (tmp_path/relative).write_text(yaml.safe_dump(entry['config']))
    (qdir/'fixture.yml').write_text(yaml.safe_dump(dict(queue_schema_version=1,configs=paths)))
    monkeypatch.setattr(coverage,'load_registry',lambda: ([],[],[]))
    app=AppTest.from_string(
        'from pathlib import Path\nfrom train_and_eval.dashboard.coverage import render_grid_coverage\n'
        f'render_grid_coverage(Path({str(tmp_path)!r}))\n').run(timeout=20)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Ready stages').value == '0'
    lr_input=next(t for t in app.text_input if t.label.startswith('Expected LR'))
    lr_input.set_value('0.000000001').run(timeout=20)
    assert not app.exception
    assert any('no_config' in df.value.get('state',pd.Series(dtype=str)).values for df in app.dataframe)
