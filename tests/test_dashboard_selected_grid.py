"""Selected queue coverage must never become an inferred Cartesian grid."""
import copy
import json
from pathlib import Path
import yaml
from train_and_eval.dashboard.coverage import coverage_rows, grid_cells, read_plan, AXES
from train_and_eval.run_config import RunConfig, normalize_config


def config(name, **changes):
    c = yaml.safe_load((Path(__file__).parent/'fixtures/temporal_run.yml').read_text())
    c['run']['name'] = name
    c['continuation'] = {'mode': 'fresh'}
    c['ppo'].update(changes)
    return json.loads(normalize_config(RunConfig.model_validate(c)))


def entries(*configs):
    return {c['run']['name']: dict(config=c, path='config.yml', queues={'selected'}) for c in configs}


def test_sparse_grid_does_not_invent_combinations_or_stages():
    a = config('a', gamma=.9, learning_rate=.00002)
    b = config('b', gamma=.99, learning_rate=.00004)
    child = copy.deepcopy(b)
    child['run']['name'] = 'b_child'
    child['continuation'] = dict(mode='resume', source_run='b', checkpoint='final')
    frame, _ = coverage_rows(entries(a, b, child), [], [], [])
    cells = grid_cells(frame)
    assert len(cells) == 2  # not two gammas times two learning rates
    assert sorted(cells.total) == [1, 2]
    assert cells.total.sum() == 3
    assert set(cells.state) == {'not_started'}


def test_entropy_and_gae_are_coordinates_not_families():
    configs = [config(f'r{i}', ent_coef=ent, gae_lambda=gae)
               for i, (ent, gae) in enumerate([(.0002,.85),(.002,.85),(.0002,.95),(.002,.95)])]
    frame, families = coverage_rows(entries(*configs), [], [], [])
    assert len(families) == 1
    assert len(grid_cells(frame)) == 4
    assert not frame.issues.any()
    assert {'ent_coef', 'gae_lambda'} <= set(AXES)


def test_outside_database_run_cannot_add_expected_cells():
    c = config('selected')
    other = config('outside', gamma=.99, ent_coef=.002)
    runs = [dict(id=99,name='outside',status='completed', normalized_config_json=other)]
    frame, _ = coverage_rows(entries(c), runs, [], [])
    assert frame.name.tolist() == ['selected']
    assert len(grid_cells(frame)) == 1
    assert grid_cells(frame).total.sum() == 1


def test_shared_run_counts_once_with_selected_physical_copy(tmp_path):
    c = config('shared')
    for folder in ('grid1','grid3'):
        directory = tmp_path / folder
        directory.mkdir()
        (directory/'config.yml').write_text(yaml.safe_dump(c))
        (directory/'queue.yml').write_text(yaml.safe_dump(dict(queue_schema_version=1,configs=[f'{folder}/config.yml'])))
    plan, errors = read_plan(tmp_path, [tmp_path/'grid3/queue.yml'])
    assert not errors and len(plan) == 1
    run = dict(id=1, name='shared', status='completed', source_checkpoint_id=None, normalized_config_json=c,
               training_steps_completed=100, training_steps_requested=100)
    cp = dict(id=2, run_id=1, save_reason='final')
    evaluations = [dict(checkpoint_id=2, data_scope=scope, trigger='final', status='completed')
                   for scope in ('run_training','run_validation')]
    frame, _ = coverage_rows(plan, [run], [cp], evaluations)
    assert frame.ready.all()
    assert grid_cells(frame).iloc[0].state == 'complete'
    assert frame.iloc[0].config_path == 'grid3/config.yml'


def test_parent_variant_change_is_an_issue():
    a = config('parent')
    b = copy.deepcopy(a)
    b['run']['name'] = 'child'
    b['ppo']['gae_lambda'] = .95
    b['continuation'] = dict(mode='resume', source_run='parent', checkpoint='final')
    frame, _ = coverage_rows(entries(a,b), [],[],[])
    assert 'different grid coordinates' in frame.iloc[1].issues


def test_switching_queues_resets_scope_and_variant_filters(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from train_and_eval.dashboard import coverage
    qdir = tmp_path/'configs/search_queues'; qdir.mkdir(parents=True)
    for name, ent, count in [('grid1_s1',.0002,1), ('grid3_s1',.002,2)]:
        paths = []
        for i in range(count):
            relative = f'{name}_{i}.yml'; paths.append(relative)
            (tmp_path/relative).write_text(yaml.safe_dump(config(f'{name}_{i}',ent_coef=ent, gamma=[.9,.99][i])))
        (qdir/f'{name}.yml').write_text(yaml.safe_dump(dict(queue_schema_version=1,configs=paths)))
    monkeypatch.setattr(coverage, 'load_registry', lambda: ([],[],[]))
    app = AppTest.from_string('from pathlib import Path\nfrom train_and_eval.dashboard.coverage import render_grid_coverage\n'
                             f'render_grid_coverage(Path({str(tmp_path)!r}))').run(timeout=20)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Planned stages').value == '1'
    app.selectbox(key='coverage_queue').select(qdir/'grid3_s1.yml').run(timeout=20)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Planned stages').value == '2'
    assert next(s for s in app.selectbox if s.label=='Entropy coefficient').value == '0.002'
    app.selectbox(key='coverage_queue').select(qdir/'grid1_s1.yml').run(timeout=20)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Planned stages').value == '1'


def test_unreadable_plan_cannot_display_successful_coverage(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from train_and_eval.dashboard import coverage
    qdir = tmp_path/'configs/search_queues'; qdir.mkdir(parents=True)
    (qdir/'broken.yml').write_text(yaml.safe_dump(dict(queue_schema_version=1, configs=['missing.yml'])))
    def unexpected_db_access():
        raise AssertionError('Invalid plan must stop before DB access')
    monkeypatch.setattr(coverage, 'load_registry', unexpected_db_access)
    app = AppTest.from_string('from pathlib import Path\nfrom train_and_eval.dashboard.coverage import render_grid_coverage\n'
                             f'render_grid_coverage(Path({str(tmp_path)!r}))').run(timeout=20)
    assert not app.exception
    assert app.error
    assert not app.metric


def test_analysis_exposes_entropy_and_gae_by_default():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import pandas as pd
from train_and_eval.dashboard.analysis import render_grid_analysis
frame = pd.DataFrame({
 'run_id':[1,2], 'run.name':['a','b'], 'run.architecture':['384×3']*2,
 'run.status':['completed']*2, 'run.seed':[1,1],
 'path.root_run_id':[1,2], 'path.parent_run_id':[None,None],
 'path.position':[1,1], 'path.status':['Complete']*2,
 'cfg.normalized_config_json.ppo.ent_coef':[.0002,.002],
 'cfg.normalized_config_json.ppo.gae_lambda':[.85,.95],
 'eval.id':[1,2], 'eval.agent_return':[.1,.2]})
render_grid_analysis(frame, 'VAL')
''').run(timeout=20)
    assert not app.exception
    for suffix in ('ent_coef', 'gae_lambda'):
        key = 'cfg.normalized_config_json.ppo.' + suffix
        assert key in app.multiselect(key='ga_filter_fields').value
        assert key in app.multiselect(key='ga_groups').value
