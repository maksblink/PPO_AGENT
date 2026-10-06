from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest

from train_and_eval.dashboard import data as dashboard
from train_and_eval.dashboard.lineage import architecture_label, run_path_metadata
from train_and_eval.dashboard.queues import queue_choices, queue_runs, queue_figure


@pytest.mark.parametrize('sizes, expected', [
    ([384, 384, 384], '384×3'), ([384]*4, '384×4'), ([768]*3, '768×3'),
    ([384, 192, 64], '384 → 192 → 64'), ([], 'Unknown'), ([True], 'Unknown'),
    (['384'], 'Unknown'), (None, 'Unknown'),
])
def test_architecture_uses_exact_layout(sizes, expected):
    assert architecture_label({'ppo': {'hidden_sizes': sizes}}) == expected


@pytest.fixture
def tables():
    # IDs and names deliberately do not encode chronological order or architecture.
    runs = pd.DataFrame([
        dict(id=i, name=f'unrelated_{i}', status='completed', seed=1,
             continuation_mode='fresh' if parent is None else 'resume',
             source_checkpoint_id=parent,
             normalized_config_json={'ppo': {'hidden_sizes': sizes, 'gamma': .9}})
        for i, parent, sizes in [
            (90, 20, [384]*3), (10, None, [384]*3), (40, 10, [384]*3),
            (30, 10, [384]*3), (80, None, [768]*3),
        ]
    ])
    checkpoints = pd.DataFrame([dict(id=cp, run_id=run) for cp, run in
                                [(10, 10), (20, 30), (30, 90), (40, 40), (80, 80)]])
    evaluations = pd.DataFrame([
        dict(id=cp + offset, checkpoint_id=cp, data_scope=scope,
             status='completed', trigger='final', agent_return=cp*scale,
             agent_max_drawdown=-cp/1000, balanced_score=cp*scale-cp/1000,
             market_exposure=.5, round_trips=cp)
        for scope, offset, scale in [('run_validation', 0, .01), ('run_training', 100, -.01)]
        for cp in (10, 20, 30, 40, 80)
    ])
    return dict(runs=runs, checkpoints=checkpoints, evaluations=evaluations,
                training_metrics=pd.DataFrame())


def test_lineage_order_branches_and_scopes(tables):
    for scope in ('run_training', 'run_validation'):
        frame = dashboard._build_explorer(**tables, data_scope=scope)
        meta = frame.set_index('run_id')
        assert meta.loc[90, 'path.root_run_id'] == 10
        assert meta.loc[90, 'path.position'] == 3
        assert meta.loc[40, 'path.position'] == 2
        assert meta.loc[80, 'path.root_run_id'] == 80
        assert queue_runs(frame, 10).run_id.tolist() == [10, 30, 40, 90]
        assert queue_choices(frame, "unrelated_90") == [10]
        assert frame['eval.data_scope'].eq(scope).all()


def test_missing_and_cyclic_ancestors_are_explicit(tables):
    runs = tables['runs'].copy()
    runs.loc[runs.id == 10, 'source_checkpoint_id'] = 30  # 10 -> 90 -> 30 -> 10
    runs.loc[runs.id == 80, 'source_checkpoint_id'] = 999
    meta = run_path_metadata(runs, tables['checkpoints']).set_index('run_id')
    assert meta.loc[80, 'path.status'] == 'Missing ancestor'
    assert meta.loc[[10, 30, 40, 90], 'path.status'].eq('Cyclic lineage').all()
    assert meta['path.position'].isna().all()
    assert meta['path.root_run_id'].isna().all()


def test_resume_without_source_is_not_reported_as_fresh(tables):
    runs = tables['runs'].copy()
    runs.loc[runs.id == 10, 'continuation_mode'] = 'resume'
    meta = run_path_metadata(runs, tables['checkpoints']).set_index('run_id')
    assert meta.loc[90, 'path.status'] == 'Missing ancestor'


def test_intermediate_source_not_connected_to_final_evaluation(tables):
    tables['checkpoints'] = pd.concat([tables['checkpoints'],
        pd.DataFrame([dict(id=11, run_id=10)])], ignore_index=True)
    tables['runs'].loc[tables['runs'].id == 30, 'source_checkpoint_id'] = 11
    frame = dashboard._build_explorer(**tables)
    assert frame.set_index('run_id').loc[30, 'path.root_run_id'] == 10
    fig = queue_figure(queue_runs(frame, 10), "eval.agent_return", label="Return")
    assert list(fig.data[0].y) == [.1, .4, None, .2, .3, None]



def test_queue_search_matches_descendants_and_not_regex(tables):
    frame = dashboard._build_explorer(**tables)
    assert queue_choices(frame) == [10, 80]
    assert queue_choices(frame, '#90') == [10]
    assert queue_choices(frame, 'UNRELATED_80') == [80]
    assert queue_choices(frame, '.*') == []
    assert queue_choices(frame, 'nonexistent') == []


def test_chart_uses_ancestry_and_does_not_connect_siblings(tables):
    frame = queue_runs(dashboard._build_explorer(**tables), 10)
    fig = queue_figure(frame, 'eval.agent_return', label='Return', percent=True, focus_run_id=90)
    assert list(fig.data[0].x) == [1, 2, None, 1, 2, None, 2, 3, None]
    assert list(fig.data[0].y) == [.1, .2, None, .1, .4, None, .2, .3, None]
    assert fig.data[1].text == ('#10', '#30', '#40', '#90')
    assert fig.data[1].marker.size[-1] == 14
    assert fig.layout.yaxis.tickformat == '.5%'
    frame.loc[frame.run_id == 30, 'eval.agent_return'] = float('nan')
    fig = queue_figure(frame, 'eval.agent_return', label='Return')
    assert list(fig.data[0].y) == [.1, .4, None]
    assert fig.data[1].text == ('#10', '#40', '#90')


def test_single_run_and_non_evaluation_metric(tables):
    frame = dashboard._build_explorer(**tables)
    fig = queue_figure(queue_runs(frame, 80), 'run.seed', label='Seed')
    assert len(fig.data) == 1 and fig.data[0].text == ('#80',)
    members = queue_runs(frame, 10)
    fig = queue_figure(members, 'run.seed', label='Seed')
    assert len(fig.data) == 2


def test_run_detail_button_navigates_to_full_queue(monkeypatch, tables):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    monkeypatch.setattr(dashboard, 'create_database_engine',
                        lambda: SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(dashboard, '_read_table', lambda engine, name: tables[name].copy())
    app = Path(__file__).resolve().parents[1] / 'train_and_eval/dashboard/app.py'
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert not any(m.label.startswith('Training paths') for m in at.multiselect)
    at.segmented_control(key='dashboard_view').set_value('Run Detail').run()
    run_selector = next(s for s in at.selectbox if s.label == 'Run')
    run_selector.set_value('#90 — unrelated_90').run()
    at.button(key='show_queue').click().run()
    assert not at.exception
    assert at.segmented_control(key='dashboard_view').value == 'Queues'
    assert at.selectbox(key='queue_root').value == 10
    assert next(m for m in at.metric if m.label == 'Runs in queue').value == '4'
    assert any('highlighted' in c.value for c in at.caption)
    # Queues ignores even an empty architecture selection.
    at.multiselect(key='architecture_filter').set_value([]).run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == 'Runs in queue').value == '4'
    at.segmented_control(key='evaluation_data_mode').set_value('TRAIN').run()
    assert not at.exception
    import json
    spec = json.loads(at.get('plotly_chart')[0].proto.spec)
    assert spec['data'][1]['y'] == [-.1, -.2, -.4, -.3]
    at.selectbox(key='queue_metric').set_value('eval.round_trips').run()
    assert not at.exception
    at.text_input(key='queue_search').set_value('unrelated_80').run()
    assert at.selectbox(key='queue_root').value == 80
    assert next(m for m in at.metric if m.label == 'Runs in queue').value == '1'
    at.text_input(key='queue_search').set_value('nonexistent').run()
    assert not at.exception
    assert any('No training paths' in m.value for m in at.info)
    st.cache_data.clear()


def test_queues_retains_failed_runs_and_missing_evaluations(tables):
    tables['runs'].loc[tables['runs'].id == 30, 'status'] = 'failed'
    tables['evaluations'] = tables['evaluations'].loc[tables['evaluations'].checkpoint_id != 20]
    members = queue_runs(dashboard._build_explorer(**tables), 10)
    assert members.run_id.tolist() == [10, 30, 40, 90]
    assert members.loc[members.run_id == 30, 'eval.agent_return'].isna().all()
    fig = queue_figure(members, 'eval.agent_return', label='Return')
    assert list(fig.data[0].y) == [.1, .4, None]


def test_every_numeric_explorer_column_can_be_plotted(tables):
    members = queue_runs(dashboard._build_explorer(**tables), 10)
    for metric in members.select_dtypes(include='number').columns:
        figure = queue_figure(members, metric, label=metric)
        assert len(figure.data[-1].y) == members[metric].notna().sum()
        figure.to_json()
