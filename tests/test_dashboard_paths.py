from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest

from train_and_eval.dashboard import data as dashboard
from train_and_eval.dashboard.paths import (
    architecture_label, run_path_metadata, visible_path_edges, add_path_connections,
)


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
        assert visible_path_edges(frame) == [(10, 30), (10, 40), (30, 90)]
        # Filtering out an intermediate ancestor must not bridge the gap.
        assert visible_path_edges(frame[frame.run_id.isin([10, 90])]) == []
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
    assert visible_path_edges(frame) == [(10, 40), (30, 90)]


def test_plot_arrows_follow_parent_child_not_metric_sort(tables):
    frame = dashboard._build_explorer(**tables)
    figure = go.Figure()
    add_path_connections(figure, frame, x='eval.agent_max_drawdown', y='eval.agent_return')
    assert len(figure.data) == 1
    assert figure.data[0].line.dash == 'dot'
    assert figure.data[0].connectgaps is False
    assert list(figure.data[0].y) == [.1, .2, None, .1, .4, None, .2, .3, None]
    assert len(figure.layout.annotations) == 3
    assert figure.layout.annotations[2].y == .3
    figure = go.Figure()
    frame.loc[frame.run_id == 30, 'eval.agent_return'] = float('nan')
    add_path_connections(figure, frame, x='eval.agent_max_drawdown', y='eval.agent_return')
    assert list(figure.data[0].y) == [.1, .4, None]


def test_architecture_filter_paths_and_scope_switch_in_app(monkeypatch, tables):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    monkeypatch.setattr(dashboard, 'create_database_engine',
                        lambda: SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(dashboard, '_read_table', lambda engine, name: tables[name].copy())
    app = Path(__file__).resolve().parents[1] / 'train_and_eval/dashboard/app.py'
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    at.multiselect(key='architecture_filter').set_value(['384×3']).run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == 'Visible runs').value == '4'
    at.checkbox(key='pareto_tab_connect_paths').check().run()
    at.checkbox(key='scatter_tab_connect_paths').check().run()
    assert not at.exception
    # Plot specs expose the overlay without needing a running browser/server.
    import json
    specs = [json.loads(el.proto.spec) for el in at.get('plotly_chart')]
    assert sum(any(t.get('name') == 'Checkpoint lineage →' for t in s['data']) for s in specs) == 2
    at.segmented_control[0].set_value('TRAIN').run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == 'Best return').value == '-10.00%'
    at.multiselect(key='path_filter').set_value([10]).run()
    assert next(m for m in at.metric if m.label == 'Visible runs').value == '4'
    at.multiselect(key='architecture_filter').set_value([]).run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == 'Visible runs').value == '0'
    st.cache_data.clear()
