from pathlib import Path
from unittest.mock import patch
import runpy
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from train_and_eval.dashboard.analysis_display import (
    is_percent_metric, metric_text, statistics_display, runs_display, distribution_figure,
)
from train_and_eval.dashboard.analysis import evaluation_metrics


@pytest.mark.parametrize('name', ['agent_return','always_short_return','agent_max_drawdown',
    'market_exposure','long_exposure','win_rate','short_win_rate','total_cost_return',
    'drawdown_improvement','open_position_return_at_end'])
def test_percent_metrics(name):
    assert is_percent_metric('eval.'+name)


@pytest.mark.parametrize('name', ['eval.balanced_score','eval.profit_factor','eval.round_trips',
    'cfg.normalized_config_json.ppo.gamma','cfg.normalized_config_json.ppo.learning_rate','train.approx_kl'])
def test_raw_metrics(name):
    assert not is_percent_metric(name)


def test_units_and_storage_values_unchanged():
    raw=pd.DataFrame({'metric':['eval.agent_return','eval.profit_factor'],
        'min':[-2.1,.5],'max':[.1,2.], 'mean':[-1.,1.25],'median':[-1.,1.25],
        'std':[.2,.4],'N':[2,2]})
    original=raw.copy(deep=True)
    shown=statistics_display(raw)
    assert shown['min'].tolist()==[-210.,.5]
    assert shown['std'].tolist()==[20.,.4]
    assert shown['unit'].tolist()==['%','raw']
    assert shown['std_unit'].tolist()==['pp','raw']
    pd.testing.assert_frame_equal(raw,original)
    assert metric_text('eval.agent_return',-2.1)=='-210.00000%'
    assert metric_text('eval.agent_return',np.nan)=='n/a'


def test_matching_runs_units_and_count():
    frame=pd.DataFrame({'eval.agent_return':[.15],'eval.round_trips':[7]})
    shown=runs_display(frame)
    assert shown['eval.agent_return (%)'].iloc[0]==15
    assert shown['eval.round_trips'].iloc[0]==7
    assert frame['eval.agent_return'].iloc[0]==.15


def test_box_keeps_outliers_and_formats_categorical_groups():
    frame=pd.DataFrame({'run_id':[1,2,3], 'gamma':[.9,.9,.95], 'eval.agent_return':[-2.1,.1,.2]})
    fig=distribution_figure(frame,'eval.agent_return',['gamma'],'Return')
    assert fig.layout.xaxis.type=='category'
    assert fig.layout.yaxis.tickformat=='.5%'
    assert fig.data[0].type=='box' and fig.data[0].boxpoints=='all'
    assert min(fig.data[0].y)==-2.1
    assert '%{y:.5%}' in fig.data[0].hovertemplate
    raw=distribution_figure(frame.rename(columns={'eval.agent_return':'eval.round_trips'}),'eval.round_trips',[],'Trips')
    assert raw.layout.yaxis.tickformat is None


def test_ppo_diagnostics_remain_available():
    f=pd.DataFrame({'train.approx_kl':[.1],'train.clip_fraction':[.2],'train.id':[1]})
    assert evaluation_metrics(f)==['train.approx_kl','train.clip_fraction']
    assert is_percent_metric('train.clip_fraction')


def test_coverage_has_no_scope_selector_or_database_load():
    from train_and_eval.dashboard import data, coverage
    app_path=Path(data.__file__).with_name('app.py')
    script=f'''import streamlit as st\nimport runpy\nst.session_state['dashboard_view']='Grid Coverage'\nrunpy.run_path({str(app_path)!r},run_name='__main__')'''
    with patch.object(coverage,'render_grid_coverage') as render, patch.object(data,'load_dashboard_data') as load:
        app=AppTest.from_string(script).run()
        assert not app.exception
        render.assert_called_once()
        load.assert_not_called()
        assert 'evaluation_data_mode' not in app.session_state
