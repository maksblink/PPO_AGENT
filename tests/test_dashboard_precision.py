from decimal import Decimal
import pandas as pd
import plotly.express as px
from train_and_eval.dashboard.formatting import decimal_text, value_text, styled_frame, precise_figure
from train_and_eval.dashboard.analysis_display import distribution_figure, statistics_display


def test_parameters_not_rounded_or_scientific():
    for value, expected in [(0.000015, '0.000015'), (1e-12, '0.000000000001'),
                            (0.123456789123, '0.123456789123'), (3, '3'),
                            (Decimal('0.000000000000000000123'), '0.000000000000000000123')]:
        assert decimal_text(value) == expected
        assert value_text('cfg.ppo.learning_rate', value) == expected


def test_percent_units_no_double_scaling():
    assert value_text('eval.agent_return', .123456789) == '12.34568%'
    assert value_text('Always-short return', -.1) == '-10.00000%'
    assert value_text('train.clip_fraction', .00001) == '0.00100%'
    assert value_text('Agent return (%)', 12.3456789) == '12.34568%'
    assert value_text('eval.balanced_score', .123456789) == '0.123456789'
    assert value_text('run_id', 3) == '3'
    assert value_text('eval.agent_return', None) == '—'


def test_styler_keeps_numeric_sort_keys_and_source():
    frame = pd.DataFrame({'Learning rate': [.000025, .000015], 'Agent return': [.1, -.2]})
    before = frame.copy(deep=True)
    styled = styled_frame(frame)
    pd.testing.assert_frame_equal(styled.data, before)
    assert styled.data.sort_values('Learning rate').index.tolist() == [1, 0]
    html = styled.to_html()
    assert '0.000015' in html and '10.00000%' in html
    pd.testing.assert_frame_equal(frame, before)


def test_statistics_five_places_only_on_percentage_rows():
    frame = pd.DataFrame({'metric': ['eval.agent_return', 'eval.profit_factor'],
                          'min': [.123456789, 1.123456789]})
    # Provide the complete statistics schema.
    for col in ('max', 'mean', 'median', 'std'):
        frame[col] = frame['min']
    displayed = statistics_display(frame)
    html = styled_frame(displayed).to_html()
    assert '12.34568' in html and '1.123456789' in html
    assert displayed['min'].iloc[0] == frame['min'].iloc[0] * 100


def test_hover_parameters_and_percent_numeric_coordinates():
    frame = pd.DataFrame({'Learning rate': [.000015, .000025], 'Agent return': [.123456789, .2],
                          'gamma': [.9, .99]})
    fig = precise_figure(px.scatter(frame, x='Learning rate', y='Agent return', hover_data=['gamma']))
    assert list(fig.data[0].x) == frame['Learning rate'].tolist()
    assert '%{x:.6f}' in fig.data[0].hovertemplate
    assert '%{y:.5%}' in fig.data[0].hovertemplate
    assert fig.layout.xaxis.tickformat == '.6f'
    assert fig.layout.yaxis.tickformat == '.5%'


def test_box_group_lr_full_decimal():
    frame = pd.DataFrame({'run_id': [1, 2], 'cfg.ppo.learning_rate': [.000015, .000025],
                          'eval.agent_return': [.1, .2]})
    fig = distribution_figure(frame, 'eval.agent_return', ['cfg.ppo.learning_rate'], 'Return')
    assert list(fig.data[0].x) == ['0.000015', '0.000025']
    assert fig.layout.yaxis.tickformat == '.5%'
