"""Display units for grid analysis; stored values and statistics remain unchanged."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px


def is_percent_metric(column: str | None) -> bool:
    if not column:
        return False
    if column == 'train.clip_fraction':
        return True
    if not column.startswith('eval.'):
        return False
    name = column.removeprefix('eval.')
    return (name.endswith(('_return', '_max_drawdown', '_exposure', '_win_rate'))
            or name in {'win_rate', 'loss_rate', 'breakeven_rate', 'trade_event_rate',
                        'round_trip_rate', 'drawdown_improvement', 'open_position_return_at_end'})


def metric_text(column: str, value: object) -> str:
    if pd.isna(value):
        return 'n/a'
    return f'{value:.2%}' if is_percent_metric(column) else f'{value:.6g}'


def statistics_display(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    percent = result['metric'].map(is_percent_metric)
    for field in ('min', 'max', 'mean', 'median', 'std'):
        result.loc[percent, field] = result.loc[percent, field] * 100
    result['unit'] = np.where(percent, '%', 'raw')
    result['std_unit'] = np.where(percent, 'pp', 'raw')
    return result


def runs_display(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in frame:
        if is_percent_metric(column):
            result[column] = pd.to_numeric(frame[column], errors='coerce') * 100
    return result.rename(columns={c: f'{c} (%)' for c in frame if is_percent_metric(c)})


def distribution_figure(frame: pd.DataFrame, metric: str, grouping: list[str], label: str):
    chart = frame.copy()
    chart[metric] = pd.to_numeric(chart[metric], errors='coerce').replace([np.inf, -np.inf], np.nan)
    chart['Group'] = chart[grouping].astype(str).agg(' · '.join, axis=1) if grouping else 'All selected'
    percent = is_percent_metric(metric)
    figure = px.box(chart, x='Group', y=metric, points='all', hover_data=['run_id'],
                    labels={metric: label + (' (%)' if percent else '')})
    # Numeric-looking group labels (e.g. gamma) must remain discrete categories.
    figure.update_xaxes(type='category')
    if percent:
        figure.update_yaxes(tickformat='.1%')
        figure.update_traces(hovertemplate='Group=%{x}<br>'+label+'=%{y:.2%}<br>Run ID=%{customdata[0]}<extra></extra>',
                             yhoverformat='.2%')
    return figure
