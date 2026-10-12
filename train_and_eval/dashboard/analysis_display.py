"""Display units for grid analysis; stored values and statistics remain unchanged."""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
import re
import numpy as np
import pandas as pd
import plotly.express as px
from train_and_eval.dashboard.formatting import decimal_text, is_percent


def is_percent_metric(column: str | None) -> bool:
    return is_percent(column)


def metric_text(column: str, value: object) -> str:
    if pd.isna(value):
        return 'n/a'
    return f'{value:.5%}' if is_percent_metric(column) else decimal_text(value)


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


def group_sort_key(value):
    """Numeric order for numbers; natural text order; missing values last."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return (2,)
    text = decimal_text(value)
    try:
        number = Decimal(text)
        if number.is_finite():
            return (0, number)
    except InvalidOperation:
        pass
    parts = tuple((1, int(p)) if p.isdigit() else (0, p.casefold())
                  for p in re.split(r'(\d+)', text) if p)
    return (1, parts, text)


def distribution_figure(frame: pd.DataFrame, metric: str, grouping: list[str], label: str):
    chart = frame.copy()
    chart[metric] = pd.to_numeric(chart[metric], errors='coerce').replace([np.inf, -np.inf], np.nan)
    chart['Group'] = chart[grouping].map(decimal_text).agg(' · '.join, axis=1) if grouping else 'All selected'
    if grouping:
        ordered = sorted(chart[grouping].itertuples(index=False, name=None),
                         key=lambda row: tuple(group_sort_key(value) for value in row))
        categories = list(dict.fromkeys(' · '.join(decimal_text(v) for v in row)
                                        for row in ordered))
    else:
        categories = ['All selected']
    percent = is_percent_metric(metric)
    figure = px.box(chart, x='Group', y=metric, points='all', hover_data=['run_id'],
                    labels={metric: label + (' (%)' if percent else '')})
    # Numeric-looking group labels (e.g. gamma) must remain discrete categories.
    figure.update_xaxes(type='category', categoryorder='array', categoryarray=categories)
    if percent:
        figure.update_yaxes(tickformat='.5%')
        figure.update_traces(hovertemplate='Group=%{x}<br>'+label+'=%{y:.5%}<br>Run ID=%{customdata[0]}<extra></extra>',
                             yhoverformat='.5%')
    return figure
