"""Presentation-only numeric formatting; source frames retain numeric dtypes."""
from decimal import Decimal
from numbers import Real
import re

import numpy as np
import pandas as pd
import streamlit as st


def decimal_text(value):
    """Expand the stored scalar's decimal representation without rounding it."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return '—'
    if isinstance(value, (Real, Decimal)) and not isinstance(value, (bool, np.bool_)):
        text = format(Decimal(str(value)), 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    return str(value)


def is_percent(column):
    name = str(column or '').lower().replace('-', '_').replace(' ', '_').split('.')[-1]
    if name.endswith('(%)') or name == 'share_percent':
        return False  # Already scaled to 0..100 by the analysis layer.
    return (name.endswith(('_return', '_max_drawdown', '_exposure', '_win_rate'))
            or name in {'return', 'max_drawdown', 'win_rate', 'loss_rate', 'breakeven_rate',
                        'trade_event_rate', 'round_trip_rate', 'drawdown_improvement',
                        'open_position_return_at_end', 'clip_fraction'})


def value_text(column, value):
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return '—'
    if isinstance(value, (Real, Decimal)) and not isinstance(value, (bool, np.bool_)):
        if is_percent(column):
            return f'{value:.5%}'
        if str(column).endswith('(%)') or column == 'share_percent':
            return f'{value:.5f}%'
    return decimal_text(value)


def styled_frame(frame):
    """Styler display strings do not replace numeric Arrow cells or sort keys."""
    style = frame.style.format({c: lambda v, c=c: value_text(c, v) for c in frame.columns})
    # Analysis statistics contain differently scaled metrics in different rows.
    if 'unit' in frame:
        for unit in ('%', 'raw'):
            rows = frame.index[frame['unit'].eq(unit)]
            columns = [c for c in ('min', 'max', 'mean', 'median', 'std') if c in frame]
            if len(rows) and columns:
                formatter = (lambda v: '—' if pd.isna(v) else f'{v:.5f}') if unit == '%' else decimal_text
                style = style.format(formatter, subset=pd.IndexSlice[rows, columns])
    return style


def dataframe(frame, **kwargs):
    return st.dataframe(styled_frame(frame), **kwargs)


def decimals(values):
    result = 0
    for value in values:
        if isinstance(value, (Real, Decimal)) and not isinstance(value, (bool, np.bool_)) and np.isfinite(float(value)):
            result = max(result, max(0, -Decimal(str(value)).normalize().as_tuple().exponent))
    return result


def precise_figure(figure):
    """Keep coordinates numeric; format hover values at their stored precision."""
    placeholder = re.compile(r'%\{(x|y|z|customdata\[(\d+)\])(?::([^}]+))?\}')
    for trace in figure.data:
        template = trace.hovertemplate
        if not isinstance(template, str):
            continue
        def replace(match):
            field, index, spec = match.groups()
            if index is not None:
                values = [row[int(index)] for row in trace.customdata] if trace.customdata is not None else []
            else:
                data = getattr(trace, field, None)
                values = list(data) if data is not None else []
            axis = getattr(figure.layout, field + 'axis', None) if field in ('x', 'y') else None
            percent_axis = axis is not None and (is_percent(axis.title.text) or '%' in (axis.tickformat or ''))
            if (spec and '%' in spec) or percent_axis or (field == 'y' and is_percent(trace.name)):
                return '%{' + field + ':.5%}'
            if values and all(isinstance(v, (Real, Decimal)) and not isinstance(v, (bool, np.bool_)) for v in values):
                return '%{' + field + ':.' + str(decimals(values)) + 'f}'
            return match.group(0)
        trace.hovertemplate = placeholder.sub(replace, template)
    for axis_name in figure.layout:
        if not re.fullmatch(r'[xy]axis\d*', axis_name):
            continue
        axis = figure.layout[axis_name]
        title = axis.title.text or ''
        if (axis.tickformat and '%' in axis.tickformat) or is_percent(title):
            axis.tickformat = '.5%'
            axis.hoverformat = '.5%'
        elif any(word in title.lower() for word in ('learning', 'gamma', 'lambda', 'coefficient', 'penalty', 'ppo.', 'environment.')):
            coordinate = axis_name[0]
            values = []
            for trace in figure.data:
                data = getattr(trace, coordinate, None)
                if data is not None:
                    values.extend(data)
            axis.tickformat = f'.{decimals(values)}f'
            axis.hoverformat = axis.tickformat
            axis.exponentformat = 'none'
    for key in figure.layout:
        if not re.fullmatch(r'coloraxis\d*', key):
            continue
        bar = figure.layout[key].colorbar
        title = bar.title.text or ''
        if is_percent(title) or '%' in (bar.tickformat or ''):
            bar.tickformat = '.5%'
        elif any(word in title.lower() for word in ('learning', 'gamma', 'lambda', 'coefficient', 'penalty')):
            values = []
            for trace in figure.data:
                marker = getattr(trace, 'marker', None)
                if marker is not None and marker.coloraxis == key and marker.color is not None:
                    values.extend(marker.color)
            bar.tickformat = f'.{decimals(values)}f'
            bar.exponentformat = 'none'
    return figure


def plotly_chart(figure, **kwargs):
    return st.plotly_chart(precise_figure(figure), **kwargs)
