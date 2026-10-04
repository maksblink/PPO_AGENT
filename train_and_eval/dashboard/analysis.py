"""Read-only descriptive statistics of final evaluations, one row per run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from train_and_eval.dashboard.analysis_display import (
    statistics_display, runs_display, metric_text, distribution_figure,
)


def prepare_analysis(explorer: pd.DataFrame) -> pd.DataFrame:
    frame = explorer.copy()
    # Resolve endpoints against the full database before any UI filtering.
    parents = set(frame.get('path.parent_run_id', pd.Series(dtype=float)).dropna())
    complete = frame.get('path.status', pd.Series('', index=frame.index)).eq('Complete')
    frame['analysis.stage'] = pd.to_numeric(frame.get('path.position'), errors='coerce') - 1
    frame['analysis.latest_existing'] = complete & ~frame['run_id'].isin(parents)
    return frame


def filter_rows(frame: pd.DataFrame, selections: dict, ranges: dict,
                stages: list[int] | None = None, latest: bool = False) -> pd.DataFrame:
    result = frame.copy()
    for column, values in selections.items():
        result = result[result[column].isin(values)]
    for column, (low, high) in ranges.items():
        if low > high:
            raise ValueError(f'{column}: lower bound exceeds upper bound')
        values = pd.to_numeric(result[column], errors='coerce')
        result = result[values.between(low, high, inclusive='both')]
    if stages is not None:
        result = result[result['analysis.stage'].isin(stages)]
    if latest:
        result = result[result['analysis.latest_existing']]
    return result


def evaluation_metrics(frame: pd.DataFrame) -> list[str]:
    exclude = {'id', 'checkpoint_id', 'run_id'}
    return [c for c in frame if c.startswith(('eval.', 'train.')) and c.split('.', 1)[1] not in exclude
            and pd.api.types.is_numeric_dtype(frame[c])
            and not pd.api.types.is_bool_dtype(frame[c])]


def dimension_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if (c.startswith('cfg.') or c in
            {'run.architecture', 'run.seed', 'run.status', 'analysis.stage',
             'path.root_run_id', 'run.continuation_mode'})
            and not frame[c].dropna().empty
            and frame[c].dropna().map(lambda v: isinstance(v, (str, int, float, bool, np.number))).all()]


def summarize(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame['run_id'].duplicated().any():
        raise ValueError('Expected one final evaluation row per run')
    if not metrics or frame.empty:
        return pd.DataFrame()
    batches = frame.groupby(groups, dropna=False, observed=True, sort=True) if groups else [((), frame)]
    records = []
    for key, group in batches:
        keys = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(groups, keys))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            records.append({**labels, 'metric': metric, 'runs': len(group),
                'paths': group['path.root_run_id'].nunique(), 'N': len(values),
                'missing_or_nonfinite': len(group)-len(values),
                'min': values.min(), 'max': values.max(), 'mean': values.mean(),
                'median': values.median(), 'std': values.std(ddof=1)})
    return pd.DataFrame(records)


def render_grid_analysis(explorer: pd.DataFrame, scope: str) -> None:
    import streamlit as st

    st.subheader(f'Grid Analysis · {scope}')
    st.caption('Final evaluations only; one record per run. Filters below are independent of the sidebar. '
               'Mean and average are the same statistic; median is shown separately. '
               'Return, drawdown, exposure and success rates are displayed as percentages. '
               'PPO diagnostics use the latest update and are shared between TRAIN and VAL.')
    frame = prepare_analysis(explorer)
    dimensions = dimension_columns(frame)
    # Keep configuration and database fields distinct in widget option labels.
    label = lambda c: ('config.' + c.removeprefix('cfg.normalized_config_json.')) if c.startswith('cfg.normalized_config_json.') else c
    with st.expander('Filters', expanded=True):
        defaults = [c for c in dimensions if c == 'run.architecture' or
                    c.endswith(('ppo.learning_rate', 'ppo.gamma', 'ppo.n_epochs', 'run.seed'))]
        fields = st.multiselect('Filter parameters', dimensions, default=defaults,
                                format_func=label, key='ga_filter_fields')
        selections, ranges = {}, {}
        for column in fields:
            series = frame[column].dropna()
            values = sorted(series.unique(), key=lambda x: (str(type(x)), x))
            numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            if numeric:
                kind = st.radio(f'{label(column)} filter', ['Values', 'Range'], horizontal=True,
                                key=f'ga_kind_{column}')
            else:
                kind = 'Values'
            if kind == 'Range':
                lo, hi = float(series.min()), float(series.max())
                left, right = st.columns(2)
                lower = left.number_input(f'{label(column)} minimum', value=lo, format='%.9f', key=f'ga_lo_{column}')
                upper = right.number_input(f'{label(column)} maximum', value=hi, format='%.9f', key=f'ga_hi_{column}')
                if lower > upper:
                    st.error(f'{label(column)}: minimum must not exceed maximum.')
                    return
                ranges[column] = (lower, upper)
            else:
                selections[column] = st.multiselect(label(column), values, default=values,
                    key=f'ga_values_{column}', format_func=lambda v: f'{v:.9f}' if isinstance(v, float) else str(v))
        positions = sorted(frame['analysis.stage'].dropna().astype(int).unique().tolist())
        mode = st.radio('Training path stage', ['All stages', 'Selected stages', 'Latest existing runs'],
                        horizontal=True, key='ga_stage_mode')
        stages = None
        if mode == 'Selected stages':
            stages = st.multiselect('Stages (v0 = fresh)', positions, default=positions,
                                   format_func=lambda v: f'v{v}', key='ga_stages')
        st.caption('Stage follows checkpoint ancestry, not YAML queue order. v0 is fresh. '
                   'Latest existing runs are leaves of the full database lineage, resolved before filters; '
                   'branches may have several leaves. This does not guarantee the planned final epoch. '
                   'One run may contain more than one data epoch.')
        completed = st.checkbox('Completed runs only', value=True, key='ga_completed')
        evaluated = st.checkbox('Require a completed final evaluation in the selected scope', value=True, key='ga_evaluated')
    selected = filter_rows(frame, selections, ranges, stages, mode == 'Latest existing runs')
    if completed:
        selected = selected[selected['run.status'].eq('completed')]
    if evaluated:
        if 'eval.id' not in selected:
            selected = selected.iloc[:0]
        else:
            selected = selected[selected['eval.id'].notna()]
    a,b,c = st.columns(3)
    a.metric('Selected runs', len(selected))
    b.metric('Training paths', selected['path.root_run_id'].nunique())
    c.metric('Final evaluations', int(selected.get('eval.id', pd.Series(dtype=float)).notna().sum()))
    if selected.empty:
        st.info('No runs match these filters.')
        return
    metrics = evaluation_metrics(frame)
    defaults = [m for m in ['eval.agent_return', 'eval.agent_max_drawdown', 'eval.balanced_score',
                'eval.market_exposure', 'eval.round_trips'] if m in metrics]
    chosen = st.multiselect('Evaluation metrics', metrics, default=defaults, format_func=label, key='ga_metrics')
    grouping = st.multiselect('Group by (combine any parameters)', dimensions,
                default=[c for c in dimensions if c.endswith('ppo.gamma')], format_func=label, key='ga_groups')
    st.caption('N = finite values for this metric; paths = distinct fresh ancestors. '
               'Averages weight runs equally. Mixing stages gives longer paths more weight; '
               'runs within one path are not independent repetitions. Missing values are not zero.')
    overall = statistics_display(summarize(selected, [], chosen))
    grouped = statistics_display(summarize(selected, grouping, chosen))
    st.caption('Percentage rows use % for values and percentage points (pp) for standard deviation. '
               'Balanced score, profit factor, counts and hyperparameters retain their original units. '
               'Downloaded tables use the same display units as these tables.')
    st.markdown('#### Overall statistics')
    st.dataframe(overall, hide_index=True, width='stretch')
    st.markdown('#### Grouped statistics')
    st.dataframe(grouped, hide_index=True, width='stretch')
    st.download_button('Download grouped statistics CSV', grouped.to_csv(index=False),
                       f'grid_statistics_{scope.lower()}.csv', 'text/csv', key='ga_download_stats')
    if chosen:
        metric = st.selectbox('Distribution metric', chosen, format_func=label, key='ga_chart_metric')
        st.plotly_chart(distribution_figure(selected, metric, grouping, label(metric)), width='stretch')
        st.caption('Box = middle 50% (Q1–Q3); line = median. Whiskers extend to the most extreme '
                   'observations within 1.5 × IQR of the box. All run values are plotted; '
                   'points outside whiskers still count in the table minimum and maximum.')
        st.markdown('#### Path progression')
        st.caption('Each cell lists run ID and value. Multiple runs at one stage are branches, not averaged observations.')
        progression = selected[['path.root_run_id','analysis.stage','run_id',metric]].copy()
        progression['value'] = progression.apply(lambda r: f"#{int(r['run_id'])}: {metric_text(metric, r[metric])}", axis=1)
        resolved = progression.dropna(subset=['path.root_run_id','analysis.stage'])
        st.dataframe(resolved.pivot_table(index='path.root_run_id', columns='analysis.stage', values='value',
                     aggfunc=lambda values: ' | '.join(values)), width='stretch')
    if chosen:
        from train_and_eval.dashboard.path_improvement import render_path_improvements
        render_path_improvements(selected, frame, chosen, scope)
    st.markdown('#### Matching runs')
    columns = list(dict.fromkeys([c for c in ['run_id','run.name','run.architecture','analysis.stage',
                'path.root_run_id',*fields,*grouping,*chosen] if c in selected]))
    displayed = runs_display(selected[columns])
    st.dataframe(displayed, hide_index=True, width='stretch')
    st.download_button('Download matching runs CSV', displayed.to_csv(index=False),
                       f'grid_runs_{scope.lower()}.csv', 'text/csv', key='ga_download_runs')
