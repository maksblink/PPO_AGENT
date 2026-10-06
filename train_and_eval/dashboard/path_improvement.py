"""Compare direct checkpoint continuation edges after dashboard filtering."""
from __future__ import annotations

import math
import pandas as pd

from train_and_eval.dashboard.analysis_display import metric_text


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def path_improvements(selected: pd.DataFrame, population: pd.DataFrame,
                      metric: str, direction: str = 'Increase') -> tuple[pd.DataFrame, pd.DataFrame]:
    """One unbranched fresh-root path per observation; no bridging filtered gaps."""
    if direction not in {'Increase', 'Decrease'}:
        raise ValueError('Unknown improvement direction')
    if population['run_id'].duplicated().any() or selected['run_id'].duplicated().any():
        raise ValueError('Expected one row per run')
    children = population.dropna(subset=['path.parent_run_id']).groupby('path.parent_run_id').size()
    fork_ids = set(children[children > 1].index)
    branched_roots = set(population.loc[population['run_id'].isin(fork_ids), 'path.root_run_id'].dropna())
    names = population.set_index('run_id')['run.name'].to_dict()
    records = []
    for root, members in selected.groupby('path.root_run_id', dropna=False, sort=True):
        members = members.sort_values(['analysis.stage', 'run_id'], na_position='last')
        lookup = members.set_index('run_id')
        improved = worsened = ties = missing = gaps = 0
        unresolved = pd.isna(root) or not members['path.status'].eq('Complete').all()
        branched = not unresolved and root in branched_roots
        for row in members.to_dict('records'):
            parent = row['path.parent_run_id']
            if pd.isna(parent):
                continue
            if parent not in lookup.index:
                gaps += 1
                continue
            previous, current = lookup.at[parent, metric], row[metric]
            if not (_finite(previous) and _finite(current)):
                missing += 1
                continue
            difference = float(current) - float(previous)
            if direction == 'Decrease':
                difference = -difference
            improved += difference > 0
            worsened += difference < 0
            ties += difference == 0
        comparable = improved + worsened + ties
        status = ('Unresolved ancestry' if unresolved else 'Branched path (excluded)' if branched
                  else 'No comparable transitions' if comparable == 0 else 'Included')
        values = '; '.join(
            f"v{int(r['analysis.stage']) if pd.notna(r['analysis.stage']) else '?'} "
            f"#{int(r['run_id'])}: {metric_text(metric, r[metric])}"
            for r in members.to_dict('records'))
        records.append({'root_run_id': root, 'root_name': names.get(root, 'Unknown'),
            'status': status, 'selected_runs': len(members), 'comparisons': comparable,
            'improvements': improved, 'deteriorations': worsened, 'ties': ties,
            'missing_metric_edges': missing, 'filtered_parent_edges': gaps,
            'fraction': f'{improved}/{comparable}' if comparable else '—', 'progression': values})
    details = pd.DataFrame(records)
    hist = []
    if not details.empty:
        included = details[details['status'].eq('Included')]
        for n, paths in included.groupby('comparisons', sort=True):
            for k in range(int(n), -1, -1):
                count = int(paths['improvements'].eq(k).sum())
                hist.append({'comparisons': int(n), 'improved': f'{k}/{n}',
                    'paths': count, 'paths_with_this_denominator': len(paths),
                    'share_percent': 100 * count / len(paths)})
    return pd.DataFrame(hist), details


def render_path_improvements(selected: pd.DataFrame, population: pd.DataFrame,
                             metrics: list[str], scope: str) -> None:
    import streamlit as st
    from train_and_eval.dashboard.formatting import dataframe
    st.markdown('#### Training path improvement')
    metric = st.selectbox('Path improvement metric', metrics, key='ga_path_metric')
    direction = st.radio('Improvement direction', ['Increase', 'Decrease'], horizontal=True,
                         key='ga_path_direction')
    st.caption('For return, balanced score and negative drawdown, choose Increase (drawdown closer to zero). '
               'For exposure or trade counts, choose the direction you want to examine. Ties are not improvements.')
    histogram, details = path_improvements(selected, population, metric, direction)
    st.caption('n counts direct parent→child comparisons with finite values in the selected data. '
               'Five runs normally give four comparisons. Filtered gaps are never bridged. '
               'Percentages are calculated separately for each denominator. Missing edges are reported below. '
               'Branched roots are excluded to avoid counting a tree as one linear path; '
               'branching is checked against the full database before filters.')
    if details.empty:
        st.info('No training paths selected.')
        return
    a, b, c = st.columns(3)
    a.metric('Paths included', int(details.status.eq('Included').sum()))
    b.metric('Paths without comparisons', int(details.status.eq('No comparable transitions').sum()))
    c.metric('Branched / unresolved groups', int(details.status.isin(['Branched path (excluded)', 'Unresolved ancestry']).sum()))
    if histogram.empty:
        st.info('No comparable linear paths. Select adjacent stages and runs with finite metric values.')
    else:
        dataframe(histogram, hide_index=True, width='stretch',
            column_config={'share_percent': st.column_config.NumberColumn('Share (%)', format='%.5f%%')})
    dataframe(details, hide_index=True, width='stretch')
    st.download_button('Download path improvement summary CSV', histogram.to_csv(index=False),
                       f'path_improvement_summary_{scope.lower()}.csv', 'text/csv', key='ga_path_summary_csv')
    st.download_button('Download path improvement details CSV', details.to_csv(index=False),
                       f'path_improvement_details_{scope.lower()}.csv', 'text/csv', key='ga_path_details_csv')
