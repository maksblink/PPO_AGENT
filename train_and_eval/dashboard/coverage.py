"""Read-only grid coverage: planned configurations versus persisted execution."""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml

AXES = ['architecture', 'gamma', 'lr', 'n_epochs', 'seed', 'ent_coef', 'gae_lambda']


def decimal_text(value):
    return format(Decimal(str(value)).normalize(), 'f')


def coordinates(config):
    from train_and_eval.dashboard.lineage import architecture_label
    ppo = config['ppo']
    return (architecture_label(config), decimal_text(ppo['gamma']),
            decimal_text(ppo['learning_rate']), int(ppo['n_epochs']), int(config['run']['seed']),
            decimal_text(ppo['ent_coef']), decimal_text(ppo['gae_lambda']))


def family_settings(config):
    """Keep all non-axis scientific settings; omit names and presentation only."""
    value = copy.deepcopy(config)
    for section in ('run', 'continuation', 'logging', 'artifacts'):
        value.pop(section, None)
    for key in ('hidden_sizes', 'gamma', 'learning_rate', 'n_epochs', 'ent_coef', 'gae_lambda'):
        value['ppo'].pop(key, None)
    return value


def read_plan(root: Path, manifests: list[Path]):
    from train_and_eval.run_config import RunConfig, normalize_config
    entries, errors = {}, []
    for manifest in manifests:
        try:
            queue = yaml.safe_load(manifest.read_text())
            if queue.get('queue_schema_version') != 1 or not isinstance(queue.get('configs'), list):
                raise ValueError('Invalid queue schema')
            if len(set(queue['configs'])) != len(queue['configs']):
                raise ValueError('Duplicate paths in queue')
            for relative in queue['configs']:
                try:
                    path = (root / relative).resolve()
                    path.relative_to(root.resolve())
                    config = json.loads(normalize_config(RunConfig.model_validate(yaml.safe_load(path.read_text()))))
                    name = config['run']['name']
                    if name in entries:
                        if entries[name]['config'] != config:
                            raise ValueError(f'Duplicate run name with different config: {name}')
                        entries[name]['queues'].add(manifest.stem)
                        entries[name]['paths'].add(relative)
                    else:
                        entries[name] = dict(config=config, path=relative, paths={relative}, queues={manifest.stem})
                except Exception as exc:
                    errors.append(f'{manifest.name}: {relative}: {exc}')
        except Exception as exc:
            errors.append(f'{manifest.name}: {exc}')
    return entries, errors


def coverage_rows(entries, runs, checkpoints, evaluations):
    """No trading-performance fields are read or returned."""
    by_name = {}
    for row in runs:
        by_name.setdefault(row['name'], []).append(row)
    cps = {row['id']: row for row in checkpoints}
    finals = {}
    for cp in checkpoints:
        if str(cp['save_reason']) == 'final':
            finals.setdefault(cp['run_id'], []).append(cp)
    phases, visiting = {}, set()

    def phase(name):
        if name in phases:
            return phases[name]
        if name not in entries:
            raise ValueError('Parent config outside selected plan')
        if name in visiting:
            raise ValueError('Cycle in configured ancestry')
        visiting.add(name)
        try:
            continuation = entries[name]['config']['continuation']
            result = 0 if continuation['mode'] == 'fresh' else phase(continuation['source_run']) + 1
            phases[name] = result
            return result
        finally:
            visiting.remove(name)

    completed_evals = {(e['checkpoint_id'], e['data_scope']) for e in evaluations
                       if str(e['status']) == 'completed' and str(e['trigger']) == 'final'}
    rows, families = [], {}
    for name, entry in entries.items():
        config = entry['config']
        settings = family_settings(config)
        family = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        families[family] = settings
        issues = []
        try:
            position = phase(name)
        except ValueError as exc:
            position = None
            issues.append(str(exc))
        source = config['continuation'].get('source_run')
        if source in entries and coordinates(entries[source]['config']) != coordinates(config):
            issues.append('Parent uses different grid coordinates')
        if source in entries and family_settings(entries[source]['config']) != settings:
            issues.append('Parent uses different non-axis settings')
        matches = by_name.get(name, [])
        if len(matches) > 1:
            issues.append('Duplicate database run name')
        run = matches[0] if matches else None
        final = finals.get(run['id'], []) if run else []
        train = val = False
        status = str(run['status']) if run else 'not_started'
        if run:
            if run.get('normalized_config_json') != config:
                issues.append('Disk/database configuration mismatch')
            cp = cps.get(run.get('source_checkpoint_id'))
            parent = by_name.get(source, []) if source else []
            if source and (not cp or len(parent) != 1 or cp['run_id'] != parent[0]['id']):
                issues.append('Source checkpoint does not belong to configured parent')
            elif source and config['continuation'].get('checkpoint') == 'final' and str(cp['save_reason']) != 'final':
                issues.append('Source checkpoint is not final')
            elif not source and run.get('source_checkpoint_id') is not None:
                issues.append('Fresh run has a source checkpoint')
            if len(final) == 1:
                train = (final[0]['id'], 'run_training') in completed_evals
                val = (final[0]['id'], 'run_validation') in completed_evals
            if status == 'completed':
                if len(final) != 1:
                    issues.append('Missing or ambiguous final checkpoint record')
                if not train:
                    issues.append('Missing completed final TRAIN evaluation')
                if not val:
                    issues.append('Missing completed final VAL evaluation')
                requested = run.get('training_steps_requested')
                done = run.get('training_steps_completed')
                if requested is not None and done is not None and done < requested:
                    issues.append('Training stopped before requested budget')
        rows.append(dict(zip(AXES, coordinates(config)), family=family, phase=position,
            name=name, config_path=entry['path'], queues=', '.join(sorted(entry['queues'])),
            parent=source, checkpoint_selector=config['continuation'].get('checkpoint'),
            source_checkpoint_id=run.get('source_checkpoint_id') if run else None,
            run_id=run['id'] if run else None, status=status,
            training_steps_completed=run.get('training_steps_completed') if run else None,
            training_steps_requested=run.get('training_steps_requested') if run else None,
            data_epochs_completed=run.get('data_epochs_completed') if run else None,
            final_checkpoint_id=final[0]['id'] if len(final) == 1 else None,
            train_evaluation=train, val_evaluation=val,
            ready=status == 'completed' and not issues, issues='; '.join(issues)))
    # Two separate chains with equal coordinates must never masquerade as one chain.
    counts = {}
    for row in rows:
        key = (row['family'], *(row[a] for a in AXES), row['phase'])
        counts[key] = counts.get(key, 0) + 1
    for row in rows:
        key = (row['family'], *(row[a] for a in AXES), row['phase'])
        if counts[key] > 1:
            row['issues'] = '; '.join(filter(None, [row['issues'], 'Duplicate grid stage']))
            row['ready'] = False
    return pd.DataFrame(rows), families


def grid_cells(frame):
    """One cell per declared coordinate tuple; never invent Cartesian configs."""
    cells = []
    for key, subset in frame.groupby(['family', *AXES], dropna=False, sort=True):
        done = int(subset.ready.sum())
        if subset.issues.ne('').any() or subset.status.isin(['failed', 'cancelled']).any():
            state = 'issue'
        elif subset.status.eq('running').any():
            state = 'running'
        elif done == len(subset):
            state = 'complete'
        elif subset.status.eq('completed').any():
            state = 'partial'
        else:
            state = 'not_started'
        cells.append(dict(zip(['family', *AXES], key), state=state, done=done,
                          total=len(subset), planned=len(subset),
                          planned_phases=', '.join(f'v{p:02d}' for p in sorted(set(subset.phase.dropna().astype(int)))),
                          completed_runs=int(subset.status.eq('completed').sum())))
    return pd.DataFrame(cells, columns=['family', *AXES, 'state', 'done', 'total',
                                      'planned', 'planned_phases', 'completed_runs'])


def load_registry():
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    engine = create_database_engine()
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
                runs = [dict(r) for r in connection.execute(text(
                    'SELECT id,name,status,source_checkpoint_id,normalized_config_json,'
                    'training_steps_completed,training_steps_requested,data_epochs_completed FROM runs')).mappings()]
                cps = [dict(r) for r in connection.execute(text(
                    'SELECT id,run_id,save_reason FROM checkpoints')).mappings()]
                evaluations = [dict(r) for r in connection.execute(text(
                    'SELECT checkpoint_id,data_scope,status,trigger FROM evaluations')).mappings()]
        return runs, cps, evaluations
    finally:
        engine.dispose()


LABELS = {'complete': '✓ Complete', 'partial': '◐ Partial', 'running': '▶ Running',
          'issue': '! Needs review', 'not_started': '○ Not started'}
COLORS = {'complete': '#2ca873', 'partial': '#eab54b', 'running': '#559ee8',
          'issue': '#e36868', 'not_started': '#939ba7'}


def coverage_figure(cells):
    import plotly.graph_objects as go
    figure = go.Figure()
    for state, label in LABELS.items():
        part = cells.loc[cells.state == state]
        if part.empty:
            continue
        figure.add_scatter(x=part.gamma.tolist(), y=[format(Decimal(v), f'.{max(9, -Decimal(v).as_tuple().exponent)}f') for v in part.lr],
            mode='markers+text', name=label,
            marker=dict(symbol='square', size=42, color=COLORS[state], line=dict(width=1, color='#444')),
            text=[f'{r.done}/{r.total}' for r in part.itertuples()],
            textfont=dict(color='#111'),
            customdata=part[AXES].values.tolist(),
            hovertemplate=('Gamma %{x}<br>LR %{y}<br>Ready stages %{text}<extra>'+label+'</extra>'))
    figure.update_layout(height=max(350, 48*cells.lr.nunique()+150),
                         margin=dict(l=30,r=20,t=20,b=30), clickmode='event+select',
                         legend=dict(orientation='h'))
    figure.update_xaxes(title='Gamma', type='category', categoryorder='array',
                       categoryarray=sorted(cells.gamma.unique(), key=Decimal))
    figure.update_yaxes(title='Learning rate', type='category', categoryorder='array',
                       categoryarray=sorted({format(Decimal(v), f'.{max(9, -Decimal(v).as_tuple().exponent)}f') for v in cells.lr}, key=Decimal, reverse=True))
    return figure


def render_grid_coverage(root: Path):
    import streamlit as st
    from train_and_eval.dashboard.formatting import dataframe, plotly_chart
    st.subheader('Grid Coverage')
    st.button('Refresh coverage')
    st.caption('Read-only plan and execution coverage. TRAIN/VAL performance values and sidebar filters do not apply.')
    manifests = sorted(
        p for p in (root / 'configs/search_queues').rglob('*')
        if p.is_file() and p.suffix in {'.yml', '.yaml'}
    )
    if not manifests:
        st.info('No queue manifests found.'); return
    selected = st.selectbox('Grid / queue', manifests,
        format_func=lambda p: str(p.relative_to(root / 'configs/search_queues')), key='coverage_queue')
    scope = str(selected.relative_to(root))
    entries, errors = read_plan(root, [selected])
    if errors:
        st.error('Some planned configs could not be read. Coverage is incomplete; see plan errors.')
        with st.expander('Plan errors', expanded=True):
            for error in errors:
                st.write(error)
        return  # Never report full coverage from a partially readable plan.
    if not entries:
        return
    try:
        runs, cps, evals = load_registry()
    except Exception as exc:
        st.error('Could not read coverage registry.'); st.exception(exc); return
    frame, families = coverage_rows(entries, runs, cps, evals)
    st.caption(f'Selected queue: {selected.stem} · {len(frame)} declared runs · '
               f'{int(frame.ready.sum())} ready. Only this queue defines expected coverage. '
               'A matching database run is reused even when shared with another grid.')
    st.caption('Families separate non-axis settings such as data ranges, reward, costs, rollout/batch settings and evaluation protocol.')
    family = st.selectbox('Configuration family', sorted(families), key=f'coverage_family_{scope}',
        format_func=lambda f: f'{f} · {int(frame.family.eq(f).sum())} planned runs')
    with st.expander('Family settings — parameters held constant'):
        st.json(families[family])
    included = frame.loc[frame.family == family].copy()
    context = f'{scope}_{family}'
    for axis, label in [('n_epochs', 'PPO n_epochs'), ('seed', 'Seed'),
                        ('ent_coef', 'Entropy coefficient'), ('gae_lambda', 'GAE lambda')]:
        options = sorted(included[axis].unique(), key=lambda value: Decimal(str(value)))
        choice = st.selectbox(label, options, key=f'coverage_{axis}_{context}')
        included = included.loc[included[axis] == choice]
        context += f'_{choice}'
    architectures = sorted(included.architecture.unique())
    arches = st.multiselect('Architectures', architectures, default=architectures,
                            key=f'coverage_arch_{context}')
    included = included.loc[included.architecture.isin(arches)]
    if included.empty:
        st.info('Select at least one architecture.'); return
    cells = grid_cells(included)
    a,b,c,d = st.columns(4)
    a.metric('Planned stages', len(included))
    b.metric('Completed run records', int(included.status.eq('completed').sum()))
    c.metric('Ready stages', int(cells.done.sum()))
    d.metric('Complete paths', f'{int(cells.state.eq("complete").sum())}/{len(cells)}')
    st.caption('Only declared configurations are expected. Blank positions are not planned, not missing. '
               'Each path uses its own declared stages; no global path length is inferred.')
    st.caption('Ready = completed full requested budget + matching config/parent + one final checkpoint record + completed final TRAIN and VAL evaluations. Files on disk are not scanned. Stage number is continuation depth, not a measured data-epoch count.')
    for arch in arches:
        st.markdown(f'**{arch}**')
        event = plotly_chart(coverage_figure(cells.loc[cells.architecture == arch]),
            use_container_width=True, on_select='rerun', selection_mode='points', key=f'coverage_chart_{context}_{arch}')
        points = event.get('selection', {}).get('points', [])
        if points:
            point = points[0].get('customdata')
            if point:
                detail = included
                for axis, value in zip(AXES, point):
                    detail = detail.loc[detail[axis] == value]
                dataframe(detail.drop(columns=['family', 'ready']).sort_values('phase'), hide_index=True)
                if detail.empty:
                    st.info('No configuration is planned for this cell.')
    st.subheader('Unfinished planned paths')
    gaps = cells.loc[cells.state != 'complete'].copy()
    dataframe(gaps, hide_index=True, use_container_width=True)
    st.download_button('Download gaps CSV', gaps.to_csv(index=False), f'{selected.stem}_coverage_unfinished.csv', 'text/csv')
    st.subheader('Stage details')
    problems_only = st.checkbox('Only unfinished stages or issues', value=True)
    details = included.loc[~included.ready] if problems_only else included
    details = details.drop(columns=['family', 'ready']).sort_values(AXES+['phase'], na_position='last')
    dataframe(details, hide_index=True, use_container_width=True)
    st.download_button('Download stage details CSV', details.to_csv(index=False), f'{selected.stem}_coverage_stages.csv', 'text/csv')
