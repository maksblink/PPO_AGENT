#!/usr/bin/env python3
"""Delete selected PPO_AGENT experiments and their registered artifacts.

Run from the repository: python extra_tools/clean_training.py MODE [--dry-run].
No sequence resets, schema changes, data/config deletion, or implicit cascades.
Stop trainers/evaluators before use. Stage-two selection always covers a study.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import delete, select, text, update, func
from train_and_eval.database.models import (
    Run, Checkpoint, Evaluation, TrainingMetric, WalkForwardCycle, WalkForwardStudy,
)
from train_and_eval.database.session import create_database_engine, get_database_url

CP_FIELDS = ('source_checkpoint_id', 'selected_checkpoint_id', 'refit_checkpoint_id')
EV_FIELDS = ('reference_evaluation_id', 'test_evaluation_id')


def rows(connection, model):
    return list(connection.execute(select(model.__table__)).mappings())


def status(row):
    return getattr(row['status'], 'value', row['status'])


def safe_path(root, relative):
    """Allow only descendants of artifacts, without following symlinks."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts) < 3 or relative.parts[0] != 'artifacts':
        raise ValueError(f'Unsafe or nonstandard artifact path: {relative}')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Artifact path contains a symlink: {current}')
    return current


def make_plan(connection, root, mode, run_ids=(), study_ids=()):
    runs, checkpoints, evaluations, cycles, studies = (
        rows(connection, cls) for cls in (Run, Checkpoint, Evaluation, WalkForwardCycle, WalkForwardStudy))
    by_run = {r['id']: r for r in runs}
    by_cycle = {c['id']: c for c in cycles}
    by_study = {s['id']: s for s in studies}
    requested = set(run_ids)
    requested_studies = set(study_ids)
    if requested - by_run.keys() or requested_studies - by_study.keys():
        raise ValueError('An explicitly requested run/study does not exist')
    if mode == 'clean-all' and (requested or requested_studies):
        raise ValueError('clean-all does not accept ID filters; use a stage-specific mode')
    if mode == 'clean-first-stage':
        if requested_studies:
            raise ValueError('--study-id is only available for clean-second-stage')
        selected_runs = requested or {r['id'] for r in runs if r['cycle_id'] is None and r['stage_role'] is None}
        if any(by_run[i]['cycle_id'] is not None or by_run[i]['stage_role'] is not None for i in selected_runs):
            raise ValueError('Stage one only accepts ordinary run IDs')
        selected_studies = set()
    elif mode == 'clean-second-stage':
        if any(by_run[i]['cycle_id'] not in by_cycle for i in requested):
            raise ValueError('Stage two only accepts walk-forward run IDs')
        selected_studies = requested_studies | {by_cycle[by_run[i]['cycle_id']]['study_id'] for i in requested}
        if not requested and not requested_studies:
            selected_studies = set(by_study)
        selected_runs = {r['id'] for r in runs if r['cycle_id'] in by_cycle
                         and by_cycle[r['cycle_id']]['study_id'] in selected_studies}
    else:
        selected_runs, selected_studies = set(by_run), set(by_study)
    selected_cycles = {c['id'] for c in cycles if c['study_id'] in selected_studies}
    selected_cp = {c['id'] for c in checkpoints if c['run_id'] in selected_runs}
    # A reference evaluation may be attached to a stage-one checkpoint.
    selected_ev = {e['id'] for e in evaluations if e['checkpoint_id'] in selected_cp}
    extra_ev = {c[key] for c in cycles if c['id'] in selected_cycles for key in EV_FIELDS if c[key] is not None}
    shared_ev = {c[key] for c in cycles if c['id'] not in selected_cycles for key in EV_FIELDS if c[key] is not None}
    selected_ev |= extra_ev - shared_ev  # Shared references belong to retained studies too.

    for r in runs:
        if r['id'] not in selected_runs and r['source_checkpoint_id'] in selected_cp:
            raise ValueError(f"Kept run #{r['id']} depends on selected checkpoints; include its descendants or clean-all")
    for c in cycles:
        if c['id'] not in selected_cycles and (any(c[k] in selected_cp for k in CP_FIELDS)
                                               or any(c[k] in selected_ev for k in EV_FIELDS)):
            raise ValueError(f"Kept study #{c['study_id']} depends on selected records; clean-second-stage first or clean-all")
    for s in studies:
        if s['id'] in selected_studies:
            continue
        source = s['plan'].get('stage_one_source', {})
        ancestor_runs = {a.get('run_id') for a in source.get('ancestors', [])} | {source.get('run_id')}
        source_cp = s['protocol'].get('source_checkpoint_id', source.get('checkpoint_id'))
        if ancestor_runs & selected_runs or source_cp in selected_cp:
            raise ValueError(f"Kept study #{s['id']} retains selected source ancestry; clean-second-stage first or clean-all")
    # A worker may be between database writes. Locking tables alone is not enough.
    if any(status(r) == 'running' for r in runs) or any(status(e) == 'running' for e in evaluations) \
            or any(status(s) == 'running' for s in studies) or any(status(c) == 'running' for c in cycles):
        raise ValueError('Running work exists in the database. Stop/resolve it before cleanup; nothing deleted')

    paths = {f'artifacts/runs/{i:08d}' for i in selected_runs}
    paths |= {f'artifacts/walk_forward/{i:08d}' for i in selected_studies}
    cp_run = {c['id']: c['run_id'] for c in checkpoints}
    for e in evaluations:
        if e['id'] in selected_ev and cp_run[e['checkpoint_id']] not in selected_runs:
            paths.add(f"artifacts/runs/{cp_run[e['checkpoint_id']]:08d}/evaluations/{e['id']:08d}")
    for c in checkpoints:
        if c['id'] in selected_cp:
            expected = Path(f"artifacts/runs/{c['run_id']:08d}")
            relative = Path(c['relative_path'])
            if not relative.is_relative_to(expected):
                raise ValueError(f"Checkpoint #{c['id']} has a nonstandard artifact path: {relative}; nothing deleted")
            safe_path(root, relative)
    for path in paths:
        safe_path(root, path)
    count = connection.scalar(select(func.count()).select_from(TrainingMetric).where(TrainingMetric.run_id.in_(selected_runs)))
    return {'mode': mode, 'run_ids': sorted(selected_runs), 'study_ids': sorted(selected_studies),
            'cycle_ids': sorted(selected_cycles), 'checkpoint_ids': sorted(selected_cp),
            'evaluation_ids': sorted(selected_ev), 'training_metrics': count,
            'runs': [{'id': i, 'name': by_run[i]['name']} for i in sorted(selected_runs)],
            'studies': [{'id': i, 'name': by_study[i]['name']} for i in sorted(selected_studies)],
            'paths': sorted(paths)}


def delete_rows(connection, plan):
    rid, cid = plan['run_ids'], plan['cycle_ids']
    # Break only internal links, within this transaction and deletion scope.
    connection.execute(update(WalkForwardCycle).where(WalkForwardCycle.id.in_(cid)).values(
        **{key: None for key in CP_FIELDS + EV_FIELDS}))
    connection.execute(update(Run).where(Run.id.in_(rid)).values(
        source_checkpoint_id=None, continuation_mode='fresh'))
    connection.execute(delete(Evaluation).where(Evaluation.id.in_(plan['evaluation_ids'])))
    connection.execute(delete(TrainingMetric).where(TrainingMetric.run_id.in_(rid)))
    connection.execute(delete(Checkpoint).where(Checkpoint.id.in_(plan['checkpoint_ids'])))
    connection.execute(delete(Run).where(Run.id.in_(rid)))
    connection.execute(delete(WalkForwardCycle).where(WalkForwardCycle.id.in_(cid)))
    connection.execute(delete(WalkForwardStudy).where(WalkForwardStudy.id.in_(plan['study_ids'])))


@contextmanager
def locked(connection):
    with connection.begin():
        connection.execute(text('SET LOCAL lock_timeout = \'3s\''))
        connection.execute(text('LOCK TABLE runs, checkpoints, evaluations, training_metrics, '
                                'walk_forward_cycles, walk_forward_studies IN SHARE ROW EXCLUSIVE MODE'))
        yield


def journal_path(root):
    path = root / 'artifacts' / '.cleanup_pending.json'
    if (root / 'artifacts').is_symlink() or path.is_symlink():
        raise ValueError('Cleanup journal must not be a symlink')
    return path


def write_journal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finish_pending(connection, root, identity):
    """Recover a crash by checking the transaction's anchor records, not a flag."""
    path = journal_path(root)
    if not path.exists():
        return
    pending = json.loads(path.read_text())
    if pending['database'] != identity:
        raise ValueError('Pending cleanup belongs to another database; do not change DATABASE_URL during recovery')
    plan = pending['plan']
    total = len(plan['run_ids']) + len(plan['study_ids'])
    existing = connection.scalar(select(func.count()).select_from(Run).where(Run.id.in_(plan['run_ids'])))
    existing += connection.scalar(select(func.count()).select_from(WalkForwardStudy).where(WalkForwardStudy.id.in_(plan['study_ids'])))
    if total == 0 or existing not in (0, total):
        raise ValueError('Ambiguous pending cleanup state; preserve the journal and staged files for inspection')
    for item in pending['moves']:
        original, staged = (safe_path(root, item[k]) for k in ('original', 'staged'))
        if existing:
            if staged.exists():
                if original.exists():
                    raise ValueError(f'Cannot restore over an existing path: {original}')
                original.parent.mkdir(parents=True, exist_ok=True)
                staged.rename(original)
        elif staged.exists():
            if staged.is_dir():
                shutil.rmtree(staged)
            else:
                staged.unlink()
    trash = safe_path(root, pending['trash'] + '/placeholder').parent
    if trash.exists():
        trash.rmdir()
    path.unlink()
    print('Pending cleanup recovered: ' + ('artifacts restored (database rollback).' if existing else 'artifact deletion completed.'))


def cleanup(engine, root, mode, run_ids=(), study_ids=(), dry_run=False, confirm=input):
    root = Path(root).resolve()
    # Identity contains no credentials; it also distinguishes test schemas.
    with engine.connect() as connection:
        identity = list(connection.execute(text('SELECT current_database(), current_schema()')).one())
        identity.append(hashlib.sha256(engine.url.render_as_string(hide_password=True).encode()).hexdigest())
        connection.rollback()
        with locked(connection):
            if journal_path(root).exists():
                if dry_run:
                    raise ValueError('Pending cleanup exists. Run without --dry-run to recover it first')
                if confirm('Recover the interrupted cleanup first? Type RECOVER: ') != 'RECOVER':
                    return False
                finish_pending(connection, root, identity)
            plan = make_plan(connection, root, mode, run_ids, study_ids)
            print(json.dumps(plan, indent=2))
            print(f'Database: {identity[0]}; schema: {identity[1]}; project: {root}')
            if dry_run or not (plan['run_ids'] or plan['study_ids']):
                print('Preview only; nothing deleted.')
                return False
            if confirm(f'Type DELETE {mode} to permanently delete the listed records and artifacts: ') != f'DELETE {mode}':
                print('Cancelled; nothing deleted.')
                return False
            trash = 'artifacts/.cleanup-trash-' + uuid.uuid4().hex
            moves = [{'original': path, 'staged': f'{trash}/{index}'} for index, path in enumerate(plan['paths'])
                     if safe_path(root, path).exists()]
            pending = {'database': identity, 'plan': plan, 'trash': trash, 'moves': moves}
            write_journal(journal_path(root), pending)
            # Rename on the same filesystem, then commit DB deletions. A crash or
            # any exception leaves a journal; next invocation restores or purges.
            for item in moves:
                original, staged = (safe_path(root, item[k]) for k in ('original', 'staged'))
                staged.parent.mkdir(parents=True, exist_ok=True)
                original.rename(staged)
            delete_rows(connection, plan)
        with locked(connection):
            finish_pending(connection, root, identity)
    print('Cleanup completed. Market data, configs, migrations and ID sequences were preserved.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['clean-first-stage', 'clean-second-stage', 'clean-all'])
    parser.add_argument('--run-id', type=int, action='append', default=[], help='Repeatable; stage two expands to the entire owning study')
    parser.add_argument('--study-id', type=int, action='append', default=[], help='Repeatable; stage two only')
    parser.add_argument('--dry-run', action='store_true', help='Print scope without modifying database or files')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    engine = create_database_engine(get_database_url(env_path=root / '.env'))
    try:
        cleanup(engine, root, args.mode, args.run_id, args.study_id, args.dry_run)
    except (Exception, KeyboardInterrupt) as error:
        print(f'Cleanup stopped: {error}', file=sys.stderr)
        print('If artifacts/.cleanup_pending.json exists, rerun this tool to recover before starting training.', file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
