#!/usr/bin/env python3
"""Rename audited stage-one runs, with a recoverable database/files journal."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HISTORY = 'extra_tools/maintenance/run_rename'
FIELDS = ('name', 'raw_config_yaml', 'normalized_config_json', 'config_sha256', 'normalized_config_sha256')


def sha(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def safe(root, rel):
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe path: {rel}')
    current = root
    for part in p.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Symlink not allowed: {rel}')
    return current


def read_optional(path):
    return path.read_text(encoding='utf-8') if path.exists() else None


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        if path.exists():
            path.unlink()
    else:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.run_rename_', delete=False) as stream:
            temp = Path(stream.name)
            try:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def rename_config(raw, mapping):
    import yaml
    from train_and_eval.run_config import RunConfig, normalize_config
    old = yaml.safe_load(raw)
    new = copy.deepcopy(old)
    new['run']['name'] = mapping[old['run']['name']]
    if new['continuation']['mode'] == 'resume':
        new['continuation']['source_run'] = mapping[old['continuation']['source_run']]
    normalized = normalize_config(RunConfig.model_validate(new))
    # Edit only the two name scalars, preserving decimal LR spelling and comments.
    if new == old:
        return raw, json.loads(normalized), sha(normalized)
    tree = yaml.compose(raw)
    replacements = []
    for section_node, value_node in tree.value:
        fields = {'run': {'name'}, 'continuation': {'source_run'}}.get(section_node.value, set())
        if fields:
            for key_node, scalar in value_node.value:
                if key_node.value in fields:
                    replacements.append((scalar.start_mark.index, scalar.end_mark.index,
                                         new[section_node.value][key_node.value]))
    updated = raw
    for start, end, value in sorted(replacements, reverse=True):
        updated = updated[:start] + value + updated[end:]
    if normalize_config(RunConfig.model_validate(yaml.safe_load(updated))) != normalized:
        raise ValueError('Name-only YAML rewrite changed configuration semantics')
    return updated, json.loads(normalized), sha(normalized)


def project_identity(connection, engine, root):
    from sqlalchemy import text
    return {'project': str(root.resolve()), 'database': connection.scalar(text('SELECT current_database()')),
            'schema': connection.scalar(text('SELECT current_schema()')),
            'host': engine.url.host, 'port': engine.url.port, 'user': engine.url.username}


def lock_database(connection):
    from sqlalchemy import text
    connection.execute(text('SET LOCAL lock_timeout = \'5s\''))
    connection.execute(text('LOCK TABLE runs, checkpoints, evaluations, training_metrics, '
                            'walk_forward_studies, walk_forward_cycles IN SHARE ROW EXCLUSIVE MODE'))


def run_rows(connection):
    from sqlalchemy import select
    from train_and_eval.database.models import Run
    return {r['id']: dict(r) for r in connection.execute(select(Run.__table__)).mappings()}


def build_plan(connection, root, audit):
    from sqlalchemy import select, func
    from train_and_eval.database.models import Checkpoint, Evaluation, WalkForwardStudy, WalkForwardCycle
    from train_and_eval.run_config import RunConfig, normalize_config
    import yaml
    from extra_tools.audit_run_names import proposed_names, LR_DECIMAL_PLACES
    if audit.get('audit_version') != 2 or audit.get('lr_decimal_places') != LR_DECIMAL_PLACES or audit.get('errors'):
        raise ValueError('A successful version-2 fixed-LR audit is required')
    entries = audit['mapping']
    old_names = [e['old_name'] for e in entries]
    new_names = [e['new_name'] for e in entries]
    if len(set(old_names)) != len(entries) or len(set(new_names)) != len(entries):
        raise ValueError('Duplicate name mapping')
    for entry in entries:
        if entry['new_name'] in old_names and entry['new_name'] != entry['old_name']:
            raise ValueError('Cross-name collision in migration')
    mapping = dict(zip(old_names, new_names))
    rows = run_rows(connection)
    if set(rows) != {e['run_id'] for e in entries if e['run_id'] is not None}:
        raise ValueError('Database run IDs changed since audit')
    for model in (WalkForwardStudy, WalkForwardCycle):
        if connection.scalar(select(func.count()).select_from(model)):
            raise ValueError('Stage-two records exist; refusing incomplete rename')
    if connection.scalar(select(func.count()).select_from(Evaluation).where(Evaluation.status.in_(['pending', 'running']))):
        raise ValueError('Pending/running evaluations exist; finish them before rename')
    cps = {c['id']: dict(c) for c in connection.execute(select(Checkpoint.__table__)).mappings()}
    if len(cps) != audit['counts']['checkpoints'] or connection.scalar(select(func.count()).select_from(Evaluation)) != audit['counts']['evaluations']:
        raise ValueError('Checkpoint/evaluation counts changed since audit')
    files = {}; db = []
    def change(rel, after):
        if rel in files:
            raise ValueError(f'Duplicate file change: {rel}')
        path = safe(root, rel)
        files[rel] = {'path': rel, 'before': read_optional(path), 'after': after}
    path_mapping = {}
    disk_configs = {}
    expected_paths = {e['old_config_path'] for e in entries}
    actual_paths = {p.relative_to(root).as_posix() for p in (root/'configs/stage_one').rglob('*')
                    if p.suffix in ('.yml', '.yaml')}
    if actual_paths != expected_paths:
        raise ValueError('Stage-one config inventory changed since audit')
    for e in entries:
        old_path = e['old_config_path']
        raw_disk = safe(root, old_path).read_text(encoding='utf-8')
        if sha(raw_disk) != e['disk_config_sha256']:
            raise ValueError(f'Config changed since audit: {old_path}')
        cfg = json.loads(normalize_config(RunConfig.model_validate(yaml.safe_load(raw_disk))))
        if cfg['run']['name'] != e['old_name']:
            raise ValueError('Config name differs from audit')
        disk_configs[e['old_name']] = cfg
    expected_names = proposed_names(disk_configs)
    for e in entries:
        cfg = disk_configs[e['old_name']]
        source = cfg['continuation'].get('source_run')
        if e['new_name'] != expected_names[e['old_name']]['new_name'] or e['old_source_run'] != source:
            raise ValueError('Audit mapping does not match the fixed-LR template')
        if e['run_id'] is None:
            continue
        r = rows[e['run_id']]
        if getattr(r['status'], 'value', r['status']) != 'completed' or r['cycle_id'] is not None:
            raise ValueError(f'Run is not a completed stage-one run: {r["id"]}')
        for key in ('name', 'config_sha256', 'normalized_config_sha256', 'source_checkpoint_id'):
            expected = e['old_name'] if key == 'name' else e[key]
            if r[key] != expected:
                raise ValueError(f'Run #{r["id"]} changed: {key}')
        if sha(r['raw_config_yaml']) != r['config_sha256']:
            raise ValueError('Stored YAML hash mismatch')
        if sha(normalize_config(RunConfig.model_validate(r['normalized_config_json']))) != r['normalized_config_sha256']:
            raise ValueError('Stored normalized hash mismatch')
        if json.loads(normalize_config(RunConfig.model_validate(yaml.safe_load(r['raw_config_yaml'])))) != r['normalized_config_json']:
            raise ValueError('Stored YAML and normalized config differ')
        source = r['normalized_config_json']['continuation'].get('source_run')
        if source != e['old_source_run']:
            raise ValueError('Source name changed')
        if source:
            cp = cps.get(r['source_checkpoint_id'])
            if not cp or rows[cp['run_id']]['name'] != source or getattr(cp['save_reason'], 'value', cp['save_reason']) != 'final':
                raise ValueError('Actual checkpoint ancestry differs from config')
        elif r['source_checkpoint_id'] is not None:
            raise ValueError('Fresh run has a parent checkpoint')
        before = {k:r[k] for k in FIELDS}
        raw, norm, norm_hash = rename_config(r['raw_config_yaml'], mapping)
        after = dict(name=e['new_name'], raw_config_yaml=raw, normalized_config_json=norm,
                     config_sha256=sha(raw), normalized_config_sha256=norm_hash)
        db.append({'id': r['id'], 'before': before, 'after': after})
    for e in entries:
        old_path, new_path = e['old_config_path'], e['new_config_path']
        if not old_path.startswith('configs/stage_one/') or Path(old_path).parent != Path(new_path).parent or Path(new_path).stem != e['new_name']:
            raise ValueError('Invalid config destination')
        text = safe(root, old_path).read_text(encoding='utf-8')
        if e['run_id'] is not None and disk_configs[e['old_name']] != rows[e['run_id']]['normalized_config_json']:
            raise ValueError('Disk/database config mismatch (including LR)')
        if old_path != new_path and safe(root, new_path).exists():
            raise ValueError(f'Target exists: {new_path}')
        renamed, _, _ = rename_config(text, mapping)
        if old_path != new_path:
            change(old_path, None)
        change(new_path, renamed)
        path_mapping[old_path] = new_path
    # Known structured metadata only: no global string substitution in reports.
    for path in sorted((root/'configs/search_queues').rglob('*')):
        if path.suffix not in ('.yml', '.yaml'):
            continue
        rel = path.relative_to(root).as_posix(); raw = safe(root, rel).read_text(encoding='utf-8')
        q = yaml.safe_load(raw)
        if not isinstance(q, dict) or not isinstance(q.get('configs'), list):
            raise ValueError(f'Unrecognized queue format: {rel}')
        new = [path_mapping.get(p, p) for p in q['configs']]
        if new != q['configs']:
            q['configs'] = new; change(rel, yaml.safe_dump(q, sort_keys=False, allow_unicode=True))
        else:
            change(rel, raw)
    for e in entries:
        if e['run_id'] is None:
            continue
        rel = f'artifacts/runs/{e["run_id"]:08d}/reports/summary.json'
        path = safe(root, rel)
        if path.exists():
            obj = json.loads(path.read_text(encoding='utf-8'))
            if obj.get('run_id') != e['run_id'] or obj.get('run_name') != e['old_name']:
                raise ValueError(f'Unexpected report identity: {rel}')
            raw = path.read_text(encoding='utf-8')
            obj['run_name'] = e['new_name']
            change(rel, raw if e['old_name'] == e['new_name'] else dump(obj))
    for ref in audit['metadata_references']:
        rel = ref['path']
        if ref.get('scan') or rel not in files:
            raise ValueError(f'Unresolved audit reference: {rel}')
        if sha(safe(root, rel).read_text(encoding='utf-8')) != ref['sha256']:
            raise ValueError(f'Referenced file changed since audit: {rel}')
    for cp in cps.values():
        path = safe(root, cp['relative_path'])
        if not path.is_file() or path.stat().st_size != cp['size_bytes']:
            raise ValueError(f'Checkpoint missing or wrong size: {cp["id"]}')
    return {'file_only': not any(r['before'] != r['after'] for r in db), 'runs': db, 'files': list(files.values()), 'mapping': entries,
            'checkpoint_sha256': {str(k):v['sha256'] for k,v in cps.items()}}


def check_files(root, plan, allow_after=False):
    for f in plan['files']:
        actual = read_optional(safe(root, f['path']))
        allowed = [f['before'], f['after']] if allow_after else [f['before']]
        if actual not in allowed:
            raise ValueError(f'File changed outside migration: {f["path"]}')


def finish_files(root, plan, side):
    check_files(root, plan, allow_after=True)
    # New/replaced files first, old paths removed only afterwards.
    for f in sorted(plan['files'], key=lambda f: f[side] is None):
        path = safe(root, f['path'])
        if read_optional(path) != f[side]:
            write_atomic(path, f[side])
    for f in plan['files']:
        if read_optional(safe(root, f['path'])) != f[side]:
            raise ValueError('File verification failed')


def db_side(connection, plan):
    rows = run_rows(connection)
    for side in ('before', 'after'):
        if set(rows) == {r['id'] for r in plan['runs']} and all(
            {k:rows[r['id']][k] for k in FIELDS} == r[side] for r in plan['runs']):
            # With no DB mutation, a durable journal is sufficient to roll files forward.
            return 'after' if plan.get('file_only') else side
    raise ValueError('Database differs from both journal states; no recovery performed')


def apply_database(connection, plan):
    from sqlalchemy import update
    from train_and_eval.database.models import Run
    if db_side(connection, plan) != 'before':
        raise ValueError('Database changed before application')
    for row in plan['runs']:
        connection.execute(update(Run.__table__).where(Run.id == row['id']).values(**row['after']))
    if db_side(connection, plan) != 'after':
        raise ValueError('Database verification failed')


def pending_path(root):
    return safe(root, HISTORY+'/pending.json')


def recover(engine, root):
    pending = pending_path(root)
    if not pending.exists():
        print('No pending rename.'); return
    pointer = json.loads(pending.read_text())
    journal = safe(root, pointer['manifest'])
    raw = journal.read_text()
    if sha(raw) != pointer['sha256']:
        raise ValueError('Journal hash mismatch')
    plan = json.loads(raw)
    with engine.begin() as c:
        lock_database(c)
        if project_identity(c, engine, root) != plan['identity']:
            raise ValueError('Journal belongs to a different project/database')
        side = db_side(c, plan)
        finish_files(root, plan, side)
        write_atomic(journal.parent/'result.json', dump({'status': 'completed' if side=='after' else 'rolled_back',
                                                       'recovered_at': datetime.now(timezone.utc).isoformat()}))
        write_atomic(pending, None)
    print(f'Recovery finished: {side}. History: {journal.parent}')


def migrate(engine, root, audit, apply=False, confirm=input):
    if pending_path(root).exists():
        raise ValueError('Pending migration: run --recover before any training or cleanup')
    if safe(root,'artifacts/.cleanup_pending.json').exists():
        raise ValueError('Recover pending cleanup before rename')
    with engine.begin() as c:
        lock_database(c)
        identity = project_identity(c, engine, root)
        if audit.get('project', str(root)) != str(root) or audit.get('database', identity['database']) != identity['database']:
            raise ValueError('Audit belongs to another project/database')
        plan = build_plan(c, root, audit)
        plan.update(identity=project_identity(c, engine, root), created_at=datetime.now(timezone.utc).isoformat(),
                    audit_git_commit=audit.get('git_commit'), format_version=1)
        pending = sum(e['run_id'] is None for e in plan['mapping'])
        print(f'Runs: {len(plan["runs"])}; pending configs: {pending}; file operations: {len(plan["files"])}; checkpoint files unchanged.')
        if not any(f['before'] != f['after'] for f in plan['files']) and not any(r['before'] != r['after'] for r in plan['runs']):
            print('Already normalized; nothing to change.'); return
        if not apply:
            print('Preview only; no configs, artifacts or database records changed.'); return
        if confirm('Type YES to rename the audited runs and configs: ') != 'YES':
            print('Cancelled; nothing changed.'); return
        check_files(root, plan)
        directory = safe(root, HISTORY+'/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_migration')
        directory.mkdir(parents=True, exist_ok=False)
        for f in plan['files']:
            if f['before'] is not None:
                write_atomic(safe(directory, 'before/'+f['path']), f['before'])
        write_atomic(directory/'before/database.json', dump(plan['runs']))
        csv_path = directory/'mapping.csv'
        with csv_path.open('x', newline='', encoding='utf-8') as stream:
            keys = ['run_id','old_name','new_name','old_config_path','new_config_path',
                    'old_source_run','new_source_run','status','learning_rate']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
            writer.writerows({k:e.get(k) for k in keys} for e in plan['mapping'])
            stream.flush();os.fsync(stream.fileno())
        raw = dump(plan); journal = directory/'manifest.json'
        write_atomic(journal, raw)
        write_atomic(pending_path(root), dump({'manifest':journal.relative_to(root).as_posix(),'sha256':sha(raw)}))
        # Database commits first; persistent journal makes the filesystem step recoverable.
        if not plan['file_only']:
            apply_database(c, plan)
    recover(engine, root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--apply',action='store_true')
    group.add_argument('--recover',action='store_true')
    args=parser.parse_args()
    if not args.recover and args.audit is None:parser.error('--audit is required')
    from train_and_eval.database.session import create_database_engine
    lock=safe(ROOT,HISTORY+'/.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    engine=create_database_engine()
    try:
        with lock.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.recover:recover(engine,ROOT)
            else:migrate(engine,ROOT,json.loads(args.audit.read_text()),args.apply)
    except (Exception, KeyboardInterrupt) as error:
        print(f'Rename stopped: {error}',file=sys.stderr)
        if pending_path(ROOT).exists():
            print('Before training/cleanup, run: python extra_tools/rename_runs.py --recover',file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
