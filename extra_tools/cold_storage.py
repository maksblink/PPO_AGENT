#!/usr/bin/env python3
"""Complete Linux project backup. Run --help; no Git ignore rules filter payloads.

The destination is managed exclusively by this program. CURRENT is an atomic
pointer to a verified generation. Never train in, or edit, backup generations. Payloads are SHA-256 objects; Linux metadata live in manifests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

STATE = '.cold_storage'
FORMAT = 2
ACTIVE_TABLES = ('runs', 'evaluations', 'walk_forward_studies', 'walk_forward_cycles')


class BackupError(RuntimeError):
    pass


def command(args, **kwargs):
    if args[0] == 'git':
        kwargs['env'] = dict(kwargs.get('env', os.environ), GIT_OPTIONAL_LOCKS='0')
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if result.returncode:
        # External errors may contain connection strings; do not print them.
        raise BackupError(f'{Path(str(args[0])).name} failed (exit {result.returncode}).')
    return result.stdout


def fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    atomic_bytes(path, data)


def atomic_bytes(path, data):
    fd, name = tempfile.mkstemp(prefix='.write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise BackupError(f'Expected regular file: {path}')
    return json.loads(path.read_text())


def clean_git(root):
    if not (root / '.git').is_dir() or (root / '.git').is_symlink():
        raise BackupError('A standalone Git clone is required (.git directory, not an external worktree).')
    top = command(['git', '-C', str(root), 'rev-parse', '--show-toplevel']).decode().strip()
    if Path(top).resolve() != root.resolve():
        raise BackupError('Project must be the Git repository root.')
    if command(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all']):
        raise BackupError('Git is not clean. Commit changes and untracked source files first.')
    staged = command(['git', '-C', str(root), 'ls-files', '--stage'])
    if staged.startswith(b'160000 ') or b'\n160000 ' in staged:
        raise BackupError('Submodules are not supported; use a standalone repository.')
    if (root / '.git/objects/info/alternates').exists():
        raise BackupError('External Git object stores are not supported; use a full clone.')
    ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', STATE + '/probe'])
    if ignored.returncode:
        raise BackupError('Add /.cold_storage/ to .gitignore and commit it first.')


def no_workers(root):
    """Refuse local project workers even before their first database insert."""
    patterns = ('run_search_queue.py', 'train_and_eval.training', 'train_and_eval.walk_forward',
                'train_and_eval.evaluation', 'clean_training.py', 'rename_runs.py')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = (proc / 'cwd').resolve()
        except (OSError, RuntimeError):
            continue
        if any(p in cmd for p in patterns) and (cwd == root or root in cwd.parents or str(root) in cmd):
            raise BackupError(f'Project worker PID {proc.name} is active. Stop it before backup.')
    if (root / 'extra_tools/maintenance/run_rename/pending.json').exists():
        raise BackupError('Finish pending run-name migration before backup.')
    if (root / 'artifacts/.cleanup_pending.json').exists():
        raise BackupError('Finish pending training cleanup before backup.')


@contextmanager
def local_state(root):
    clean_git(root)
    path = root / STATE
    if path.is_symlink():
        raise BackupError('Local backup state must not be a symlink.')
    path.mkdir(mode=0o700, exist_ok=True)
    lock = path / 'lock'
    if lock.is_symlink():
        raise BackupError('Unsafe local lock.')
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupError('Another backup command is active.') from exc
        identity = path / 'identity.json'
        if not identity.exists():
            atomic_json(identity, {'format': FORMAT, 'project_id': str(uuid.uuid4())})
        project_id = read_json(identity)['project_id']
        if str(uuid.UUID(project_id)) != project_id:
            raise BackupError('Invalid local project identity.')
        yield path, project_id


def digest(path):
    before = path.lstat()
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise BackupError(f'File changed while opening: {path}')
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(stream.fileno())
    final = path.lstat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after) or signature(after) != signature(final):
        raise BackupError(f'File changed while hashing: {path}')
    return h


def scan(root, exclude_state=False):
    entries = {}
    def visit(directory):
        for path in sorted(directory.iterdir()):
            rel = path.relative_to(root).as_posix()
            if exclude_state and rel == STATE:
                continue
            s = path.lstat()
            entry = {'mode': stat.S_IMODE(s.st_mode), 'mtime_ns': s.st_mtime_ns,
                     'uid': s.st_uid, 'gid': s.st_gid,
                     'xattrs': {k: os.getxattr(path, k, follow_symlinks=False).hex()
                                for k in os.listxattr(path, follow_symlinks=False)}}
            if stat.S_ISLNK(s.st_mode):
                entry.update(kind='link', target=os.readlink(path))
            elif stat.S_ISDIR(s.st_mode):
                entry.update(kind='dir')
            elif stat.S_ISREG(s.st_mode):
                entry.update(kind='file', size=s.st_size, sha256=digest(path))
            else:
                raise BackupError(f'Unsupported socket/device/FIFO: {path}')
            entries[rel] = entry
            if entry['kind'] == 'dir':
                visit(path)
    visit(root)
    return entries


def validate_entries(entries):
    for name, entry in entries.items():
        p = PurePosixPath(name)
        if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\x00' in name:
            raise BackupError(f'Unsafe manifest path: {name!r}')
        if p.parts[0] == STATE:
            raise BackupError('Manifest contains tool control state.')
        if entry.get('kind') not in ('file', 'dir', 'link'):
            raise BackupError('Invalid manifest entry.')
        for parent in p.parents:
            if str(parent) != '.' and entries.get(str(parent), {}).get('kind') != 'dir':
                raise BackupError('Manifest parent is missing or is a symlink.')


def changes(old, new):
    return [{'path': p, 'status': 'D' if p not in new else 'N' if p not in old else 'M'}
            for p in sorted(old.keys() | new.keys()) if old.get(p) != new.get(p)]


def report(delta):
    for row in delta:
        print(f"{row['status']}  {row['path']}")
    print('Operations:', ', '.join(f'{s}={sum(r["status"] == s for r in delta)}' for s in ('N', 'M', 'D')))


def external_links(root, entries):
    links = []
    for name, entry in entries.items():
        if entry['kind'] != 'link':
            continue
        try:
            resolved = (root / name).resolve()
        except RuntimeError as exc:
            raise BackupError(f'Symlink loop: {name}') from exc
        if resolved != root and root not in resolved.parents:
            links.append({'path': name, 'target': entry['target']})
            # Virtual environments normally reference the system interpreter.
            if PurePosixPath(name).parts[0] not in ('.venv', 'venv'):
                raise BackupError(f'External symlink {name}: its data is outside the project. Move the data into the project first.')
    return links


def pg_environment(root):
    from dotenv import dotenv_values
    from sqlalchemy.engine import make_url
    raw = os.environ.get('DATABASE_URL') or dotenv_values(root / '.env').get('DATABASE_URL')
    if not raw:
        raise BackupError('DATABASE_URL is missing.')
    url = make_url(raw)
    if url.get_backend_name() != 'postgresql' or not url.database:
        raise BackupError('Expected a PostgreSQL DATABASE_URL.')
    env = dict(os.environ)
    # Avoid accidental PG service/default overrides.
    for key in list(env):
        if key.startswith('PG'):
            del env[key]
    env.update(PGDATABASE=url.database, PGHOST=url.host or 'localhost', PGPORT=str(url.port or 5432),
               PGUSER=url.username or '', PGPASSWORD=url.password or '', PGCONNECT_TIMEOUT='10')
    allowed = {'sslmode': 'PGSSLMODE', 'sslrootcert': 'PGSSLROOTCERT', 'sslcert': 'PGSSLCERT',
               'sslkey': 'PGSSLKEY', 'options': 'PGOPTIONS', 'connect_timeout': 'PGCONNECT_TIMEOUT'}
    for k, v in url.query.items():
        if k not in allowed or not isinstance(v, str):
            raise BackupError(f'Unsupported database URL option: {k}')
        env[allowed[k]] = v
    return url, env


@contextmanager
def frozen_database(root):
    """Prevent application writes while pg_dump and filesystem capture run."""
    from sqlalchemy import create_engine, text
    url, env = pg_environment(root)
    for binary in ('pg_dump', 'pg_dumpall', 'pg_restore'):
        if not shutil.which(binary):
            raise BackupError(f'Missing {binary}; install the PostgreSQL client matching the server.')
    engine = create_engine(url, hide_parameters=True, connect_args={'connect_timeout': 10})
    try:
        with engine.connect() as conn, conn.begin():
            conn.execute(text("SET LOCAL lock_timeout = '5s'"))
            tables = conn.execute(text("SELECT n.nspname, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind IN ('r','p') AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%' ORDER BY 1,2")).all()
            quote = conn.dialect.identifier_preparer.quote
            if tables:
                names = ', '.join(f'{quote(s)}.{quote(t)}' for s, t in tables)
                conn.execute(text('LOCK TABLE ' + names + ' IN SHARE MODE'))
            for schema, table in tables:
                if table in ACTIVE_TABLES:
                    active = conn.execute(text(f'SELECT count(*) FROM {quote(schema)}.{quote(table)} WHERE status = \'running\'')).scalar_one()
                    if active:
                        raise BackupError(f'{schema}.{table} has {active} running records. Stop/recover them first.')
            no_workers(root)
            version = conn.execute(text('SHOW server_version_num')).scalar_one()
            major = int(version) // 10000
            client = command(['pg_dump', '--version']).decode()
            if not re.search(rf'\b{major}\.', client):
                raise BackupError(f'pg_dump must match server major version {major}.')
            snapshot = conn.execute(text('SELECT pg_export_snapshot()')).scalar_one()
            yield env, snapshot, {'name': url.database, 'server_version_num': version,
                                   'pg_dump_version': client.strip(),
                                   'size_bytes': conn.execute(text('SELECT pg_database_size(current_database())')).scalar_one()}
    finally:
        engine.dispose()


def dump_database(directory, env, snapshot):
    dump = directory / 'database.dump'
    roles = directory / 'roles.sql'
    command(['pg_dump', '--no-password', '--format=custom', '--create', '--snapshot=' + snapshot,
             '--file=' + str(dump)], env=env)
    # Project PostgreSQL container's initial POSTGRES_USER is a superuser.
    command(['pg_dumpall', '--no-password', '--globals-only', '--file=' + str(roles)], env=env)
    command(['pg_restore', '--list', str(dump)], env=env)
    result = {}
    for p in (dump, roles):
        p.chmod(0o600)
        with p.open('rb') as f:
            os.fsync(f.fileno())
        result[p.name] = {'size': p.stat().st_size, 'sha256': digest(p)}
    return result


def discover():
    raw = command(['lsblk', '--json', '--bytes', '--output', 'NAME,PATH,TYPE,SIZE,MODEL,TRAN,FSTYPE,MOUNTPOINTS,UUID,RO'])
    data = json.loads(raw)
    mounts = []
    def visit(node, disk):
        disk = node if node['type'] == 'disk' else disk
        print(f"{node['path']:20} {node['type']:6} {int(node.get('size') or 0)/1e12:6.2f} TB  {node.get('model') or ''} {node.get('fstype') or ''} {node.get('mountpoints') or ''}")
        for mount in node.get('mountpoints') or []:
            if mount and mount != '[SWAP]' and not node.get('ro'):
                mounts.append(Path(mount))
        for child in node.get('children', []):
            visit(child, disk)
    for disk in data['blockdevices']:
        visit(disk, disk)
    return list(dict.fromkeys(mounts))


def choose_mount(value):
    if value:
        mount = Path(value).resolve()
    else:
        mounts = discover()
        if not mounts:
            raise BackupError('No mounted writable destinations found. Mount the HDD first.')
        for i, path in enumerate(mounts, 1):
            print(f'{i}: {path}')
        while True:
            answer = input('Mounted destination number (STOP to cancel): ').strip()
            if answer == 'STOP':
                raise BackupError('Cancelled.')
            if answer.isdigit() and 1 <= int(answer) <= len(mounts):
                mount = mounts[int(answer)-1]
                break
    if not mount.is_dir() or not os.path.ismount(mount) or mount == Path('/'):
        raise BackupError('Select an existing mounted data disk, not / or an ordinary directory.')
    return mount


def store_path(root, mount, project_id):
    if mount == root or root in mount.parents or mount in root.parents:
        raise BackupError('Destination cannot contain the source or be inside it.')
    if mount.stat().st_dev == root.stat().st_dev:
        raise BackupError('Destination must be a different filesystem from the source.')
    return mount / ('PPO_AGENT_backup_' + project_id)


def check_capabilities(parent):
    with tempfile.TemporaryDirectory(prefix='.cold-storage-probe-', dir=parent) as d:
        path = Path(d)
        a = path / 'a'
        a.write_bytes(b'probe')
        a.chmod(0o640)
        os.utime(a, ns=(1234567890000000000, 1234567890123456789))
        os.link(a, path / 'b')
        os.symlink('a', path / 'c')
        if stat.S_IMODE(a.stat().st_mode) != 0o640 or a.stat().st_mtime_ns != 1234567890123456789:
            raise BackupError('Destination must preserve Linux permissions and nanosecond timestamps (use ext4).')


@contextmanager
def destination(store, project_id, create=False):
    if store.is_symlink():
        raise BackupError('Backup root cannot be a symlink.')
    if not store.exists():
        if not create:
            yield None
            return
        store.mkdir(mode=0o700)
        atomic_json(store / 'identity.json', {'format': FORMAT, 'project_id': project_id})
    marker = read_json(store / 'identity.json')
    if marker != {'format': FORMAT, 'project_id': project_id}:
        raise BackupError('Destination belongs to a different project or format.')
    lock = store / 'lock'
    if lock.is_symlink():
        raise BackupError('Unsafe backup lock.')
    with lock.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupError('Backup disk is in use.') from exc
        yield store


def current(store):
    if store is None or not (store / 'CURRENT').exists():
        return None, None
    pointer = read_json(store / 'CURRENT')
    name = pointer['generation']
    if not re.fullmatch(r'[0-9TZ]+_[a-f0-9]{12}', name):
        raise BackupError('Invalid generation pointer.')
    gen = store / name
    if gen.is_symlink() or not gen.is_dir():
        raise BackupError('Missing generation.')
    if digest(gen / 'manifest.json') != pointer['manifest_sha256']:
        raise BackupError('Manifest checksum mismatch.')
    manifest = read_json(gen / 'manifest.json')
    if manifest['format'] != FORMAT or manifest['project_id'] != read_json(store / 'identity.json')['project_id']:
        raise BackupError('Invalid manifest identity.')
    validate_entries(manifest['files'])
    return gen, manifest


def object_path(store, sha):
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
        raise BackupError('Invalid object SHA-256.')
    base = store / 'objects'
    shard = base / sha[:2]
    if base.is_symlink() or shard.is_symlink():
        raise BackupError('Object directories must not be symlinks.')
    return shard / sha[2:]


def put_object(store, source, entry):
    target = object_path(store, entry['sha256'])
    target.parent.mkdir(parents=True, exist_ok=True)
    fsync_dir(store)
    fsync_dir(target.parent.parent)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.stat().st_size != entry['size'] or digest(target) != entry['sha256']:
            raise BackupError('Existing object is damaged; refusing to overwrite it.')
        return
    fd, name = tempfile.mkstemp(prefix='.object-', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as dst:
            src_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(src_fd, 'rb') as src:
                if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                    raise BackupError('Object source is not a regular file.')
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if temporary.stat().st_size != entry['size'] or digest(temporary) != entry['sha256']:
            raise BackupError('Object changed during copying or failed checksum.')
        os.replace(temporary, target)
        fsync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def references(doc):
    validate_entries(doc['files'])
    if set(doc['database_files']) != {'database.dump', 'roles.sql'}:
        raise BackupError('Incomplete database backup.')
    result = {}
    for entry in list(doc['files'].values()) + list(doc['database_files'].values()):
        if 'sha256' not in entry:
            continue
        sha, size = entry['sha256'], entry['size']
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha) or not isinstance(size, int) or size < 0:
            raise BackupError('Invalid object reference.')
        if sha in result and result[sha] != size:
            raise BackupError('Conflicting object sizes.')
        result[sha] = size
    for entry in doc['files'].values():
        if entry['kind'] == 'file' and ('sha256' not in entry or 'size' not in entry):
            raise BackupError('Missing file content reference.')
    for entry in doc['database_files'].values():
        if 'sha256' not in entry or 'size' not in entry:
            raise BackupError('Missing database content reference.')
    return result


def verify_generation(gen, manifest):
    for sha, size in references(manifest).items():
        path = object_path(gen.parent, sha)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size or digest(path) != sha:
            raise BackupError(f'Object missing or damaged: {sha}. No backup was replaced.')


def restore_tree(store, target, entries):
    validate_entries(entries)
    target.mkdir(mode=0o700)
    for name in sorted(entries):
        entry = entries[name]
        path = target / name
        if entry['kind'] == 'dir':
            path.mkdir(mode=0o700)
        elif entry['kind'] == 'link':
            os.symlink(entry['target'], path)
            set_metadata(path, entry)
        else:
            source = object_path(store, entry['sha256'])
            fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            set_metadata(path, entry)
    for name in sorted(entries, reverse=True):
        if entries[name]['kind'] == 'dir':
            set_metadata(target / name, entries[name])
            fsync_dir(target / name)
    fsync_dir(target)


def set_metadata(path, entry):
    current_stat = path.lstat()
    if (current_stat.st_uid, current_stat.st_gid) != (entry['uid'], entry['gid']):
        os.chown(path, entry['uid'], entry['gid'], follow_symlinks=False)
    if entry['kind'] != 'link':
        path.chmod(entry['mode'])
    for key in os.listxattr(path, follow_symlinks=False):
        if key not in entry['xattrs']:
            os.removexattr(path, key, follow_symlinks=False)
    for key, value in entry['xattrs'].items():
        os.setxattr(path, key, bytes.fromhex(value), follow_symlinks=False)
    os.utime(path, ns=(entry['mtime_ns'], entry['mtime_ns']), follow_symlinks=False)
    if entry['kind'] == 'file':
        with path.open('rb') as f:
            os.fsync(f.fileno())


def manifest(root, project_id, files, database_files, database, previous):
    return {'format': FORMAT, 'project_id': project_id, 'source_root': str(root),
            'created_at': datetime.now(timezone.utc).isoformat(), 'files': files,
            'changes_since_local_scan': changes(previous, files),
            'database_files': database_files, 'database': database,
            'external_venv_links': external_links(root, files),
            'git_commit': command(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip(),
            'python_version': sys.version, 'platform': list(os.uname()),
            'root_mode': stat.S_IMODE(root.stat().st_mode)}


def publish(store, pending, doc):
    atomic_json(pending / 'manifest.json', doc)
    fsync_dir(pending)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12]
    gen = store / name
    os.rename(pending, gen)
    fsync_dir(store)
    # Previous generations are never deleted automatically.
    atomic_json(store / 'CURRENT', {'generation': name, 'manifest_sha256': digest(gen / 'manifest.json')})
    return gen


def snapshot(root, local, project_id, store=None):
    no_workers(root)
    old_gen, old = current(store)
    if old:
        print('Verifying previous backup objects (SHA-256)...', flush=True)
        verify_generation(old_gen, old)
    previous = read_json(local / 'manifest.json')['files'] if (local / 'manifest.json').exists() else {}
    with frozen_database(root) as (env, snapshot_id, database):
        print('Scanning all project files, including ignored files...', flush=True)
        files = scan(root, exclude_state=True)
        external_links(root, files)
        delta = changes(old['files'] if old else {}, files)
        unique = {e['sha256']: e['size'] for e in files.values() if e['kind'] == 'file'}
        needed = sum(size for sha, size in unique.items() if store and not object_path(store, sha).exists())
        dump_reserve = max(1024**3, int(database['size_bytes']) * 2)
        metadata_reserve = max(256 * 1024**2, len(files) * 16384)
        if shutil.disk_usage(local).free < dump_reserve + metadata_reserve:
            raise BackupError('Not enough free local space for the database snapshot and manifest.')
        if store:
            required = needed + dump_reserve + metadata_reserve
            if shutil.disk_usage(store).free < required:
                raise BackupError(f'Not enough free backup space: need at least {required / 1024**3:.2f} GiB.')
            report(delta)
            print(f'New file content bytes: {needed:,}; existing content is reused without hardlinks.')
            if input('Type YES to create a new complete backup generation: ').strip() != 'YES':
                raise BackupError('Cancelled; previous backup unchanged.')
        # Dumps use local Linux storage, never permissions/xattrs on exFAT.
        local_pending = Path(tempfile.mkdtemp(prefix='.incomplete-', dir=local))
        pending = None
        try:
            print('Dumping PostgreSQL...', flush=True)
            db_files = dump_database(local_pending, env, snapshot_id)
            doc = manifest(root, project_id, files, db_files, database, previous)
            doc['changes_since_backup'] = delta if store else None
            if store:
                pending = Path(tempfile.mkdtemp(prefix='.incomplete-', dir=store))
                print('Writing new content objects...', flush=True)
                seen = set()
                for name, entry in files.items():
                    if entry['kind'] == 'file' and entry['sha256'] not in seen:
                        put_object(store, root / name, entry)
                        seen.add(entry['sha256'])
                for name, entry in db_files.items():
                    put_object(store, local_pending / name, entry)
                verify_generation(pending, doc)
            print('Rechecking source stability...', flush=True)
            clean_git(root)
            no_workers(root)
            if scan(root, exclude_state=True) != files:
                raise BackupError('Source changed during backup. Stop writers and retry; previous backup is intact.')
            if store:
                atomic_bytes(store / 'RESTORE.py', Path(__file__).read_bytes())
                gen = publish(store, pending, doc)
                atomic_json(local / 'manifest.json', doc)
                atomic_json(local / 'last_destination.json', {'path': str(store), 'generation': gen.name})
                print(f'Backup committed and verified: {gen}\nPrevious manifests retained; prune removes unreferenced content.')
            else:
                atomic_json(local_pending / 'manifest.json', doc)
                capture = local / ('capture-' + uuid.uuid4().hex)
                os.rename(local_pending, capture)
                atomic_json(local / 'manifest.json', doc)
                atomic_json(local / 'capture.json', {'path': capture.name})
                report(doc['changes_since_local_scan'])
                print(f'Local manifest: {local / "manifest.json"}\nDatabase snapshot: {capture}')
        finally:
            if pending is not None and pending.exists():
                shutil.rmtree(pending)
            if local_pending.exists():
                shutil.rmtree(local_pending)


def restore_files(store, target):
    gen, doc = current(store)
    if not gen:
        raise BackupError('No complete backup exists.')
    # Git cleanliness is checked after reconstructing into a temporary directory.
    verify_generation(gen, doc)
    if target.exists() or target.is_symlink():
        raise BackupError('Restore target must not exist; nothing is overwritten.')
    if store == target or store in target.parents or target in store.parents:
        raise BackupError('Restore target must be separate from backup storage.')
    if not target.parent.is_dir():
        raise BackupError('Create the restore parent directory first.')
    needed = sum(e.get('size', 0) for e in doc['files'].values())
    if shutil.disk_usage(target.parent).free < needed + 1024**3:
        raise BackupError('Insufficient free space for restoration.')
    if any(e['uid'] != os.getuid() for e in doc['files'].values()) and os.geteuid() != 0:
        raise BackupError('Archive contains different file owners. Restore with matching UID or an administrator; do not silently change owners.')
    check_capabilities(target.parent)
    pending = Path(tempfile.mkdtemp(prefix='.restore-', dir=target.parent))
    try:
        restore_tree(store, pending / 'project', doc['files'])
        clean_git(pending / 'project')
        if scan(pending / 'project') != doc['files']:
            raise BackupError('Restored files failed verification.')
        state = pending / 'project' / STATE
        state.mkdir(mode=0o700)
        atomic_json(state / 'identity.json', {'format': FORMAT, 'project_id': doc['project_id']})
        atomic_json(state / 'manifest.json', doc)
        for name in ('database.dump', 'roles.sql'):
            shutil.copyfile(object_path(store, doc['database_files'][name]['sha256']), state / name)
            (state / name).chmod(0o600)
            with (state / name).open('rb') as f:
                os.fsync(f.fileno())
        fsync_dir(state)
        (pending / 'project').chmod(doc['root_mode'])
        os.rename(pending / 'project', target)
        fsync_dir(target.parent)
    finally:
        shutil.rmtree(pending)
    print(f'Files restored and verified: {target}\nNEXT: restore PostgreSQL from {target / STATE / "database.dump"}. See INSTRUCTIONS.\nDo not train until database restoration is complete. Recreate .venv on a different machine/path.')


def restore_database(root, local):
    from sqlalchemy import create_engine, text
    doc = read_json(local / 'manifest.json')
    dump = local / 'database.dump'
    entry = doc['database_files']['database.dump']
    if dump.stat().st_size != entry['size'] or digest(dump) != entry['sha256']:
        raise BackupError('Restored database dump checksum mismatch.')
    url, env = pg_environment(root)
    if url.database != doc['database']['name'] or url.database in ('postgres', 'template0', 'template1'):
        raise BackupError('DATABASE_URL must name the original application database, not a maintenance database.')
    if not shutil.which('pg_restore'):
        raise BackupError('Install the matching PostgreSQL client before database restoration.')
    no_workers(root)
    engine = create_engine(url.set(database='postgres'), hide_parameters=True)
    try:
        with engine.connect() as conn:
            exists = conn.execute(text('SELECT 1 FROM pg_database WHERE datname = :name'), {'name': url.database}).first()
            major = int(conn.execute(text('SHOW server_version_num')).scalar_one()) // 10000
            if exists:
                raise BackupError('Target database already exists. Refusing to overwrite it, even if empty. Use a fresh PostgreSQL instance initialized with POSTGRES_DB=postgres.')
            if major != int(doc['database']['server_version_num']) // 10000:
                raise BackupError('Restore into the same PostgreSQL major version as the backup.')
        print(f'Restore database {url.database!r} on {url.host}:{url.port or 5432}. The database must not exist.')
        if input('Type YES to restore the database: ').strip() != 'YES':
            raise BackupError('Cancelled.')
        command(['pg_restore', '--no-password', '--exit-on-error', '--create', '--dbname=postgres', str(dump)], env=env)
        atomic_json(local / 'database_restored.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
                    'dump_sha256': entry['sha256'], 'database': url.database})
        print('Database restored: schema, rows, IDs, sequences, ownership and grants from the dump. Run project verification before training.')
    finally:
        engine.dispose()


def prune(store):
    gen, doc = current(store)
    if not gen:
        raise BackupError('No current backup; refusing to prune.')
    verify_generation(gen, doc)
    keep = set(references(doc))
    candidates = [p for p in store.iterdir() if p != gen and (re.fullmatch(r'[0-9TZ]+_[a-f0-9]{12}', p.name) or p.name.startswith('.incomplete-'))]
    for path in candidates:
        if path.is_symlink() or not path.is_dir():
            raise BackupError('Unexpected path in backup storage.')
    garbage = []
    base = store / 'objects'
    if base.is_symlink() or not base.is_dir():
        raise BackupError('Invalid object directory.')
    for shard in base.iterdir():
        if shard.is_symlink() or not shard.is_dir() or not re.fullmatch(r'[0-9a-f]{2}', shard.name):
            raise BackupError('Unexpected object shard; nothing deleted.')
        for path in shard.iterdir():
            if path.is_symlink() or not path.is_file():
                raise BackupError('Unexpected object type; nothing deleted.')
            if re.fullmatch(r'[0-9a-f]{62}', path.name):
                if shard.name + path.name not in keep:
                    garbage.append(path)
            elif path.name.startswith('.object-'):
                garbage.append(path)
            else:
                raise BackupError('Unexpected object name; nothing deleted.')
    for path in candidates:
        print('DELETE generation:', path.name)
    print(f'Unreferenced objects/partial objects: {len(garbage)}; bytes: {sum(p.stat().st_size for p in garbage):,}')
    if (candidates or garbage) and input('Type YES to delete old generations and unreferenced objects: ').strip() == 'YES':
        # Delete old manifests first: after interruption remaining objects are harmless.
        for path in candidates:
            shutil.rmtree(path)
        fsync_dir(store)
        for path in garbage:
            path.unlink()
            fsync_dir(path.parent)
    print('Current generation and every referenced object retained.')


def probe(local, mount):
    """Tiny round-trip on the actual HDD; only owned temporary folders touched."""
    with tempfile.TemporaryDirectory(prefix='probe-', dir=local) as local_tmp, tempfile.TemporaryDirectory(prefix='.ppo-backup-probe-', dir=mount) as disk_tmp:
        local_root, store = Path(local_tmp), Path(disk_tmp)
        source = local_root / 'source'
        source.mkdir()
        (source / 'Case').write_bytes(b'upper case')
        (source / 'case').write_bytes(b'lower case')
        (source / 'empty').mkdir()
        (source / '.hidden').write_bytes(b'same content')
        (source / 'duplicate').write_bytes(b'same content')
        (source / 'link').symlink_to('.hidden')
        (source / '.hidden').chmod(0o640)
        os.setxattr(source / '.hidden', 'user.backup_probe', b'metadata')
        entries = scan(source)
        for name, entry in entries.items():
            if entry['kind'] == 'file':
                put_object(store, source / name, entry)
        # Exercise control-file replacement/fsync and advisory locking on HDD.
        with (store / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(store / 'probe.json', {'files': entries})
            atomic_json(store / 'probe.json', {'files': entries, 'revision': 2})
            read_back = read_json(store / 'probe.json')['files']
            restored = local_root / 'restored'
            restore_tree(store, restored, read_back)
            if scan(restored) != entries:
                raise BackupError('Destination probe failed metadata/content round-trip.')
        print('PASS: content, case-sensitive names, deduplication, symlinks, metadata, locking and atomic pointer writes. Temporary test files removed on exit. No experiment database accessed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['disks', 'probe', 'scan', 'compare', 'sync', 'verify', 'prune', 'restore-files', 'restore-database'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--disk', help='Mounted filesystem root; omit for interactive selection.')
    parser.add_argument('--backup', type=Path, help='Existing backup folder, for restore-files only.')
    parser.add_argument('--target', type=Path, help='New project folder, for restore-files only.')
    args = parser.parse_args(argv)
    try:
        if args.mode == 'restore-files':
            if not args.backup or not args.target:
                raise BackupError('restore-files requires --backup and --target.')
            store = args.backup.absolute()
            identity = read_json(store / 'identity.json')
            with destination(store, identity['project_id']):
                restore_files(store, args.target.resolve())
            return 0
        root = args.project.resolve()
        with local_state(root) as (local, project_id):
            if args.mode == 'disks':
                discover()
            elif args.mode == 'restore-database':
                restore_database(root, local)
            elif args.mode == 'scan':
                snapshot(root, local, project_id)
            else:
                mount = choose_mount(args.disk)
                store = store_path(root, mount, project_id)
                if args.mode == 'probe':
                    probe(local, mount)
                    return 0
                with destination(store, project_id, create=args.mode == 'sync') as dest:
                    if args.mode == 'sync':
                        snapshot(root, local, project_id, dest)
                    elif args.mode == 'compare':
                        local_doc = read_json(local / 'manifest.json')
                        _, remote = current(dest)
                        print('Manifest comparison only. Run scan to refresh local inventory first.')
                        report(changes(remote['files'] if remote else {}, local_doc['files']))
                        print('Database dump:', 'unchanged' if remote and remote['database_files'] == local_doc['database_files'] else 'new/changed snapshot')
                    else:
                        gen, doc = current(dest)
                        if not gen:
                            raise BackupError('No completed backup on selected disk.')
                        if args.mode == 'verify':
                            verify_generation(gen, doc)
                            print('PASS: every file and database dump matches SHA-256 and manifest metadata.')
                        else:
                            prune(dest)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Previously committed backup remains available.', file=sys.stderr)
        return 130
    except Exception as exc:
        # SQLAlchemy exceptions can include credentials or database records.
        message = str(exc) if isinstance(exc, (BackupError, FileNotFoundError, PermissionError)) else type(exc).__name__
        print('Backup stopped: ' + message, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
