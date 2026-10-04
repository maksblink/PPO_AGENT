from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('cold_storage', Path(__file__).resolve().parents[1] / 'extra_tools/cold_storage.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    git(root, 'init')
    git(root, 'config', 'user.email', 'test@example.invalid')
    git(root, 'config', 'user.name', 'Test')
    (root / '.gitignore').write_text('/.cold_storage/\n/data/\n/.env\n/.venv/\n')
    (root / 'code.py').write_text('print(1)\n')
    git(root, 'add', '.')
    git(root, 'commit', '-m', 'fixture')
    (root / 'data').mkdir()
    (root / 'data/market.bin').write_bytes(b'market')
    (root / '.env').write_text('PASSWORD=secret')
    return root


def test_inventory_includes_git_ignored_secrets_and_virtual_environment(project):
    (project / '.venv').mkdir()
    (project / '.venv/python').symlink_to('/usr/bin/python3')
    (project / backup.STATE).mkdir()
    (project / backup.STATE / 'manifest.json').write_text('{}')
    entries = backup.scan(project, exclude_state=True)
    assert 'data/market.bin' in entries and '.env' in entries and '.git/HEAD' in entries
    assert '.venv/python' in entries
    assert not any(p.startswith(backup.STATE) for p in entries)
    assert backup.external_links(project, entries)[0]['path'] == '.venv/python'


def archive_files(store, project, entries):
    store.mkdir(exist_ok=True)
    for name, entry in entries.items():
        if entry['kind'] == 'file':
            backup.put_object(store, project / name, entry)


def test_scan_and_copy_preserve_modes_times_links_and_xattrs(project, tmp_path):
    p = project / 'data/market.bin'
    p.chmod(0o640)
    os.setxattr(p, 'user.test', b'value')
    (project / 'data/alias').symlink_to('market.bin')
    source = backup.scan(project)
    target = tmp_path / 'copy'
    store = tmp_path / 'objects-store'
    archive_files(store, project, source)
    backup.restore_tree(store, target, source)
    assert backup.scan(target) == source
    assert os.getxattr(target / 'data/market.bin', 'user.test') == b'value'


def test_incremental_objects_reuse_content_and_handle_rename(project, tmp_path):
    first = backup.scan(project)
    store = tmp_path / 'store'
    archive_files(store, project, first)
    original = backup.object_path(store, first['data/market.bin']['sha256'])
    original_stat = original.stat()
    (project / 'data/market.bin').rename(project / 'data/renamed.bin')
    (project / '.env').write_text('PASSWORD=changed')
    new = backup.scan(project)
    delta = {r['path']: r['status'] for r in backup.changes(first, new)}
    assert delta['data/market.bin'] == 'D'
    assert delta['data/renamed.bin'] == 'N'
    assert delta['.env'] == 'M'
    archive_files(store, project, new)
    assert original.stat().st_ino == original_stat.st_ino
    assert original.stat().st_mtime_ns == original_stat.st_mtime_ns
    old = tmp_path / 'old'
    target = tmp_path / 'new'
    backup.restore_tree(store, old, first)
    backup.restore_tree(store, target, new)
    assert backup.scan(target) == new
    assert (old / '.env').read_text() == 'PASSWORD=secret'
    assert not (target / 'data/market.bin').exists()


def test_same_size_and_mtime_content_change_is_detected(tmp_path):
    p = tmp_path / 'a'
    p.write_bytes(b'abc')
    before = backup.scan(tmp_path)
    when = p.stat().st_mtime_ns
    p.write_bytes(b'def')
    os.utime(p, ns=(when, when))
    assert backup.changes(before, backup.scan(tmp_path)) == [{'path': 'a', 'status': 'M'}]


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', './a', '.cold_storage/x'])
def test_reject_manifest_escape(name):
    with pytest.raises(backup.BackupError):
        backup.validate_entries({name: {'kind': 'file'}})


def test_reject_manifest_file_under_symlink():
    with pytest.raises(backup.BackupError):
        backup.validate_entries({'a': {'kind': 'link'}, 'a/b': {'kind': 'file'}})


def test_reject_external_data_and_special_files(project):
    (project / 'outside').symlink_to('/tmp')
    with pytest.raises(backup.BackupError, match='External symlink'):
        backup.external_links(project, backup.scan(project))
    (project / 'outside').unlink()
    os.mkfifo(project / 'fifo')
    with pytest.raises(backup.BackupError, match='FIFO'):
        backup.scan(project)


def test_clean_git_rejects_untracked_and_modified_but_not_ignored(project):
    backup.clean_git(project)
    (project / 'new.py').write_text('')
    with pytest.raises(backup.BackupError, match='not clean'):
        backup.clean_git(project)
    (project / 'new.py').unlink()
    (project / 'code.py').write_text('changed')
    with pytest.raises(backup.BackupError, match='not clean'):
        backup.clean_git(project)


def test_clean_check_does_not_modify_archived_index(project):
    before = backup.scan(project)
    backup.clean_git(project)
    assert backup.scan(project) == before


def test_reject_same_filesystem_and_nested_destination(project, tmp_path):
    with pytest.raises(backup.BackupError):
        backup.store_path(project, tmp_path, 'abc')
    sibling = tmp_path / 'disk'
    sibling.mkdir()
    with pytest.raises(backup.BackupError, match='different filesystem'):
        backup.store_path(project, sibling, 'abc')


def make_generation(project, store, pid='id'):
    store.mkdir()
    backup.atomic_json(store / 'identity.json', {'format': backup.FORMAT, 'project_id': pid})
    staging = store / '.incomplete-test'
    staging.mkdir()
    entries = backup.scan(project, exclude_state=True)
    archive_files(store, project, entries)
    db = {}
    for name in ('database.dump', 'roles.sql'):
        (staging / name).write_bytes(name.encode())
        db[name] = {'size': len(name), 'sha256': backup.digest(staging / name)}
        backup.put_object(store, staging / name, db[name])
        (staging / name).unlink()
    doc = {'format': backup.FORMAT, 'project_id': pid, 'files': entries, 'database_files': db, 'root_mode': 0o755}
    gen = backup.publish(store, staging, doc)
    return gen, doc


def test_pointer_checksum_and_payload_corruption(project, tmp_path):
    store = tmp_path / 'backup'
    gen, doc = make_generation(project, store)
    assert backup.current(store) == (gen, doc)
    backup.verify_generation(gen, doc)
    backup.object_path(store, doc['files']['.env']['sha256']).write_text('corruption')
    with pytest.raises(backup.BackupError, match='missing or damaged'):
        backup.verify_generation(gen, doc)
    (gen / 'manifest.json').write_text('{}')
    with pytest.raises(backup.BackupError, match='checksum'):
        backup.current(store)


def test_failed_publication_keeps_current_generation(project, tmp_path, monkeypatch):
    store = tmp_path / 'backup'
    gen, doc = make_generation(project, store)
    pending = store / '.incomplete-next'
    pending.mkdir()
    original = backup.atomic_json
    def fail(path, value):
        if path.name == 'CURRENT':
            raise OSError('simulated disk failure')
        original(path, value)
    monkeypatch.setattr(backup, 'atomic_json', fail)
    with pytest.raises(OSError):
        backup.publish(store, pending, doc)
    assert backup.current(store) == (gen, doc)
    backup.verify_generation(gen, doc)


def test_restore_into_new_folder_preserves_payload_and_identity(project, tmp_path):
    store = tmp_path / 'backup'
    gen, doc = make_generation(project, store)
    target = tmp_path / 'restored'
    backup.restore_files(store, target)
    assert backup.scan(target, exclude_state=True) == doc['files']
    assert (target / '.cold_storage/database.dump').read_bytes() == b'database.dump'
    assert backup.read_json(target / '.cold_storage/identity.json')['project_id'] == 'id'
    with pytest.raises(backup.BackupError, match='must not exist'):
        backup.restore_files(store, target)


def test_wrong_destination_identity_and_symlink_refused(tmp_path):
    store = tmp_path / 'backup'
    store.mkdir()
    backup.atomic_json(store / 'identity.json', {'format': backup.FORMAT, 'project_id': 'other'})
    with pytest.raises(backup.BackupError, match='different project'):
        with backup.destination(store, 'mine'):
            pass
    alias = tmp_path / 'alias'
    alias.symlink_to(store)
    with pytest.raises(backup.BackupError):
        with backup.destination(alias, 'other'):
            pass


def test_prune_keeps_verified_current(project, tmp_path, monkeypatch):
    store = tmp_path / 'backup'
    gen, doc = make_generation(project, store)
    stale = store / '.incomplete-abandoned'
    stale.mkdir()
    (stale / 'partial').write_text('partial')
    monkeypatch.setattr('builtins.input', lambda _: 'YES')
    backup.prune(store)
    assert not stale.exists()
    assert backup.current(store) == (gen, doc)


def test_disk_selection_does_not_accept_plain_directory(tmp_path):
    with pytest.raises(backup.BackupError, match='mounted'):
        backup.choose_mount(str(tmp_path))


def test_dump_commands_use_snapshot_no_password_and_verify_archive(tmp_path, monkeypatch):
    calls = []
    def fake(args, **kwargs):
        calls.append(args)
        for arg in args:
            if arg.startswith('--file='):
                Path(arg.split('=', 1)[1]).write_bytes(b'dump')
        return b''
    monkeypatch.setattr(backup, 'command', fake)
    result = backup.dump_database(tmp_path, {}, 'snapshot-123')
    assert '--snapshot=snapshot-123' in calls[0]
    assert '--create' in calls[0] and '--no-password' in calls[0]
    assert '--globals-only' in calls[1]
    assert calls[2][:2] == ['pg_restore', '--list']
    assert set(result) == {'database.dump', 'roles.sql'}


@pytest.fixture
def offline_snapshot(project, monkeypatch):
    """Snapshot filesystem/failure tests use fake DB bytes, never the user's DB."""
    from contextlib import contextmanager
    @contextmanager
    def frozen(root):
        yield {}, 'test-snapshot', {'name': 'fixture', 'size_bytes': 100, 'server_version_num': '160000'}
    def dump(path, env, token):
        result = {}
        for name in ('database.dump', 'roles.sql'):
            (path / name).write_bytes(b'consistent fixture database')
            result[name] = {'size': (path/name).stat().st_size, 'sha256': backup.digest(path/name)}
        return result
    monkeypatch.setattr(backup, 'frozen_database', frozen)
    monkeypatch.setattr(backup, 'dump_database', dump)
    monkeypatch.setattr(backup, 'no_workers', lambda root: None)
    monkeypatch.setattr('builtins.input', lambda _: 'YES')
    return project


def test_sync_twice_and_scan_compare_semantics(offline_snapshot, tmp_path):
    project = offline_snapshot
    store = tmp_path / 'disk'
    with backup.local_state(project) as (local, pid):
        with backup.destination(store, pid, create=True):
            backup.snapshot(project, local, pid, store)
            first, old = backup.current(store)
            (project / 'data/market.bin').write_bytes(b'updated data')
            backup.snapshot(project, local, pid)
            local_doc = backup.read_json(local / 'manifest.json')
            assert any(r == {'path': 'data/market.bin', 'status': 'M'} for r in backup.changes(old['files'], local_doc['files']))
            backup.snapshot(project, local, pid, store)
            second, new = backup.current(store)
            assert second != first
            backup.verify_generation(first, old)
            backup.verify_generation(second, new)
            assert old['files']['code.py']['sha256'] == new['files']['code.py']['sha256']
            assert not (first / 'project').exists() and not (second / 'project').exists()
            assert not backup.changes(new['files'], backup.read_json(local/'manifest.json')['files'])


def test_source_changes_during_copy_does_not_commit(offline_snapshot, tmp_path, monkeypatch):
    project = offline_snapshot
    store = tmp_path / 'disk'
    original = backup.put_object
    with backup.local_state(project) as (local, pid), backup.destination(store, pid, create=True):
        backup.snapshot(project, local, pid, store)
        committed = backup.current(store)
        def changed(*args, **kwargs):
            original(*args, **kwargs)
            (project / 'data/market.bin').write_bytes(b'concurrent writer')
        monkeypatch.setattr(backup, 'put_object', changed)
        with pytest.raises(backup.BackupError, match='Source changed'):
            backup.snapshot(project, local, pid, store)
        assert backup.current(store) == committed
        backup.verify_generation(*committed)


def test_dump_failure_preserves_committed_generation(offline_snapshot, tmp_path, monkeypatch):
    project = offline_snapshot
    store = tmp_path / 'disk'
    with backup.local_state(project) as (local, pid), backup.destination(store, pid, create=True):
        backup.snapshot(project, local, pid, store)
        committed = backup.current(store)
        def fail(*args):
            raise backup.BackupError('dump failed')
        monkeypatch.setattr(backup, 'dump_database', fail)
        with pytest.raises(backup.BackupError, match='dump failed'):
            backup.snapshot(project, local, pid, store)
        assert backup.current(store) == committed
        assert not list(store.glob('.incomplete-*'))


def test_not_enough_space_does_not_copy_or_dump(offline_snapshot, tmp_path, monkeypatch):
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')
    project = offline_snapshot
    store = tmp_path / 'disk'
    with backup.local_state(project) as (local, pid), backup.destination(store, pid, create=True):
        monkeypatch.setattr(backup.shutil, 'disk_usage', lambda _: usage(10, 10, 0))
        monkeypatch.setattr(backup, 'dump_database', lambda *a: pytest.fail('must not dump'))
        with pytest.raises(backup.BackupError, match='Not enough free'):
            backup.snapshot(project, local, pid, store)
        assert backup.current(store) == (None, None)


def test_postgresql_round_trip_in_isolated_database(project, tmp_path, monkeypatch):
    """Creates/drops only a random database; never restores over user data."""
    import shutil
    import uuid
    raw = os.environ.get('COLD_STORAGE_TEST_DATABASE_URL')
    if not raw:
        from train_and_eval.database.session import get_database_url
        raw = get_database_url()
    for binary in ('pg_dump', 'pg_dumpall', 'pg_restore'):
        if not shutil.which(binary):
            pytest.fail(f'Integration test requires {binary}')
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    url = make_url(raw)
    name = 'cold_backup_test_' + uuid.uuid4().hex
    admin = create_engine(url.set(database='postgres'), isolation_level='AUTOCOMMIT')
    test_url = url.set(database=name)
    source = None
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        source = create_engine(test_url)
        with source.begin() as conn:
            conn.execute(text("CREATE TABLE runs (id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, status text, value text)"))
            conn.execute(text("INSERT INTO runs (status, value) VALUES ('completed', 'Zażółć gęślą')"))
            conn.execute(text("SELECT setval(pg_get_serial_sequence('runs', 'id'), 200, true)"))
        source.dispose()
        monkeypatch.setenv('DATABASE_URL', test_url.render_as_string(hide_password=False))
        monkeypatch.setattr('builtins.input', lambda _: 'YES')
        from sqlalchemy.exc import DBAPIError
        with backup.frozen_database(project):
            with source.connect() as writer:
                writer.execute(text("SET lock_timeout = '100ms'"))
                with pytest.raises(DBAPIError):
                    writer.execute(text("INSERT INTO runs (status) VALUES ('completed')"))
        source.dispose()
        store = tmp_path / 'disk'
        with backup.local_state(project) as (local, pid), backup.destination(store, pid, create=True):
            backup.snapshot(project, local, pid, store)
        restored = tmp_path / 'restored'
        backup.restore_files(store, restored)
        with pytest.raises(backup.BackupError, match='already exists'):
            backup.restore_database(restored, restored / backup.STATE)
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}"'))
        backup.restore_database(restored, restored / backup.STATE)
        source = create_engine(test_url)
        with source.connect() as conn:
            assert conn.execute(text('SELECT id, status, value FROM runs')).one() == (1, 'completed', 'Zażółć gęślą')
            assert conn.execute(text("SELECT nextval(pg_get_serial_sequence('runs', 'id'))")).scalar_one() == 201
    finally:
        if source is not None:
            source.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_destination_never_needs_posix_metadata_or_links(offline_snapshot, tmp_path, monkeypatch):
    project = offline_snapshot
    store = tmp_path / 'exfat'
    def guard(original):
        def wrapped(path, *args, **kwargs):
            # exFAT rejects POSIX metadata operations: exercise that boundary.
            if not isinstance(path, int):
                p = Path(path).absolute()
                if p == store or store in p.parents:
                    raise OSError('simulated exFAT: unsupported operation')
            return original(path, *args, **kwargs)
        return wrapped
    for name in ('chmod', 'chown', 'utime', 'setxattr', 'removexattr', 'symlink', 'link'):
        monkeypatch.setattr(backup.os, name, guard(getattr(backup.os, name)))
    (project / 'data/Case').write_text('A')
    (project / 'data/case').write_text('a')
    (project / 'data/symbolic').symlink_to('Case')
    with backup.local_state(project) as (local, pid), backup.destination(store, pid, create=True):
        backup.snapshot(project, local, pid, store)
        gen, doc = backup.current(store)
        backup.verify_generation(gen, doc)
        assert (store / 'RESTORE.py').is_file()
        target = tmp_path / 'restored'
        backup.restore_files(store, target)
        assert backup.scan(target, exclude_state=True) == doc['files']


def test_prune_preserves_shared_content_and_removes_orphans(offline_snapshot, tmp_path):
    project = offline_snapshot
    store = tmp_path / 'backup'
    with backup.local_state(project) as (local, pid), backup.destination(store, pid, create=True):
        backup.snapshot(project, local, pid, store)
        first, old = backup.current(store)
        obsolete = backup.object_path(store, old['files']['data/market.bin']['sha256'])
        shared = backup.object_path(store, old['files']['code.py']['sha256'])
        (project / 'data/market.bin').write_bytes(b'new')
        backup.snapshot(project, local, pid, store)
        second, new = backup.current(store)
        orphan_file = local / 'orphan'
        orphan_file.write_bytes(b'orphan')
        orphan = {'sha256': backup.digest(orphan_file), 'size': 6}
        backup.put_object(store, orphan_file, orphan)
        backup.prune(store)
        assert not first.exists() and second.exists()
        assert not obsolete.exists()
        assert shared.exists()
        assert not backup.object_path(store, orphan['sha256']).exists()
        backup.verify_generation(second, new)


def test_corrupt_object_refused_without_replacing_it(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'abc')
    entry = {'size': 3, 'sha256': backup.digest(source)}
    store = tmp_path / 'store'
    store.mkdir()
    backup.put_object(store, source, entry)
    target = backup.object_path(store, entry['sha256'])
    target.write_bytes(b'xyz')
    with pytest.raises(backup.BackupError, match='damaged'):
        backup.put_object(store, source, entry)
    assert target.read_bytes() == b'xyz'


def test_duplicate_bytes_with_different_metadata_restore_separately(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'a').write_bytes(b'same')
    (root / 'b').write_bytes(b'same')
    (root / 'a').chmod(0o600)
    (root / 'b').chmod(0o755)
    entries = backup.scan(root)
    store = tmp_path / 'store'
    archive_files(store, root, entries)
    assert len(list((store / 'objects').glob('*/*'))) == 1
    target = tmp_path / 'restored'
    backup.restore_tree(store, target, entries)
    assert backup.scan(target) == entries
    assert (target / 'a').stat().st_ino != (target / 'b').stat().st_ino


def test_object_shard_symlink_cannot_escape_store(tmp_path):
    store = tmp_path / 'store'
    store.mkdir()
    (store / 'objects').symlink_to(tmp_path)
    with pytest.raises(backup.BackupError):
        backup.object_path(store, 'a' * 64)


def test_probe_cleans_temporary_files(tmp_path):
    local, disk = tmp_path / 'local', tmp_path / 'disk'
    local.mkdir()
    disk.mkdir()
    backup.probe(local, disk)
    assert list(local.iterdir()) == []
    assert list(disk.iterdir()) == []
