"""Latest-only retention and interruption recovery on real temporary files."""
from pathlib import Path
import pytest
from tests.test_cold_storage import backup, project, offline_snapshot


def generations(store):
    return sorted(p for p in store.iterdir() if p.is_dir() and backup.re.fullmatch(r'[0-9TZ]+_[a-f0-9]{12}',p.name))


def test_sync_replaces_changed_removes_deleted_reuses_unchanged(offline_snapshot,tmp_path,monkeypatch):
    root=offline_snapshot;store=tmp_path/'backup'
    untouched=tmp_path/'other-personal-folder';untouched.mkdir();(untouched/'important').write_bytes(b'KEEP')
    (root/'data/deleted').write_bytes(b'unique obsolete payload')
    with backup.local_state(root) as (local,pid), backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store);first,old=backup.current(store)
        stable=backup.object_path(store,old['files']['code.py']['sha256']);identity=stable.stat()
        modified=backup.object_path(store,old['files']['data/market.bin']['sha256'])
        deleted=backup.object_path(store,old['files']['data/deleted']['sha256'])
        (root/'data/market.bin').write_bytes(b'brand new payload')
        (root/'data/deleted').unlink();(root/'data/new').write_bytes(b'new file')
        prompts=[]
        monkeypatch.setattr('builtins.input',lambda text: prompts.append(text) or 'YES')
        backup.snapshot(root,local,pid,store);second,new=backup.current(store)
        assert len(prompts)==1  # No separate prune confirmation.
        assert generations(store)==[second] and not first.exists()
        assert not modified.exists() and not deleted.exists()
        assert stable.stat().st_ino==identity.st_ino and stable.stat().st_mtime_ns==identity.st_mtime_ns
        assert 'data/new' in new['files'] and 'data/deleted' not in new['files']
        assert not (store/backup.CLEANUP_MARKER).exists()
        backup.verify_generation(second,new)
        restored=tmp_path/'restored';backup.restore_files(store,restored)
        assert (restored/'data/market.bin').read_bytes()==b'brand new payload'
        assert not (restored/'data/deleted').exists()
    assert (untouched/'important').read_bytes()==b'KEEP'


def test_copy_failure_keeps_old_complete_backup(offline_snapshot,tmp_path,monkeypatch):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store);old=backup.current(store)
        (root/'data/market.bin').write_bytes(b'updated')
        with monkeypatch.context() as m:
            m.setattr(backup,'put_object',lambda *args: (_ for _ in ()).throw(OSError('copy failed')))
            with pytest.raises(OSError):backup.snapshot(root,local,pid,store)
        assert backup.current(store)==old
        backup.verify_generation(*old)


def test_failed_pointer_update_preserves_old_and_resumes_cleanup(offline_snapshot,tmp_path,monkeypatch):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store);old=backup.current(store)
        (root/'data/market.bin').write_bytes(b'new version')
        original=backup.atomic_json
        def fail(path,value):
            if path==store/'CURRENT':raise OSError('pointer failed')
            return original(path,value)
        with monkeypatch.context() as m:
            m.setattr(backup,'atomic_json',fail)
            with pytest.raises(OSError):backup.snapshot(root,local,pid,store)
        assert backup.current(store)==old
        backup.verify_generation(*old)
        assert (store/backup.CLEANUP_MARKER).exists()
        backup.snapshot(root,local,pid,store)
        assert len(generations(store))==1
        assert not (store/backup.CLEANUP_MARKER).exists()
        backup.verify_generation(*backup.current(store))


def test_cleanup_interruption_retains_new_and_next_sync_recovers(offline_snapshot,tmp_path,monkeypatch):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store);first,_=backup.current(store)
        (root/'data/market.bin').write_bytes(b'new version')
        original=backup.shutil.rmtree
        def fail(path,*args,**kwargs):
            if path==first:raise OSError('cleanup failed')
            return original(path,*args,**kwargs)
        with monkeypatch.context() as m:
            m.setattr(backup.shutil,'rmtree',fail)
            with pytest.raises(backup.BackupError,match='latest backup is committed'):
                backup.snapshot(root,local,pid,store)
        second,new=backup.current(store)
        assert second!=first and (store/backup.CLEANUP_MARKER).exists()
        backup.verify_generation(second,new)
        backup.snapshot(root,local,pid,store)
        assert len(generations(store))==1
        assert not first.exists() and not second.exists()
        assert not (store/backup.CLEANUP_MARKER).exists()


def test_prune_failure_mid_object_deletion_is_repeatable(offline_snapshot,tmp_path,monkeypatch):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store)
        for i in range(2):
            f=local/f'extra{i}';f.write_bytes(f'extra payload {i}'.encode())
            backup.put_object(store,f,{'sha256':backup.digest(f),'size':f.stat().st_size})
        backup.atomic_json(store/backup.CLEANUP_MARKER,{'format':backup.FORMAT,'project_id':pid,'policy':'latest-only'})
        original=Path.unlink;count=[0]
        def fail(path,*args,**kwargs):
            if store/'objects' in path.parents:
                count[0]+=1
                if count[0]==2:raise OSError('interrupted')
            return original(path,*args,**kwargs)
        with monkeypatch.context() as m:
            m.setattr(Path,'unlink',fail)
            with pytest.raises(OSError):backup.prune(store,confirm=False)
        backup.verify_generation(*backup.current(store))
        backup.prune(store,confirm=False)
        assert not (store/backup.CLEANUP_MARKER).exists()
        current=backup.current(store)
        assert set(backup.references(current[1]))=={p.parent.name+p.name for p in (store/'objects').glob('*/*')}


def test_unknown_entries_and_sibling_files_preserved(offline_snapshot,tmp_path):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        (store/'my-notes').mkdir();(store/'my-notes/keep.txt').write_text('keep')
        backup.snapshot(root,local,pid,store)
        assert (store/'my-notes/keep.txt').read_text()=='keep'


def test_corrupt_current_refuses_prune_before_deletion(offline_snapshot,tmp_path):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store);gen,doc=backup.current(store)
        stale=store/'.incomplete-stale';stale.mkdir()
        backup.object_path(store,doc['files']['code.py']['sha256']).write_bytes(b'corrupt')
        with pytest.raises(backup.BackupError):backup.prune(store,confirm=False)
        assert stale.exists()


def test_wrong_cleanup_marker_refused(offline_snapshot,tmp_path):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.snapshot(root,local,pid,store)
        backup.atomic_json(store/backup.CLEANUP_MARKER,{'format':2,'project_id':'someone-else','policy':'latest-only'})
        with pytest.raises(backup.BackupError,match='identity'):backup.prune(store,confirm=False)


def test_old_identity_keeps_uuid_after_success(offline_snapshot,tmp_path):
    root=offline_snapshot;store=tmp_path/'backup'
    with backup.local_state(root) as (local,pid),backup.destination(store,pid,create=True):
        backup.atomic_json(local/'identity.json',{'format':1,'project_id':pid})
        backup.snapshot(root,local,pid,store)
        assert backup.read_json(local/'identity.json')=={'format':2,'project_id':pid}
