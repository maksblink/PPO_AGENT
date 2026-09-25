"""Queue deletion uses the existing isolated PostgreSQL cleanup fixtures."""
from datetime import datetime, timezone
from types import SimpleNamespace
import pytest
from sqlalchemy import select, update
from train_and_eval.database.models import Run, Checkpoint, TrainingMetric
from tests.test_clean_training import cleanup_database, seed
import run_search_queue as queue


@pytest.mark.parametrize('status', ['pending', 'failed', 'cancelled'])
def test_queue_deletes_only_confirmed_incomplete_run(cleanup_database, tmp_path, monkeypatch, status):
    engine = cleanup_database
    ids = seed(engine, tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    if status != 'pending':
        now = datetime.now(timezone.utc)
        with engine.begin() as c:
            c.execute(update(Run).where(Run.id == ids['other']).values(status=status,
                      started_at=now, finished_at=now,
                      error_type='TestError' if status == 'failed' else None,
                      error_message='Test failure' if status == 'failed' else None))
    snapshot = queue.inspect_incomplete_run(engine, ids['other'], 'other')
    queue.delete_incomplete_run(engine, SimpleNamespace(name='other'), snapshot)
    with engine.connect() as c:
        assert c.scalar(select(Run.id).where(Run.id == ids['other'])) is None
        assert c.scalar(select(Checkpoint.id).where(Checkpoint.run_id == ids['other'])) is None
        assert c.scalar(select(TrainingMetric.id).where(TrainingMetric.run_id == ids['other'])) is None
        assert c.scalar(select(Run.id).where(Run.id == ids['first'])) == ids['first']
    assert not (tmp_path/f"artifacts/runs/{ids['other']:08d}").exists()
    assert (tmp_path/f"artifacts/runs/{ids['first']:08d}/checkpoints/source.zip").read_text() == 'weights'
    assert not (tmp_path/'artifacts/.cleanup_pending.json').exists()


def test_queue_refuses_changed_run(cleanup_database, tmp_path, monkeypatch):
    engine = cleanup_database; ids = seed(engine, tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    snapshot = queue.inspect_incomplete_run(engine, ids['other'], 'other')
    with engine.begin() as c:
        c.execute(update(Run).where(Run.id == ids['other']).values(normalized_config_json={'changed': True}))
    with pytest.raises(ValueError, match='changed since'):
        queue.delete_incomplete_run(engine, SimpleNamespace(name='other'), snapshot)
    assert (tmp_path/f"artifacts/runs/{ids['other']:08d}").exists()


def test_queue_refuses_run_with_descendants(cleanup_database, tmp_path, monkeypatch):
    engine = cleanup_database; ids = seed(engine, tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    snapshot = queue.inspect_incomplete_run(engine, ids['first'], 'first')
    with pytest.raises(ValueError, match='depends'):
        queue.delete_incomplete_run(engine, SimpleNamespace(name='first'), snapshot)
    assert (tmp_path/f"artifacts/runs/{ids['first']:08d}").exists()


def test_queue_refuses_running_work(cleanup_database, tmp_path, monkeypatch):
    engine = cleanup_database; ids = seed(engine, tmp_path)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    with engine.begin() as c:
        c.execute(update(Run).where(Run.id == ids['other']).values(status='running', started_at=datetime.now(timezone.utc)))
    snapshot = queue.inspect_incomplete_run(engine, ids['other'], 'other')
    with pytest.raises(ValueError, match='Running work'):
        queue.delete_incomplete_run(engine, SimpleNamespace(name='other'), snapshot)
    assert (tmp_path/f"artifacts/runs/{ids['other']:08d}").exists()
