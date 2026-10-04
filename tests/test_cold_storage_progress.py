from io import StringIO

import pytest
from tests.test_cold_storage import backup, project, offline_snapshot


class Terminal(StringIO):
    def isatty(self):
        return True


def test_progress_bar_and_stage_timing():
    now = [10.0]
    output = Terminal()
    p = backup.Progress(output, clock=lambda: now[0])
    with p.phase('Hash', 100):
        p.advance(50)
        now[0] = 12.0
        p.render()
        p.advance(50)
        now[0] = 14.0
    p.finish('OK')
    assert '50.0%' in output.getvalue()
    assert '100.0%' in output.getvalue()
    assert 'ETA' in output.getvalue()
    assert '\x1b[2J' not in output.getvalue()
    assert p.rows[0]['seconds'] == 4.0
    assert p.rows[0]['bytes'] == 100


def test_unknown_total_does_not_invent_percentage():
    output = Terminal()
    p = backup.Progress(output)
    with p.phase('Database'):
        p.render()
    assert 'working' in output.getvalue()
    assert '%' not in output.getvalue()


def test_non_tty_is_compact_and_failed_stage_is_recorded():
    output = StringIO()
    p = backup.Progress(output)
    with pytest.raises(RuntimeError):
        with p.phase('Copy', 10):
            p.advance(3)
            for _ in range(100):
                p.render()
            raise RuntimeError('injected')
    assert len(output.getvalue().splitlines()) == 2
    assert '\r' not in output.getvalue()
    assert p.rows[0]['status'] == 'FAILED/INTERRUPTED'


def test_report_only_prints_counts(capsys):
    backup.report([{'path': 'secret/name', 'status': 'N'}, {'path': 'other', 'status': 'D'}])
    output = capsys.readouterr().out
    assert 'N=1' in output and 'M=0' in output and 'D=1' in output
    assert 'secret/name' not in output


def test_cancel_keeps_complete_plan_without_committed_manifest(offline_snapshot, tmp_path, monkeypatch):
    root = offline_snapshot
    store = tmp_path / 'backup'
    monkeypatch.setattr('builtins.input', lambda _: 'NO')
    with backup.local_state(root) as (local, pid), backup.destination(store, pid, create=True):
        with pytest.raises(backup.BackupError, match='Cancelled'):
            backup.snapshot(root, local, pid, store)
        plan = backup.read_json(local / 'plan.json')
        assert plan['status'] == 'PLANNED_NOT_BACKED_UP'
        assert plan['files']['data/market.bin']['sha256'] == backup.digest(root/'data/market.bin')
        assert plan['changes']
        assert not (local / 'manifest.json').exists()
        assert backup.current(store) == (None, None)


def test_progress_preserves_full_round_trip_and_saves_summary(offline_snapshot, tmp_path, monkeypatch):
    root = offline_snapshot
    store = tmp_path / 'backup'
    output = StringIO()
    p = backup.Progress(output)
    monkeypatch.setattr(backup, 'PROGRESS', p)
    with backup.local_state(root) as (local, pid), backup.destination(store, pid, create=True):
        p.summary_path = local / 'last_operation_summary.json'
        backup.snapshot(root, local, pid, store)
        gen, doc = backup.current(store)
        backup.verify_generation(gen, doc)
        p.finish('OK')
        saved = backup.read_json(p.summary_path)
        assert saved['status'] == 'OK'
        assert saved['copied_objects'] > 0 and saved['copied_bytes'] > 0
        assert any(r['name'] == 'Store content + checksums' for r in saved['stages'])
        assert any(r['name'] == 'Recheck source' for r in saved['stages'])
        for row in saved['stages']:
            if row['total'] is not None:
                assert row['bytes'] == row['total']
        # Repeated sync reuses objects instead of counting them as copied.
        copied_before = p.copied_objects
        backup.snapshot(root, local, pid, store)
        assert p.reused_objects > 0
        assert p.copied_objects == copied_before
