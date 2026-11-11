"""Timestamp-only reports preserve batch identity and reject collisions."""
from datetime import UTC, datetime

import pytest

from train_and_eval.optuna import report


def test_timestamp_only_and_collision_does_not_overwrite(tmp_path, monkeypatch):
    class Clock:
        @classmethod
        def now(cls, zone):
            assert zone == UTC
            return datetime(2026, 11, 11, 12, 34, 56, 123456, tzinfo=UTC)

    monkeypatch.setattr(report, 'datetime', Clock)
    data = {'status': 'complete', 'config': {}}
    directory = report.save(data, tmp_path)
    assert directory.name == '20261111T123456123456Z'
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        report.save({'status': 'blocked', 'config': {}}, tmp_path)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert list(tmp_path.iterdir()) == [directory]
