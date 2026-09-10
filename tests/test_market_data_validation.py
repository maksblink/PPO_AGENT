from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from train_and_eval.market_data import validate_market_data as validator
from train_and_eval.market_data.load_market_data import calculate_file_sha256

from train_and_eval.market_data.validate_market_data import (
    validate_market_data_file,
)


def _valid_table() -> pa.Table:
    timestamps = pd.date_range(
        "2026-01-05 00:00:00",
        periods=3,
        freq="5min",
        tz="UTC",
    )

    return pa.table(
        {
            "Datetime": pa.array(
                timestamps,
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "Open_NQ": pa.array(
                [100.0, 101.0, 102.0],
                type=pa.float64(),
            ),
            "High_NQ": pa.array(
                [102.0, 103.0, 104.0],
                type=pa.float64(),
            ),
            "Low_NQ": pa.array(
                [99.0, 100.0, 101.0],
                type=pa.float64(),
            ),
            "Close_NQ": pa.array(
                [101.0, 102.0, 103.0],
                type=pa.float64(),
            ),
            "Volume_NQ": pa.array(
                [10, 20, 30],
                type=pa.uint64(),
            ),
            "symbol": pa.array(
                ["NQ.v.0", "NQ.v.0", "NQ.v.0"],
                type=pa.large_string(),
            ),
            "instrument_id": pa.array(
                [6641, 6641, 6641],
                type=pa.uint32(),
            ),
        }
    )


def _invalid_ohlc_table() -> pa.Table:
    table = _valid_table()

    invalid_high = pa.array(
        [98.0, 103.0, 104.0],
        type=pa.float64(),
    )

    column_index = table.schema.get_field_index("High_NQ")

    return table.set_column(
        column_index,
        "High_NQ",
        invalid_high,
    )


def _prepare(tmp_path, table=None):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "market.parquet"
    pq.write_table(_valid_table() if table is None else table, path)
    manifest = tmp_path / "manifest.json"
    document = {
        "manifest_version": 2, "data_schema_version": 1,
        "dataset_id": "NQ_CONTINUOUS_WEEKDAYS", "release_id": "test_release",
        "status": "published",
        "files": {"5m": {
            "path": path.name, "interval": "5m", "data_schema_version": 1,
            "validation_status": "passed", "rows": 3,
            "size_bytes": path.stat().st_size, "sha256": calculate_file_sha256(path),
            "first_timestamp": "2026-01-05T00:00:00Z",
            "last_timestamp": "2026-01-05T00:10:00Z",
        }},
        "validation": {
            "status": "passed", "structural_error_count": 0,
            "warning_event_count": 0, "warning_counts": {},
            "warning_acceptance": {
                "required": False, "accepted": None,
                "mode": "not_required", "accepted_at_utc": None,
            },
        },
    }
    manifest.write_text(json.dumps(document))
    return path, data, manifest


def test_valid_file_does_not_change_manifest_or_data(tmp_path):
    path, data, manifest = _prepare(tmp_path)
    before = {p: p.read_bytes() for p in (path, manifest)}
    record = validate_market_data_file(path, data_directory=data, manifest_path=manifest)
    assert record["status"] == "okay"
    assert record["rows"] == 3
    assert record["errors"] == []
    assert all(p.read_bytes() == contents for p, contents in before.items())


def test_invalid_ohlc_is_rejected_without_modifying_manifest(tmp_path):
    path, data, manifest = _prepare(tmp_path, _invalid_ohlc_table())
    before = manifest.read_bytes()
    record = validate_market_data_file(path, data_directory=data, manifest_path=manifest)
    assert record["status"] == "not_okay"
    assert "Invalid OHLC relationship" in record["errors"][0]
    assert manifest.read_bytes() == before


def test_revalidation_never_approves_changed_file(tmp_path):
    path, data, manifest = _prepare(tmp_path)
    before = manifest.read_bytes()
    assert validate_market_data_file(path, data_directory=data, manifest_path=manifest)["status"] == "okay"
    pq.write_table(_invalid_ohlc_table(), path)
    record = validate_market_data_file(path, data_directory=data, manifest_path=manifest)
    assert record["status"] == "not_okay"
    assert "has changed" in record["errors"][0]
    assert manifest.read_bytes() == before


def test_missing_manifest_is_not_created(tmp_path):
    path, data, manifest = _prepare(tmp_path)
    manifest.unlink()
    record = validate_market_data_file(path, data_directory=data, manifest_path=manifest)
    assert record["status"] == "not_okay"
    assert not manifest.exists()


@pytest.mark.parametrize("missing_file", [False, True])
def test_cli_all_uses_manifest_and_is_read_only(tmp_path, monkeypatch, capsys, missing_file):
    path, data, manifest = _prepare(tmp_path)
    before = manifest.read_bytes()
    if missing_file:
        path.unlink()
    else:
        # An unrelated local file must not be silently registered in the release.
        (data / "unlisted.parquet").write_bytes(b"unlisted")
    monkeypatch.setattr(validator, "DATA_DIRECTORY", data)
    monkeypatch.setattr(validator, "MANIFEST_PATH", manifest)
    monkeypatch.setattr("sys.argv", ["validate_market_data", "--all"])
    assert validator.main() == (1 if missing_file else 0)
    assert manifest.read_bytes() == before
    assert "unlisted.parquet" not in capsys.readouterr().out


def test_cli_all_cannot_reset_v1_manifest(tmp_path, monkeypatch):
    _, data, manifest = _prepare(tmp_path)
    manifest.write_text('{"manifest_version":1,"files":{}}')
    before = manifest.read_bytes()
    monkeypatch.setattr(validator, "DATA_DIRECTORY", data)
    monkeypatch.setattr(validator, "MANIFEST_PATH", manifest)
    monkeypatch.setattr("sys.argv", ["validate_market_data", "--all"])
    assert validator.main() == 1
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("kind,expected", [
    ("duplicate", "duplicate timestamps"),
    ("order", "strictly increasing"),
    ("null", "missing values"),
    ("infinite", "non-finite"),
    ("price", "greater than zero"),
    ("empty_symbol", "Empty symbol"),
    ("alignment", "Unaligned"),
])
def test_structural_checks_remain_active(tmp_path, kind, expected):
    table = _valid_table()
    if kind in {"duplicate", "order", "alignment"}:
        times = table["Datetime"].to_pylist()
        if kind == "duplicate":
            times[1] = times[0]
        elif kind == "order":
            times[0], times[1] = times[1], times[0]
        else:
            times[1] += pd.Timedelta(minutes=1)
        table = table.set_column(0, "Datetime", pa.array(times, type=pa.timestamp("ns", tz="UTC")))
    elif kind == "empty_symbol":
        table = table.set_column(6, "symbol", pa.array(["", "NQ.v.0", "NQ.v.0"], type=pa.large_string()))
    else:
        value = {"null": None, "infinite": float("inf"), "price": 0.0}[kind]
        table = table.set_column(1, "Open_NQ", pa.array([value, 101.0, 102.0], type=pa.float64()))
    path, data, manifest = _prepare(tmp_path, table)
    before = manifest.read_bytes()
    record = validate_market_data_file(path, data_directory=data, manifest_path=manifest)
    assert record["status"] == "not_okay"
    assert expected in record["errors"][0]
    assert manifest.read_bytes() == before
