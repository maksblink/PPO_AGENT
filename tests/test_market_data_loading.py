from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from train_and_eval.market_data import load_market_data as loader

from train_and_eval.market_data.load_market_data import (
    MANIFEST_VERSION,
    MarketDataLoadingError,
    calculate_file_sha256,
    load_market_data,
    manifest_key,
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



def _prepare(tmp_path, *, warnings=False):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "market.parquet"
    table = _valid_table()
    pq.write_table(table, path)
    entry = {
        "path": path.name, "interval": "5m", "data_schema_version": 1,
        "validation_status": "passed", "rows": 3,
        "size_bytes": path.stat().st_size,
        "sha256": calculate_file_sha256(path),
        "first_timestamp": "2026-01-05T00:00:00Z",
        "last_timestamp": "2026-01-05T00:10:00Z",
    }
    document = {
        "manifest_version": 2, "data_schema_version": 1,
        "dataset_id": "NQ_CONTINUOUS_WEEKDAYS", "release_id": "test_release",
        "status": "published_with_warnings" if warnings else "published",
        "files": {"5m": entry},
        "validation": {
            "status": "passed_with_warnings" if warnings else "passed",
            "structural_error_count": 0,
            "warning_event_count": int(warnings),
            "warning_counts": {"missing_minute": int(warnings)},
            "warning_acceptance": {
                "required": warnings, "accepted": True if warnings else None,
                "mode": "interactive" if warnings else "not_required",
                "accepted_at_utc": "2026-01-06T00:00:00Z" if warnings else None,
            },
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return path, data, manifest, document


@pytest.mark.parametrize("warnings", [False, True])
def test_load_preserves_training_interface_and_manifest(tmp_path, warnings):
    path, data, manifest, document = _prepare(tmp_path, warnings=warnings)
    before = manifest.read_bytes()
    for supplied in (path, "market.parquet", "data/market.parquet"):
        frame = load_market_data(supplied, data_directory=data, manifest_path=manifest)
        assert list(frame) == ["DT", "Open", "High", "Low", "Close", "Volume", "symbol", "instrument_id"]
        assert frame.Close.tolist() == [101.0, 102.0, 103.0]
        assert frame.Volume.tolist() == [10, 20, 30]
        assert str(frame.DT.dtype) == "datetime64[ns, UTC]"
        assert frame.attrs["source_path"] == "data/market.parquet"
        assert frame.attrs["sha256"] == document["files"]["5m"]["sha256"]
        assert frame.attrs["release_id"] == "test_release"
        assert frame.attrs["interval"] == "5m"
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("keys,value", [
    (("manifest_version",), 1),
    (("manifest_version",), True),
    (("data_schema_version",), 2),
    (("status",), "staged"),
    (("validation", "structural_error_count"), 1),
    (("validation", "warning_event_count"), -1),
    (("validation", "warning_event_count"), True),
    (("validation", "warning_acceptance", "accepted"), False),
    (("validation", "warning_acceptance", "accepted"), "true"),
    (("validation", "warning_acceptance", "required"), False),
    (("validation", "warning_acceptance", "mode"), "unknown"),
    (("validation", "warning_acceptance", "accepted_at_utc"), None),
    (("files", "5m", "validation_status"), "failed"),
    (("files", "5m", "interval"), "1m"),
    (("files", "5m", "rows"), True),
    (("files", "5m", "rows"), 0),
    (("files", "5m", "sha256"), "z" * 64),
    (("files", "5m", "size_bytes"), -1),
    (("files", "5m", "path"), "../market.parquet"),
    (("files", "5m", "path"), "/tmp/market.parquet"),
    (("files", "5m", "path"), "a/../market.parquet"),
    (("files", "5m", "path"), "a\\market.parquet"),
    (("files", "5m", "first_timestamp"), "2026-01-05T00:00:00"),
    (("files", "5m", "last_timestamp"), "2026-01-05T00:11:00Z"),
])
def test_reject_invalid_manifest_fields(tmp_path, keys, value):
    path, data, manifest, document = _prepare(tmp_path, warnings=True)
    parent = document
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value
    manifest.write_text(json.dumps(document))
    before = manifest.read_bytes()
    with pytest.raises(MarketDataLoadingError):
        load_market_data(path, data_directory=data, manifest_path=manifest)
    assert manifest.read_bytes() == before


def test_cli_warning_acceptance_is_supported(tmp_path):
    _, _, manifest, document = _prepare(tmp_path, warnings=True)
    document["validation"]["warning_acceptance"]["mode"] = "cli_flag"
    manifest.write_text(json.dumps(document))
    assert loader.load_manifest(manifest)["status"] == "published_with_warnings"


@pytest.mark.parametrize("content", ["[]", "null", "{", '{"files":{},"files":{}}'])
def test_reject_malformed_or_ambiguous_json(tmp_path, content):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(content)
    with pytest.raises(MarketDataLoadingError):
        loader.load_manifest(manifest)


def test_missing_manifest_does_not_recommend_creating_one(tmp_path):
    with pytest.raises(MarketDataLoadingError, match="Import the published manifest"):
        loader.load_manifest(tmp_path / "absent.json")


@pytest.mark.parametrize("same_size", [True, False])
def test_reject_changed_file(tmp_path, same_size):
    path, data, manifest, _ = _prepare(tmp_path)
    payload = bytearray(path.read_bytes())
    if same_size:
        payload[10] ^= 1
    else:
        payload.extend(b"changed")
    path.write_bytes(payload)
    with pytest.raises(MarketDataLoadingError, match="has changed"):
        load_market_data(path, data_directory=data, manifest_path=manifest)


@pytest.mark.parametrize("field,value", [("rows", 4), ("last_timestamp", "2026-01-05T00:15:00Z")])
def test_reject_incorrect_loaded_metadata(tmp_path, field, value):
    path, data, manifest, document = _prepare(tmp_path)
    document["files"]["5m"][field] = value
    manifest.write_text(json.dumps(document))
    with pytest.raises(MarketDataLoadingError, match="differs from manifest"):
        load_market_data(path, data_directory=data, manifest_path=manifest)


def test_file_must_have_an_exact_manifest_path(tmp_path):
    path, data, manifest, document = _prepare(tmp_path)
    document["files"]["5m"]["path"] = "other.parquet"
    manifest.write_text(json.dumps(document))
    with pytest.raises(MarketDataLoadingError, match="no validation entry"):
        load_market_data(path, data_directory=data, manifest_path=manifest)


def test_symlink_cannot_escape_data_directory(tmp_path):
    path, data, manifest, _ = _prepare(tmp_path)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(MarketDataLoadingError, match="inside"):
        load_market_data(path, data_directory=data, manifest_path=manifest)


def test_duplicate_file_paths_rejected(tmp_path):
    _, _, manifest, document = _prepare(tmp_path)
    document["files"]["1m"] = dict(document["files"]["5m"], interval="1m")
    manifest.write_text(json.dumps(document))
    with pytest.raises(MarketDataLoadingError, match="Duplicate manifest file path"):
        loader.load_manifest(manifest)


def test_verify_result_keeps_local_path_without_rewriting_manifest(tmp_path):
    path, data, manifest, document = _prepare(tmp_path)
    _, entry = loader.verify_market_data(path, data_directory=data, manifest_path=manifest)
    assert entry["path"] == "data/market.parquet"
    assert entry["manifest_file_path"] == "market.parquet"
    assert json.loads(manifest.read_text()) == document


def test_reject_wrong_arrow_type_even_with_matching_checksum(tmp_path):
    path, data, manifest, document = _prepare(tmp_path)
    table = _valid_table().set_column(4, "Close_NQ", pa.array([101, 102, 103], type=pa.int64()))
    pq.write_table(table, path)
    document["files"]["5m"].update(sha256=calculate_file_sha256(path), size_bytes=path.stat().st_size)
    manifest.write_text(json.dumps(document))
    with pytest.raises(MarketDataLoadingError, match="Parquet type"):
        load_market_data(path, data_directory=data, manifest_path=manifest)


def test_reject_modification_during_loading(tmp_path, monkeypatch):
    path, data, manifest, _ = _prepare(tmp_path)
    real_read = pd.read_parquet
    def modifying_read(*args, **kwargs):
        frame = real_read(*args, **kwargs)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return frame
    monkeypatch.setattr(pd, "read_parquet", modifying_read)
    with pytest.raises(MarketDataLoadingError, match="while loading"):
        load_market_data(path, data_directory=data, manifest_path=manifest)
