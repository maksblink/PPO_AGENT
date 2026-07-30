from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


def test_validation_marks_valid_file_as_okay(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    data_path = data_directory / "market.parquet"
    manifest_path = tmp_path / "manifest.json"

    pq.write_table(_valid_table(), data_path)

    record = validate_market_data_file(
        data_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    assert record["status"] == "okay"
    assert record["rows"] == 3
    assert len(record["sha256"]) == 64
    assert record["errors"] == []

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["files"]["data/market.parquet"]["status"] == "okay"


def test_validation_marks_invalid_ohlc_as_not_okay(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    data_path = data_directory / "market.parquet"
    manifest_path = tmp_path / "manifest.json"

    pq.write_table(_invalid_ohlc_table(), data_path)

    record = validate_market_data_file(
        data_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    assert record["status"] == "not_okay"
    assert record["rows"] is None
    assert len(record["sha256"]) == 64
    assert "Invalid OHLC relationship" in record["errors"][0]

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    entry = manifest["files"]["data/market.parquet"]
    assert entry["status"] == "not_okay"
    assert entry["errors"]


def test_revalidation_overwrites_previous_file_entry(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    data_path = data_directory / "market.parquet"
    manifest_path = tmp_path / "manifest.json"

    pq.write_table(_valid_table(), data_path)

    first_record = validate_market_data_file(
        data_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    pq.write_table(_invalid_ohlc_table(), data_path)

    second_record = validate_market_data_file(
        data_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert first_record["status"] == "okay"
    assert second_record["status"] == "not_okay"
    assert first_record["sha256"] != second_record["sha256"]
    assert len(manifest["files"]) == 1
    assert (
        manifest["files"]["data/market.parquet"]["status"]
        == "not_okay"
    )


def test_validation_preserves_entries_for_other_files(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    first_path = data_directory / "first.parquet"
    second_path = data_directory / "second.parquet"
    manifest_path = tmp_path / "manifest.json"

    pq.write_table(_valid_table(), first_path)
    pq.write_table(_valid_table(), second_path)

    validate_market_data_file(
        first_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    validate_market_data_file(
        second_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert set(manifest["files"]) == {
        "data/first.parquet",
        "data/second.parquet",
    }
