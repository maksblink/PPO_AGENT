from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def _prepare_approved_file(
    tmp_path: Path,
    *,
    status: str = "okay",
) -> tuple[Path, Path, Path]:
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    data_path = data_directory / "market.parquet"
    pq.write_table(_valid_table(), data_path)

    sha256 = calculate_file_sha256(data_path)
    key = manifest_key(
        data_path,
        data_directory=data_directory,
    )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "files": {
            key: {
                "path": key,
                "status": status,
                "sha256": sha256,
                "rows": 3,
                "errors": [],
            }
        },
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return data_path, data_directory, manifest_path


def test_load_market_data_verifies_and_loads_file(
    tmp_path: Path,
) -> None:
    data_path, data_directory, manifest_path = (
        _prepare_approved_file(tmp_path)
    )

    frame = load_market_data(
        data_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    assert list(frame.columns) == [
        "DT",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "symbol",
        "instrument_id",
    ]
    assert len(frame) == 3
    assert frame["Close"].tolist() == [101.0, 102.0, 103.0]
    assert len(frame.attrs["sha256"]) == 64


def test_load_market_data_requires_manifest(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    data_path = data_directory / "market.parquet"
    pq.write_table(_valid_table(), data_path)

    with pytest.raises(
        MarketDataLoadingError,
        match="manifest does not exist",
    ):
        load_market_data(
            data_path,
            data_directory=data_directory,
            manifest_path=tmp_path / "missing.json",
        )


def test_load_market_data_rejects_not_approved_file(
    tmp_path: Path,
) -> None:
    data_path, data_directory, manifest_path = (
        _prepare_approved_file(tmp_path, status="not_okay")
    )

    with pytest.raises(
        MarketDataLoadingError,
        match="not approved",
    ):
        load_market_data(
            data_path,
            data_directory=data_directory,
            manifest_path=manifest_path,
        )


def test_load_market_data_rejects_changed_file(
    tmp_path: Path,
) -> None:
    data_path, data_directory, manifest_path = (
        _prepare_approved_file(tmp_path)
    )

    with data_path.open("ab") as file:
        file.write(b"changed")

    with pytest.raises(
        MarketDataLoadingError,
        match="has changed",
    ):
        load_market_data(
            data_path,
            data_directory=data_directory,
            manifest_path=manifest_path,
        )
