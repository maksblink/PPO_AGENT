from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from train_and_eval.market_data.load_market_data import (
    DATA_DIRECTORY,
    MANIFEST_PATH,
    MANIFEST_VERSION,
    SOURCE_COLUMNS,
    calculate_file_sha256,
    load_manifest,
    manifest_key,
    resolve_data_file,
)


EXPECTED_COLUMNS = SOURCE_COLUMNS

EXPECTED_TYPES = {
    "Datetime": pa.timestamp("ns", tz="UTC"),
    "Open_NQ": pa.float64(),
    "High_NQ": pa.float64(),
    "Low_NQ": pa.float64(),
    "Close_NQ": pa.float64(),
    "Volume_NQ": pa.uint64(),
    "symbol": pa.large_string(),
    "instrument_id": pa.uint32(),
}


class MarketDataValidationError(ValueError):
    """Raised when a market-data file fails full validation."""


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def new_manifest() -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "files": {},
    }


def load_or_create_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()

    if not path.exists():
        return new_manifest()

    return load_manifest(path)


def write_manifest(
    manifest: dict[str, Any],
    manifest_path: str | Path,
) -> None:
    """
    Rewrite the complete manifest atomically.

    A temporary file is written first, then replaces the previous manifest.
    """
    path = Path(manifest_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )

    temporary_path.write_text(
        serialized + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def reset_manifest(
    manifest_path: str | Path = MANIFEST_PATH,
) -> None:
    write_manifest(new_manifest(), manifest_path)


def validate_schema(path: Path) -> None:
    schema = pq.read_schema(path)

    actual_columns = tuple(schema.names)

    if actual_columns != EXPECTED_COLUMNS:
        raise MarketDataValidationError(
            "Unexpected Parquet columns.\n"
            f"Expected: {list(EXPECTED_COLUMNS)}\n"
            f"Received: {list(actual_columns)}"
        )

    mismatches: list[str] = []

    for column, expected_type in EXPECTED_TYPES.items():
        actual_type = schema.field(column).type

        if actual_type != expected_type:
            mismatches.append(
                f"{column}: expected {expected_type}, "
                f"received {actual_type}"
            )

    if mismatches:
        formatted = "\n".join(
            f"- {mismatch}" for mismatch in mismatches
        )
        raise MarketDataValidationError(
            f"Unexpected Parquet column types:\n{formatted}"
        )


def validate_dataframe(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise MarketDataValidationError(
            "Market-data file contains no rows."
        )

    columns_with_nulls = [
        column
        for column in EXPECTED_COLUMNS
        if frame[column].isna().any()
    ]

    if columns_with_nulls:
        raise MarketDataValidationError(
            f"Columns contain missing values: {columns_with_nulls}"
        )

    timestamps = frame["Datetime"]

    duplicate_count = int(timestamps.duplicated().sum())

    if duplicate_count:
        raise MarketDataValidationError(
            f"Datetime contains {duplicate_count} duplicate timestamps."
        )

    if not timestamps.is_monotonic_increasing:
        raise MarketDataValidationError(
            "Datetime must be sorted in strictly increasing order."
        )

    price_columns = (
        "Open_NQ",
        "High_NQ",
        "Low_NQ",
        "Close_NQ",
    )

    for column in price_columns:
        values = frame[column].to_numpy(copy=False)

        if not np.isfinite(values).all():
            raise MarketDataValidationError(
                f"{column} contains non-finite values."
            )

        if (values <= 0).any():
            raise MarketDataValidationError(
                f"{column} contains prices that are not greater than zero."
            )

    invalid_ohlc = (
        (frame["High_NQ"] < frame["Open_NQ"])
        | (frame["High_NQ"] < frame["Close_NQ"])
        | (frame["High_NQ"] < frame["Low_NQ"])
        | (frame["Low_NQ"] > frame["Open_NQ"])
        | (frame["Low_NQ"] > frame["Close_NQ"])
    )

    if invalid_ohlc.any():
        first_invalid_row = int(
            np.flatnonzero(invalid_ohlc.to_numpy())[0]
        )
        raise MarketDataValidationError(
            "Invalid OHLC relationship at row "
            f"{first_invalid_row}. High must be the highest value "
            "and Low must be the lowest value in the candle."
        )


def validate_market_data_file(
    path: str | Path,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    data_directory: str | Path = DATA_DIRECTORY,
) -> dict[str, Any]:
    """
    Perform complete validation and update the file's manifest entry.

    Validation failures are recorded as status='not_okay'. The manifest is
    rewritten even when validation fails.
    """
    file_path = resolve_data_file(
        path,
        data_directory=data_directory,
    )
    key = manifest_key(
        file_path,
        data_directory=data_directory,
    )

    record: dict[str, Any] = {
        "path": key,
        "status": "not_okay",
        "sha256": None,
        "size_bytes": file_path.stat().st_size,
        "rows": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "validated_at_utc": utc_now_iso(),
        "errors": [],
    }

    try:
        record["sha256"] = calculate_file_sha256(file_path)

        validate_schema(file_path)

        frame = pd.read_parquet(
            file_path,
            columns=list(EXPECTED_COLUMNS),
            engine="pyarrow",
        )

        validate_dataframe(frame)

        record.update(
            {
                "status": "okay",
                "rows": int(len(frame)),
                "first_timestamp": frame["Datetime"].iloc[0].isoformat(),
                "last_timestamp": frame["Datetime"].iloc[-1].isoformat(),
                "errors": [],
            }
        )

    except Exception as error:
        record["status"] = "not_okay"
        record["errors"] = [
            f"{type(error).__name__}: {error}"
        ]

    manifest = load_or_create_manifest(manifest_path)
    manifest["files"][key] = record
    write_manifest(manifest, manifest_path)

    return record


def print_result(record: dict[str, Any]) -> None:
    if record["status"] == "okay":
        print(f"[OK] {record['path']}")
        print(f"     rows: {record['rows']:,}")
        print(f"     SHA-256: {record['sha256']}")
        return

    print(f"[NOT OKAY] {record['path']}")

    for error in record["errors"]:
        print(f"     {error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perform full validation of market-data Parquet files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "One Parquet file from the top-level data directory. "
            "May be omitted when using --all."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every .parquet file from the top-level data directory.",
    )
    args = parser.parse_args()

    if args.all and args.path:
        parser.error("Use either one path or --all, not both.")

    if not args.all and not args.path:
        parser.error("Provide a file path or use --all.")

    if args.all:
        files = sorted(DATA_DIRECTORY.glob("*.parquet"))

        if not files:
            print(f"No Parquet files found in: {DATA_DIRECTORY}")
            return 1

        # A full-directory validation rebuilds the manifest from scratch.
        reset_manifest(MANIFEST_PATH)

        records = [
            validate_market_data_file(path)
            for path in files
        ]
    else:
        records = [
            validate_market_data_file(args.path)
        ]

    print()

    for record in records:
        print_result(record)

    okay_count = sum(
        record["status"] == "okay"
        for record in records
    )
    not_okay_count = len(records) - okay_count

    print()
    print(f"Okay: {okay_count}")
    print(f"Not okay: {not_okay_count}")
    print(f"Manifest: {MANIFEST_PATH}")

    return 0 if not_okay_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
