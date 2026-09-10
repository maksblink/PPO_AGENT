from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from train_and_eval.market_data.load_market_data import (
    DATA_DIRECTORY,
    MANIFEST_PATH,
    SOURCE_COLUMNS,
    calculate_file_sha256,
    load_manifest,
    manifest_key,
    resolve_data_file,
    verify_market_data,
    verify_loaded_frame,
    INTERVAL_MINUTES,
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
    Verify the pipeline manifest and perform local checks without writing.

    The result is returned to the caller only. Neither a successful check
    nor a failure can create, approve or modify a producer manifest.
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
        "errors": [],
    }

    try:
        record["sha256"] = calculate_file_sha256(file_path)

        _, entry = verify_market_data(
            file_path, manifest_path=manifest_path, data_directory=data_directory,
        )
        validate_schema(file_path)

        frame = pd.read_parquet(
            file_path,
            columns=list(EXPECTED_COLUMNS),
            engine="pyarrow",
        )

        validate_dataframe(frame)
        verify_loaded_frame(frame, entry)
        interval_ns = INTERVAL_MINUTES[entry["interval"]] * 60 * 1_000_000_000
        timestamps = frame["Datetime"]
        if (timestamps.astype("int64") % interval_ns != 0).any():
            raise MarketDataValidationError("Unaligned interval timestamps.")
        if (timestamps.dt.weekday >= 5).any():
            raise MarketDataValidationError("Weekend UTC records are not permitted.")
        if frame["symbol"].eq("").any():
            raise MarketDataValidationError("Empty symbol values are not permitted.")
        if calculate_file_sha256(file_path) != entry["sha256"]:
            raise MarketDataValidationError("Data file has changed while validating.")

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
        description="Verify pipeline datasets without modifying data or manifest."
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
        help="Verify every Parquet file listed in the imported manifest.",
    )
    args = parser.parse_args()

    if args.all and args.path:
        parser.error("Use either one path or --all, not both.")

    if not args.all and not args.path:
        parser.error("Provide a file path or use --all.")

    if args.all:
        try:
            manifest = load_manifest(MANIFEST_PATH)
        except Exception as error:
            print(f"[NOT OKAY] {error}")
            return 1
        files = [DATA_DIRECTORY / item["path"] for item in manifest["files"].values()]
    else:
        files = [args.path]

    records = []
    for path in files:
        try:
            records.append(validate_market_data_file(
                path, manifest_path=MANIFEST_PATH, data_directory=DATA_DIRECTORY,
            ))
        except Exception as error:
            records.append({
                "path": str(path), "status": "not_okay",
                "errors": [f"{type(error).__name__}: {error}"],
            })

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
