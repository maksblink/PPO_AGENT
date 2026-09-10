from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
MANIFEST_PATH = Path(__file__).with_name("manifest.json")
MANIFEST_VERSION = 2
DATA_SCHEMA_VERSION = 1
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}

SOURCE_COLUMNS = (
    "Datetime",
    "Open_NQ",
    "High_NQ",
    "Low_NQ",
    "Close_NQ",
    "Volume_NQ",
    "symbol",
    "instrument_id",
)

CANONICAL_COLUMN_NAMES = {
    "Datetime": "DT",
    "Open_NQ": "Open",
    "High_NQ": "High",
    "Low_NQ": "Low",
    "Close_NQ": "Close",
    "Volume_NQ": "Volume",
}

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


class MarketDataLoadingError(RuntimeError):
    """Raised when market data cannot safely be loaded for training."""


def resolve_data_file(
    path: str | Path,
    *,
    data_directory: str | Path = DATA_DIRECTORY,
) -> Path:
    """Resolve a Parquet file located inside the top-level data directory."""
    data_root = Path(data_directory).expanduser().resolve()
    supplied_path = Path(path).expanduser()

    if supplied_path.is_absolute():
        file_path = supplied_path.resolve()
    elif supplied_path.parts and supplied_path.parts[0] == "data":
        file_path = (data_root.parent / supplied_path).resolve()
    else:
        file_path = (data_root / supplied_path).resolve()

    if not file_path.is_relative_to(data_root):
        raise MarketDataLoadingError(
            f"Data file must be located inside: {data_root}"
        )

    if not file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    if not file_path.is_file():
        raise MarketDataLoadingError(
            f"Data path is not a file: {file_path}"
        )

    if file_path.suffix.lower() != ".parquet":
        raise MarketDataLoadingError(
            f"Expected a .parquet file: {file_path}"
        )

    return file_path


def manifest_key(
    file_path: str | Path,
    *,
    data_directory: str | Path = DATA_DIRECTORY,
) -> str:
    data_root = Path(data_directory).expanduser().resolve()
    resolved_path = Path(file_path).expanduser().resolve()
    relative_path = resolved_path.relative_to(data_root)

    return f"data/{relative_path.as_posix()}"


def calculate_file_sha256(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 without loading the entire file into memory."""
    file_path = Path(path).expanduser().resolve()
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MarketDataLoadingError(message)


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    _require(isinstance(value, str), f"Invalid {label} timestamp.")
    try:
        stamp = pd.Timestamp(value)
        valid = (
            not pd.isna(stamp)
            and stamp.tzinfo is not None
            and stamp.utcoffset().total_seconds() == 0
        )
        _require(valid, f"{label} must be a UTC timestamp.")
        return stamp.as_unit("ns")
    except (ValueError, TypeError, OverflowError) as error:
        raise MarketDataLoadingError(f"Invalid {label} timestamp.") from error


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    _require(isinstance(manifest, dict), "Manifest must be a JSON object.")
    _require(
        type(manifest.get("manifest_version")) is int
        and manifest["manifest_version"] == MANIFEST_VERSION,
        "Unsupported data manifest version; import a pipeline manifest v2.",
    )
    _require(
        type(manifest.get("data_schema_version")) is int
        and manifest["data_schema_version"] == DATA_SCHEMA_VERSION,
        "Unsupported data schema version.",
    )
    for name in ("dataset_id", "release_id"):
        _require(isinstance(manifest.get(name), str) and bool(manifest[name].strip()),
                 f"Manifest requires {name}.")
    validation = manifest.get("validation")
    _require(isinstance(validation, dict), "Manifest requires validation metadata.")
    _require(
        _integer(validation.get("structural_error_count"))
        and validation["structural_error_count"] == 0,
        "Dataset contains structural errors or lacks their count.",
    )
    count = validation.get("warning_event_count")
    counts = validation.get("warning_counts")
    _require(_integer(count), "Invalid warning event count.")
    _require(isinstance(counts, dict) and all(_integer(v) for v in counts.values()),
             "Invalid warning counts.")
    _require(sum(counts.values()) == count, "Inconsistent warning counts.")
    acceptance = validation.get("warning_acceptance")
    _require(isinstance(acceptance, dict), "Missing warning acceptance metadata.")
    _require({"required", "accepted", "mode", "accepted_at_utc"}.issubset(acceptance),
             "Incomplete warning acceptance metadata.")
    if count:
        _require(manifest.get("status") == "published_with_warnings"
                 and validation.get("status") == "passed_with_warnings",
                 "Dataset is not approved: inconsistent publication status.")
        _require(acceptance.get("required") is True
                 and acceptance.get("accepted") is True
                 and acceptance.get("mode") in ("interactive", "cli_flag"),
                 "Dataset warnings are not approved.")
        _timestamp(acceptance.get("accepted_at_utc"), "warning acceptance")
    else:
        _require(manifest.get("status") == "published"
                 and validation.get("status") == "passed",
                 "Dataset is not approved: inconsistent publication status.")
        _require(acceptance.get("required") is False
                 and acceptance.get("mode") == "not_required"
                 and acceptance.get("accepted") is None
                 and acceptance.get("accepted_at_utc") is None,
                 "Inconsistent warning acceptance requirement.")
    files = manifest.get("files")
    _require(isinstance(files, dict) and bool(files), "Manifest requires a nonempty files mapping.")
    paths: set[str] = set()
    for interval, entry in files.items():
        _require(interval in INTERVAL_MINUTES and isinstance(entry, dict),
                 "Invalid interval or file entry.")
        _require(entry.get("interval") == interval, "File interval does not match its key.")
        _require(type(entry.get("data_schema_version")) is int
                 and entry["data_schema_version"] == DATA_SCHEMA_VERSION,
                 "Unsupported file data schema version.")
        _require(entry.get("validation_status") == "passed", "File is not approved for training.")
        path = entry.get("path")
        _require(isinstance(path, str) and bool(path), "Invalid manifest file path.")
        relative = PurePosixPath(path)
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and "\\" not in path and ":" not in path and "\x00" not in path
                 and relative.as_posix() == path and relative.suffix == ".parquet",
                 "Manifest file path must be a normalized relative Parquet path.")
        _require(path not in paths, "Duplicate manifest file path.")
        paths.add(path)
        _require(_valid_sha256(entry.get("sha256")), "Invalid file SHA-256.")
        _require(_integer(entry.get("rows"), minimum=1), "Invalid file row count.")
        _require(_integer(entry.get("size_bytes"), minimum=1), "Invalid file size.")
        first = _timestamp(entry.get("first_timestamp"), "first")
        last = _timestamp(entry.get("last_timestamp"), "last")
        duration_ns = INTERVAL_MINUTES[interval] * 60 * 1_000_000_000
        _require(first <= last and first.value % duration_ns == 0
                 and last.value % duration_ns == 0,
                 "Invalid or unaligned file timestamp range.")
    return manifest


def load_manifest(
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()

    if not path.is_file():
        raise MarketDataLoadingError(
            "Data manifest does not exist. Import the published manifest v2 "
            "from NQ_HISTORICAL_DATA_PIPELINE together with its Parquet files."
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise MarketDataLoadingError(
            f"Data manifest is not valid JSON: {path}"
        ) from error

    return _validate_manifest(manifest)


def verify_market_data(
    path: str | Path,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    data_directory: str | Path = DATA_DIRECTORY,
) -> tuple[Path, dict[str, Any]]:
    """
    Verify that the file passed full validation and has not changed.

    No row-level or OHLC validation is repeated here.
    """
    file_path = resolve_data_file(
        path,
        data_directory=data_directory,
    )
    key = manifest_key(
        file_path,
        data_directory=data_directory,
    )
    manifest = load_manifest(manifest_path)
    relative = file_path.relative_to(Path(data_directory).expanduser().resolve()).as_posix()
    entries = [item for item in manifest["files"].values() if item["path"] == relative]
    if len(entries) != 1:
        raise MarketDataLoadingError(
            f"File has no validation entry in the manifest: {key}"
        )

    entry = entries[0]
    expected_sha256 = entry["sha256"]
    _require(file_path.stat().st_size == entry["size_bytes"],
             f"Data file has changed: size differs from manifest: {key}")

    current_sha256 = calculate_file_sha256(file_path)

    if not hmac.compare_digest(current_sha256, expected_sha256):
        raise MarketDataLoadingError(
            "Data file has changed since its last full validation.\n"
            f"File: {key}\n"
            f"Expected SHA-256: {expected_sha256}\n"
            f"Current SHA-256:  {current_sha256}"
        )

    # Keep the producer's manifest immutable. Adapt only this in-memory copy
    # to the existing PPO_AGENT data/<path> convention used by run metadata.
    result = dict(entry)
    result.update(path=key, manifest_file_path=entry["path"],
                  dataset_id=manifest["dataset_id"], release_id=manifest["release_id"])
    return file_path, result


def validate_parquet_schema(path: str | Path) -> None:
    schema = pq.read_schema(path)
    _require(tuple(schema.names) == SOURCE_COLUMNS, "Unexpected Parquet columns or order.")
    for name, expected in EXPECTED_TYPES.items():
        _require(schema.field(name).type == expected,
                 f"Unexpected Parquet type for {name}: expected {expected}.")


def verify_loaded_frame(frame: pd.DataFrame, entry: dict[str, Any]) -> None:
    _require(len(frame) == entry["rows"], "Loaded row count differs from manifest.")
    _require(not frame.empty, "Market-data file is empty.")
    _require(frame["Datetime"].iloc[0] == _timestamp(entry["first_timestamp"], "first")
             and frame["Datetime"].iloc[-1] == _timestamp(entry["last_timestamp"], "last"),
             "Loaded timestamp range differs from manifest.")


def load_market_data(
    path: str | Path,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    data_directory: str | Path = DATA_DIRECTORY,
) -> pd.DataFrame:
    """
    Verify the data hash and load approved market data for training.

    Returned columns:
        DT, Open, High, Low, Close, Volume, symbol, instrument_id
    """
    file_path, manifest_entry = verify_market_data(
        path,
        manifest_path=manifest_path,
        data_directory=data_directory,
    )

    validate_parquet_schema(file_path)
    frame = pd.read_parquet(
        file_path,
        columns=list(SOURCE_COLUMNS),
        engine="pyarrow",
    )

    verify_loaded_frame(frame, manifest_entry)
    _require(hmac.compare_digest(calculate_file_sha256(file_path), manifest_entry["sha256"]),
             "Data file has changed while loading.")

    frame = frame.rename(columns=CANONICAL_COLUMN_NAMES)
    frame = frame.reset_index(drop=True)

    # These values will later be available to the run registry.
    frame.attrs["source_path"] = manifest_entry["path"]
    frame.attrs["sha256"] = manifest_entry["sha256"]
    frame.attrs["release_id"] = manifest_entry["release_id"]
    frame.attrs["dataset_id"] = manifest_entry["dataset_id"]
    frame.attrs["interval"] = manifest_entry["interval"]

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and load approved market data."
    )
    parser.add_argument(
        "path",
        help="Parquet file from the project's top-level data directory.",
    )
    args = parser.parse_args()

    frame = load_market_data(args.path)

    print("Market data loading: OK")
    print(f"File: {frame.attrs['source_path']}")
    print(f"SHA-256: {frame.attrs['sha256']}")
    print(f"Rows: {len(frame):,}")
    print(f"First timestamp: {frame['DT'].iloc[0]}")
    print(f"Last timestamp: {frame['DT'].iloc[-1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
