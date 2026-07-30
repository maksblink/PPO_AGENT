from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
MANIFEST_PATH = Path(__file__).with_name("manifest.json")
MANIFEST_VERSION = 1

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
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest(
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()

    if not path.is_file():
        raise MarketDataLoadingError(
            "Data manifest does not exist. Run full validation first:\n"
            "python -m train_and_eval.market_data.validate_market_data --all"
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MarketDataLoadingError(
            f"Data manifest is not valid JSON: {path}"
        ) from error

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise MarketDataLoadingError(
            "Unsupported data manifest version."
        )

    if not isinstance(manifest.get("files"), dict):
        raise MarketDataLoadingError(
            "Data manifest does not contain a valid 'files' mapping."
        )

    return manifest


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
    entry = manifest["files"].get(key)

    if entry is None:
        raise MarketDataLoadingError(
            f"File has no validation entry in the manifest: {key}"
        )

    if entry.get("status") != "okay":
        raise MarketDataLoadingError(
            f"File is not approved for training: {key}"
        )

    expected_sha256 = entry.get("sha256")

    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise MarketDataLoadingError(
            f"Manifest contains an invalid SHA-256 for: {key}"
        )

    current_sha256 = calculate_file_sha256(file_path)

    if not hmac.compare_digest(current_sha256, expected_sha256):
        raise MarketDataLoadingError(
            "Data file has changed since its last full validation.\n"
            f"File: {key}\n"
            f"Expected SHA-256: {expected_sha256}\n"
            f"Current SHA-256:  {current_sha256}"
        )

    return file_path, entry


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

    frame = pd.read_parquet(
        file_path,
        columns=list(SOURCE_COLUMNS),
        engine="pyarrow",
    )

    expected_rows = manifest_entry.get("rows")

    if expected_rows is not None and len(frame) != expected_rows:
        raise MarketDataLoadingError(
            "Loaded row count differs from the validated manifest.\n"
            f"Expected: {expected_rows}\n"
            f"Loaded:   {len(frame)}"
        )

    frame = frame.rename(columns=CANONICAL_COLUMN_NAMES)
    frame = frame.reset_index(drop=True)

    # These values will later be available to the run registry.
    frame.attrs["source_path"] = manifest_entry["path"]
    frame.attrs["sha256"] = manifest_entry["sha256"]

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
