"""Preview calendar ranges and update a stage-one YAML only after confirmation."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import os
from pathlib import Path
import tempfile

import pandas as pd
import yaml

from train_and_eval.market_data.load_market_data import load_manifest, load_market_data
from train_and_eval.run_config import RunConfig
from train_and_eval.walk_forward.windows import split_time_ranges


def ceil_week(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Calendar bounds must be timezone-aware")
    value = value.tz_convert("UTC")
    monday = value.normalize() - pd.Timedelta(days=value.weekday())
    return monday if value == monday else monday + pd.Timedelta(weeks=1)


def calculate_ranges(first, end, train_fraction=0.5, train_split=0.9):
    fraction, split = Decimal(str(train_fraction)), Decimal(str(train_split))
    if not fraction.is_finite() or not split.is_finite() or not (0 < fraction < 1 and 0 < split < 1):
        raise ValueError("Fractions must be finite and strictly between zero and one")
    first, end = pd.Timestamp(first), pd.Timestamp(end)
    if first.tzinfo is None or end.tzinfo is None or first >= end:
        raise ValueError("Expected increasing timezone-aware dataset coverage")
    start = ceil_week(first)
    offset = int(Decimal((end - first).value) * fraction)
    train_end = ceil_week(first + pd.Timedelta(offset, unit="ns"))
    train_weeks = int((train_end - start) / pd.Timedelta(weeks=1))
    validation_weeks = int((Decimal(train_weeks) * (1 - split) / split).to_integral_value(rounding=ROUND_CEILING))
    validation_end = train_end + pd.Timedelta(weeks=validation_weeks)
    if train_weeks < 1 or validation_weeks < 1 or validation_end > end:
        raise ValueError("Rounded full-week training and validation do not fit the dataset")
    return {
        "train_range": {"start": start.isoformat(), "end": train_end.isoformat()},
        "validation_range": {"start": train_end.isoformat(), "end": validation_end.isoformat()},
    }


def replace_ranges(original: str, ranges: dict) -> str:
    """Replace only the two explicit block fields; preserve other YAML text."""
    document = yaml.compose(original)
    if not isinstance(document, yaml.MappingNode):
        raise ValueError("Expected a YAML mapping")
    data_nodes = [value for key, value in document.value if key.value == "data"]
    if len(data_nodes) != 1 or not isinstance(data_nodes[0], yaml.MappingNode) or data_nodes[0].flow_style:
        raise ValueError("Expected one block-style data mapping")
    data = data_nodes[0]
    keys = [key.value for key, _ in data.value]
    if len(keys) != len(set(keys)) or "train_ratio" in keys:
        raise ValueError("Use explicit train_range and validation_range fields, without train_ratio or duplicate keys")
    lines = original.splitlines(keepends=True)
    replacements = []
    for name, bounds in ranges.items():
        matches = [(key, value) for key, value in data.value if key.value == name]
        if len(matches) != 1:
            raise ValueError(f"Config must contain {name}")
        key, value = matches[0]
        if not isinstance(value, yaml.MappingNode) or value.flow_style:
            raise ValueError(f"Use a block mapping for {name}")
        # Only replace lines occupied by the key and its values, preserving following comments.
        last_line = max(v.end_mark.line for _, v in value.value)
        indent = " " * key.start_mark.column
        block = indent + name + ":\n" + "".join(
            indent + "  " + field + ": '" + bounds[field] + "'\n" for field in ("start", "end")
        )
        replacements.append((key.start_mark.line, last_line + 1, block))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines)
    expected = yaml.safe_load(original)
    expected["data"].update(ranges)
    if yaml.safe_load(result) != expected:
        raise ValueError("Unsupported YAML layout; refusing to modify unrelated values")
    return result


def confirm_and_write(path, original, updated, *, prompt=input):
    try:
        answer = prompt("Write these ranges to the config? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ("y", "yes", "t", "tak"):
        return False
    if path.read_text() != original:
        raise RuntimeError("Config changed after preview; rerun before writing")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-fraction", default="0.5")
    parser.add_argument("--train-split", default="0.9")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    root, path = args.project_root.resolve(), args.config.resolve()
    original = path.read_text()
    config = RunConfig.model_validate(yaml.safe_load(original))
    if config.data.train_range is None or config.data.validation_range is None or config.data.alignment != "trim_start":
        raise ValueError("Stage one requires explicit train/validation ranges and trim_start")
    manifest_path = root / "train_and_eval/market_data/manifest.json"
    manifest = load_manifest(manifest_path)
    frame = load_market_data(config.data.path, data_directory=root / "data", manifest_path=manifest_path)
    identity = manifest["files"][frame.attrs["interval"]]
    ranges = calculate_ranges(identity["first_timestamp"], manifest["build"]["target_end_exclusive_utc"],
                              args.train_fraction, args.train_split)
    updated = replace_ranges(original, ranges)
    proposed = RunConfig.model_validate(yaml.safe_load(updated))
    metadata = split_time_ranges(frame, proposed).window_metadata
    print(f"Config: {path}\nRelease: {manifest['release_id']}\nAll ranges are [start, end), UTC.")
    for name in ranges:
        print(f"{name}: {getattr(config.data, name).model_dump()} -> {ranges[name]}")
    train_weeks = (pd.Timestamp(ranges["train_range"]["end"]) - pd.Timestamp(ranges["train_range"]["start"])) / pd.Timedelta(weeks=1)
    val_weeks = (pd.Timestamp(ranges["validation_range"]["end"]) - pd.Timestamp(ranges["validation_range"]["start"])) / pd.Timedelta(weeks=1)
    print(f"Train: {train_weeks:g} weeks; validation: {val_weeks:g} weeks; actual calendar split: {train_weeks/(train_weeks+val_weeks):.8f}")
    print(f"Context: {metadata['history_rows']}; warm-up: {metadata['warmup_rows']}; trimmed: {metadata['trimmed_steps']}; train steps: {metadata['train']['rows']}; validation steps: {metadata['validation']['rows']}")
    print("Saved." if confirm_and_write(path, original, updated) else "No changes written.")


if __name__ == "__main__":
    main()
