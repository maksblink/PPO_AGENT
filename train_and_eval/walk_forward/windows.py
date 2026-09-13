from __future__ import annotations

import pandas as pd

from train_and_eval.environment.contexts import get_context_definition
from train_and_eval.market_data.split_market_data import (
    ChronologicalMarketDataSplit, MarketDataSplitError,
)
from train_and_eval.run_config import RunConfig


def range_indices(frame: pd.DataFrame, start: str, end: str) -> tuple[int, int]:
    """Map [start,end) to available rows, without inserting missing candles."""
    times = frame["DT"]
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a.tzinfo is None or b.tzinfo is None or a >= b:
        raise MarketDataSplitError("Expected increasing aware time bounds")
    left, right = int(times.searchsorted(a)), int(times.searchsorted(b))
    if left == right:
        raise MarketDataSplitError(f"No available candles in [{a}, {b})")
    return left, right


def range_record(frame: pd.DataFrame, start: str, end: str) -> dict:
    left, right = range_indices(frame, start, end)
    return {"start": start, "end": end, "start_index": left, "end_index": right,
            "rows": right - left, "first_timestamp": frame.DT.iloc[left].isoformat(),
            "last_timestamp": frame.DT.iloc[right - 1].isoformat()}


def split_time_ranges(frame: pd.DataFrame, config: RunConfig, *, validate_order: bool = True) -> ChronologicalMarketDataSplit:
    """Initial A: reserve context then trim. Updates/refit: prepend scored rows."""
    if frame.empty or (validate_order and (not frame.DT.is_monotonic_increasing or frame.DT.duplicated().any())):
        raise MarketDataSplitError("Time splitting requires nonempty, sorted, unique candles")
    bounds = config.data.train_range
    if bounds is None:
        raise MarketDataSplitError("train_range is required")
    nominal = range_record(frame, bounds.start, bounds.end)
    start, end = nominal["start_index"], nominal["end_index"]
    history = get_context_definition(config.environment.context).required_history_rows(config.environment.window)
    batch = config.ppo.batch_size
    trimmed = prepended = warmup = 0
    if config.data.alignment == "trim_start":
        available_start = max(start, history)
        warmup = available_start - start
        usable = end - available_start
        if usable < batch:
            raise MarketDataSplitError("Initial training is too short after reserving full context")
        trimmed = usable % batch
        actual_start = available_start + trimmed
    else:
        prepended = (-(end - start)) % batch
        actual_start = start - prepended
    if actual_start < history or actual_start >= end:
        raise MarketDataSplitError("Aligned training lacks a complete earlier observation context")
    lookback_start = actual_start - history
    train = frame.iloc[lookback_start:end].reset_index(drop=True)
    train.attrs = dict(frame.attrs)
    validation = None
    validation_frame = frame.iloc[:0].copy()
    validation_rows = 0
    if config.data.validation_range is not None:
        v = config.data.validation_range
        validation = range_record(frame, v.start, v.end)
        vs, ve = validation["start_index"], validation["end_index"]
        if vs < history or vs < end:
            raise MarketDataSplitError("Validation overlaps training or lacks earlier context")
        validation_frame = frame.iloc[vs-history:ve].reset_index(drop=True)
        validation_frame.attrs = dict(frame.attrs)
        validation_rows = ve - vs
    metadata = {
        "schema_version": 1, "data_rows": len(frame),
        "nominal_train": nominal,
        "train": {"start_index": actual_start, "end_index": end, "rows": end-actual_start,
                  "first_timestamp": frame.DT.iloc[actual_start].isoformat(),
                  "last_timestamp": frame.DT.iloc[end-1].isoformat()},
        "history_rows": history, "lookback_start_index": lookback_start,
        "warmup_rows": warmup, "trimmed_steps": trimmed, "prepended_steps": prepended,
        "alignment": config.data.alignment, "validation": validation,
    }
    return ChronologicalMarketDataSplit(
        train_data=train, validation_data_with_lookback=validation_frame,
        split_index=len(train), train_rows=len(train), validation_rows=validation_rows,
        history_rows_required=history, training_start_index=history-1,
        validation_lookback_rows=history if validation else 0,
        validation_start_index=history-1 if validation else 0,
        steps_per_data_epoch=end-actual_start, validation_steps=validation_rows,
        window_metadata=metadata,
    )
