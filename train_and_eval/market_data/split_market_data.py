from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from train_and_eval.run_config import TrainingSection


class MarketDataSplitError(ValueError):
    """Raised when a chronological split cannot be constructed."""


@dataclass(frozen=True)
class ChronologicalMarketDataSplit:
    """
    Chronological TRAIN/VALIDATION partition.

    train_data:
        Only rows belonging to TRAIN.

    validation_data_with_lookback:
        Validation rows plus exactly `window` preceding TRAIN rows.
        This permits the first validation decision to be made after the
        final TRAIN candle and executed at the first VALIDATION open.

    validation_start_index:
        Local observation index corresponding to the final TRAIN candle.
        Its next environment step executes on the first VALIDATION row.
    """

    train_data: pd.DataFrame
    validation_data_with_lookback: pd.DataFrame

    split_index: int
    train_rows: int
    validation_rows: int

    validation_lookback_rows: int
    validation_start_index: int

    steps_per_data_epoch: int
    validation_steps: int

    def requested_training_steps(
        self,
        training: TrainingSection,
    ) -> int:
        return training.resolve_training_steps(
            steps_per_data_epoch=self.steps_per_data_epoch,
        )


def _slice_with_attrs(
    frame: pd.DataFrame,
    start: int,
    stop: int | None,
) -> pd.DataFrame:
    result = (
        frame
        .iloc[start:stop]
        .reset_index(drop=True)
    )

    result.attrs = dict(frame.attrs)
    return result


def split_market_data_chronologically(
    market_data: pd.DataFrame,
    *,
    train_ratio: float,
    window: int,
) -> ChronologicalMarketDataSplit:
    """
    Split rows without shuffling.

    TRAIN:
        market_data[0:split_index]

    VALIDATION scoring:
        market_data[split_index:]

    Validation receives earlier TRAIN rows only as historical lookback.
    They are not counted as validation steps or validation results.
    """
    if market_data.empty:
        raise MarketDataSplitError(
            "Market data cannot be empty."
        )

    if not 0.0 < train_ratio < 1.0:
        raise MarketDataSplitError(
            "train_ratio must be between 0 and 1."
        )

    if window <= 0:
        raise MarketDataSplitError(
            "window must be greater than zero."
        )

    total_rows = len(market_data)

    split_index = math.floor(
        total_rows * train_ratio
    )

    train_rows = split_index
    validation_rows = total_rows - split_index

    # The training environment starts at index window - 1 and needs one
    # later candle for next-open execution.
    if train_rows < window + 1:
        raise MarketDataSplitError(
            "TRAIN split is too short. It must contain at least "
            f"window + 1 rows. Train rows: {train_rows}, "
            f"window: {window}."
        )

    if validation_rows < 1:
        raise MarketDataSplitError(
            "VALIDATION split must contain at least one row."
        )

    lookback_start = split_index - window

    train_data = _slice_with_attrs(
        market_data,
        0,
        split_index,
    )

    validation_data_with_lookback = _slice_with_attrs(
        market_data,
        lookback_start,
        None,
    )

    validation_start_index = window - 1

    steps_per_data_epoch = (
        train_rows - window
    )

    validation_steps = validation_rows

    return ChronologicalMarketDataSplit(
        train_data=train_data,
        validation_data_with_lookback=(
            validation_data_with_lookback
        ),
        split_index=split_index,
        train_rows=train_rows,
        validation_rows=validation_rows,
        validation_lookback_rows=window,
        validation_start_index=validation_start_index,
        steps_per_data_epoch=steps_per_data_epoch,
        validation_steps=validation_steps,
    )
