from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from train_and_eval.environment.contexts import (
    get_context_definition,
)
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
        Validation rows plus the complete earlier TRAIN history required
        to build both the market window and every rolling context feature.
        These earlier rows are context only and are never scored.

    training_start_index:
        Local TRAIN observation index of the first fully initialized
        observation. Its next step executes on the first scored TRAIN row.

    validation_start_index:
        Local observation index corresponding to the final TRAIN candle.
        Its next environment step executes on the first VALIDATION row.
    """

    train_data: pd.DataFrame
    validation_data_with_lookback: pd.DataFrame

    split_index: int
    train_rows: int
    validation_rows: int

    history_rows_required: int
    training_start_index: int
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
    context: str,
) -> ChronologicalMarketDataSplit:
    """
    Split rows without shuffling and preserve complete context history.

    TRAIN:
        market_data[0:split_index]

    VALIDATION scoring:
        market_data[split_index:]

    TRAIN begins only after the first fully initialized observation.
    VALIDATION receives enough earlier TRAIN rows to make its first
    observation fully initialized, while every VALIDATION row remains a
    scored next-open execution step.
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

    definition = get_context_definition(
        context
    )
    history_rows_required = (
        definition.required_history_rows(
            window
        )
    )

    total_rows = len(market_data)

    split_index = math.floor(
        total_rows * train_ratio
    )

    train_rows = split_index
    validation_rows = total_rows - split_index

    # One additional row is required after the first complete observation
    # because actions execute at the next candle open.
    minimum_train_rows = (
        history_rows_required + 1
    )

    if train_rows < minimum_train_rows:
        raise MarketDataSplitError(
            "TRAIN split is too short for a complete context and one "
            "next-open execution step. "
            f"Train rows: {train_rows}, "
            f"required history rows: {history_rows_required}, "
            f"minimum train rows: {minimum_train_rows}."
        )

    if validation_rows < 1:
        raise MarketDataSplitError(
            "VALIDATION split must contain at least one row."
        )

    lookback_start = (
        split_index
        - history_rows_required
    )

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

    training_start_index = (
        history_rows_required - 1
    )
    validation_start_index = (
        history_rows_required - 1
    )

    # From observation index history_rows_required - 1, the environment
    # executes rows history_rows_required through train_rows - 1.
    steps_per_data_epoch = (
        train_rows
        - history_rows_required
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
        history_rows_required=(
            history_rows_required
        ),
        training_start_index=(
            training_start_index
        ),
        validation_lookback_rows=(
            history_rows_required
        ),
        validation_start_index=(
            validation_start_index
        ),
        steps_per_data_epoch=steps_per_data_epoch,
        validation_steps=validation_steps,
    )
