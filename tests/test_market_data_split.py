from __future__ import annotations

import pandas as pd
import pytest

from train_and_eval.market_data.split_market_data import (
    MarketDataSplitError,
    split_market_data_chronologically,
)
from train_and_eval.run_config import TrainingSection


CONTEXT = "baseline_multiscale_v1"
HISTORY_ROWS = 6145


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(rows),
            "DT": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="5min",
                tz="UTC",
            ),
        }
    )


def _split():
    return split_market_data_chronologically(
        _frame(7000),
        train_ratio=0.9,
        window=40,
        context=CONTEXT,
    )


def test_split_is_strictly_chronological() -> None:
    split = _split()

    assert split.split_index == 6300
    assert split.train_rows == 6300
    assert split.validation_rows == 700
    assert split.history_rows_required == HISTORY_ROWS
    assert split.training_start_index == 6144

    assert split.train_data["row_id"].iloc[0] == 0
    assert split.train_data["row_id"].iloc[-1] == 6299

    # Complete history from TRAIN plus every validation row.
    validation_ids = (
        split.validation_data_with_lookback[
            "row_id"
        ]
    )

    assert validation_ids.iloc[0] == 155
    assert validation_ids.iloc[HISTORY_ROWS - 1] == 6299
    assert validation_ids.iloc[HISTORY_ROWS] == 6300
    assert validation_ids.iloc[-1] == 6999
    assert len(validation_ids) == HISTORY_ROWS + 700


def test_first_validation_execution_uses_first_validation_row() -> None:
    split = _split()

    local_observation_index = (
        split.validation_start_index
    )
    local_execution_index = (
        local_observation_index + 1
    )

    validation_frame = (
        split.validation_data_with_lookback
    )

    assert local_observation_index == 6144
    assert validation_frame.loc[
        local_observation_index,
        "row_id",
    ] == 6299

    assert validation_frame.loc[
        local_execution_index,
        "row_id",
    ] == 6300


def test_step_counts_match_complete_context_environment() -> None:
    split = _split()

    assert split.steps_per_data_epoch == 155
    assert split.validation_steps == 700


def test_data_epochs_resolve_from_split() -> None:
    split = _split()

    training = TrainingSection(
        duration_unit="data_epochs",
        duration_amount=3,
    )

    assert split.requested_training_steps(
        training
    ) == 465


def test_timesteps_are_used_directly() -> None:
    split = _split()

    training = TrainingSection(
        duration_unit="timesteps",
        duration_amount=12_345,
    )

    assert split.requested_training_steps(
        training
    ) == 12_345


def test_split_rejects_train_shorter_than_complete_context() -> None:
    with pytest.raises(
        MarketDataSplitError,
        match="TRAIN split is too short",
    ):
        split_market_data_chronologically(
            _frame(7000),
            train_ratio=0.8,
            window=40,
            context=CONTEXT,
        )


def test_daily_dataset_is_too_short_for_baseline_context() -> None:
    with pytest.raises(
        MarketDataSplitError,
        match="required history rows: 6145",
    ):
        split_market_data_chronologically(
            _frame(4159),
            train_ratio=0.8,
            window=40,
            context=CONTEXT,
        )


@pytest.mark.parametrize(
    "train_ratio",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_split_rejects_invalid_train_ratio(
    train_ratio: float,
) -> None:
    with pytest.raises(
        MarketDataSplitError,
        match="train_ratio",
    ):
        split_market_data_chronologically(
            _frame(7000),
            train_ratio=train_ratio,
            window=40,
            context=CONTEXT,
        )
