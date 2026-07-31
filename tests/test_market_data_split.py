from __future__ import annotations

import pandas as pd
import pytest

from train_and_eval.market_data.split_market_data import (
    MarketDataSplitError,
    split_market_data_chronologically,
)
from train_and_eval.run_config import TrainingSection


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


def test_split_is_strictly_chronological() -> None:
    split = split_market_data_chronologically(
        _frame(10),
        train_ratio=0.7,
        window=3,
    )

    assert split.split_index == 7
    assert split.train_rows == 7
    assert split.validation_rows == 3

    assert split.train_data["row_id"].tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    # Three lookback rows plus all three validation rows.
    assert (
        split.validation_data_with_lookback[
            "row_id"
        ].tolist()
        == [
            4,
            5,
            6,
            7,
            8,
            9,
        ]
    )


def test_first_validation_execution_uses_first_validation_row() -> None:
    split = split_market_data_chronologically(
        _frame(10),
        train_ratio=0.7,
        window=3,
    )

    local_observation_index = (
        split.validation_start_index
    )
    local_execution_index = (
        local_observation_index + 1
    )

    validation_frame = (
        split.validation_data_with_lookback
    )

    assert validation_frame.loc[
        local_observation_index,
        "row_id",
    ] == 6

    assert validation_frame.loc[
        local_execution_index,
        "row_id",
    ] == 7


def test_step_counts_match_next_open_environment() -> None:
    split = split_market_data_chronologically(
        _frame(100),
        train_ratio=0.8,
        window=10,
    )

    assert split.train_rows == 80
    assert split.validation_rows == 20

    assert split.steps_per_data_epoch == 70
    assert split.validation_steps == 20


def test_data_epochs_resolve_from_split() -> None:
    split = split_market_data_chronologically(
        _frame(100),
        train_ratio=0.8,
        window=10,
    )

    training = TrainingSection(
        duration_unit="data_epochs",
        duration_amount=3,
    )

    assert split.requested_training_steps(
        training
    ) == 210


def test_timesteps_are_used_directly() -> None:
    split = split_market_data_chronologically(
        _frame(100),
        train_ratio=0.8,
        window=10,
    )

    training = TrainingSection(
        duration_unit="timesteps",
        duration_amount=12_345,
    )

    assert split.requested_training_steps(
        training
    ) == 12_345


def test_split_rejects_train_shorter_than_window() -> None:
    with pytest.raises(
        MarketDataSplitError,
        match="TRAIN split is too short",
    ):
        split_market_data_chronologically(
            _frame(10),
            train_ratio=0.5,
            window=5,
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
            _frame(100),
            train_ratio=train_ratio,
            window=10,
        )
