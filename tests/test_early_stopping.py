from __future__ import annotations

import math

import pytest

from train_and_eval.training.early_stopping import (
    BalancedScoreEarlyStopping,
    EarlyStoppingError,
)


def test_tracks_strict_improvements_and_patience() -> None:
    tracker = BalancedScoreEarlyStopping(
        patience=2
    )

    first = tracker.observe(
        checkpoint_id=1,
        evaluation_id=11,
        balanced_score=0.10,
    )
    second = tracker.observe(
        checkpoint_id=2,
        evaluation_id=12,
        balanced_score=0.05,
    )
    third = tracker.observe(
        checkpoint_id=3,
        evaluation_id=13,
        balanced_score=0.05,
    )

    assert first.improved is True
    assert first.no_improvement_evals == 0
    assert first.should_stop is False

    assert second.improved is False
    assert second.no_improvement_evals == 1
    assert second.should_stop is False

    assert third.improved is False
    assert third.no_improvement_evals == 2
    assert third.should_stop is True
    assert third.best_checkpoint_id == 1
    assert third.best_evaluation_id == 11
    assert third.best_score == pytest.approx(0.10)


def test_improvement_resets_no_improvement_counter() -> None:
    tracker = BalancedScoreEarlyStopping(
        patience=2
    )

    tracker.observe(
        checkpoint_id=1,
        evaluation_id=11,
        balanced_score=0.10,
    )
    tracker.observe(
        checkpoint_id=2,
        evaluation_id=12,
        balanced_score=0.05,
    )
    improved = tracker.observe(
        checkpoint_id=3,
        evaluation_id=13,
        balanced_score=0.20,
    )
    next_result = tracker.observe(
        checkpoint_id=4,
        evaluation_id=14,
        balanced_score=0.19,
    )

    assert improved.improved is True
    assert improved.no_improvement_evals == 0
    assert improved.best_checkpoint_id == 3
    assert next_result.no_improvement_evals == 1
    assert next_result.should_stop is False


def test_equal_score_is_not_an_improvement() -> None:
    tracker = BalancedScoreEarlyStopping(
        patience=1
    )

    tracker.observe(
        checkpoint_id=1,
        evaluation_id=11,
        balanced_score=0.10,
    )
    equal = tracker.observe(
        checkpoint_id=2,
        evaluation_id=12,
        balanced_score=0.10,
    )

    assert equal.improved is False
    assert equal.should_stop is True
    assert equal.best_checkpoint_id == 1


@pytest.mark.parametrize(
    "value",
    [None, math.nan, math.inf, -math.inf, True],
)
def test_rejects_invalid_balanced_score(value) -> None:
    tracker = BalancedScoreEarlyStopping(
        patience=2
    )

    with pytest.raises(
        EarlyStoppingError,
        match="finite number",
    ):
        tracker.observe(
            checkpoint_id=1,
            evaluation_id=11,
            balanced_score=value,
        )


def test_rejects_invalid_patience() -> None:
    with pytest.raises(
        EarlyStoppingError,
        match="positive integer",
    ):
        BalancedScoreEarlyStopping(
            patience=0
        )
