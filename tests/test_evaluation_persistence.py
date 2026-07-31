from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from train_and_eval.database.models import (
    Checkpoint,
    Evaluation,
    EvaluationCurrentStreakType,
    EvaluationStatus,
)
from train_and_eval.evaluation.metrics import (
    calculate_evaluation_metrics,
)
from train_and_eval.evaluation.persistence import (
    EvaluationCheckpointNotFoundError,
    EvaluationIdentityError,
    EvaluationResultMismatchError,
    EvaluationStateTransitionError,
    complete_evaluation,
    create_pending_evaluation,
    fail_evaluation,
    mark_evaluation_running,
)
from train_and_eval.evaluation.runner import (
    EvaluationPolicyTrace,
    EvaluationRunResult,
)


class FakeSession:
    def __init__(self) -> None:
        self.checkpoint: object | None = (
            SimpleNamespace(id=9)
        )
        self.evaluation: Evaluation | None = None

        self.commit_calls = 0
        self.rollback_calls = 0
        self.flush_calls = 0

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        return False

    def get(
        self,
        model: type,
        identity: int,
        **kwargs: Any,
    ) -> object | None:
        if model is Checkpoint:
            if (
                self.checkpoint is not None
                and int(
                    self.checkpoint.id
                ) == int(identity)
            ):
                return self.checkpoint

            return None

        if model is Evaluation:
            if (
                self.evaluation is not None
                and int(
                    self.evaluation.id
                ) == int(identity)
            ):
                return self.evaluation

            return None

        raise AssertionError(
            f"Unexpected model: {model}"
        )

    def add(
        self,
        value: object,
    ) -> None:
        assert isinstance(
            value,
            Evaluation,
        )
        self.evaluation = value

    def flush(self) -> None:
        self.flush_calls += 1

        if (
            self.evaluation is not None
            and self.evaluation.id is None
        ):
            self.evaluation.id = 41

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


class FakeSessionFactory:
    def __init__(
        self,
        session: FakeSession,
    ) -> None:
        self.session = session

    def __call__(self) -> FakeSession:
        return self.session


def _create_pending(
    session: FakeSession,
    *,
    policy_mode: str = "deterministic_argmax",
    threshold_action: int | None = None,
    probability_threshold: float | None = None,
):
    return create_pending_evaluation(
        FakeSessionFactory(session),
        checkpoint_id=9,
        trigger="scheduled",
        policy_mode=policy_mode,
        threshold_action=threshold_action,
        probability_threshold=(
            probability_threshold
        ),
        seed=123,
        data_scope="run_validation",
        data_path="data/test.parquet",
        data_sha256="a" * 64,
        data_rows=8,
        evaluation_start_index=4,
        evaluation_end_index=6,
        evaluation_start_at=datetime(
            2026,
            1,
            5,
            14,
            20,
            tzinfo=timezone.utc,
        ),
        evaluation_end_at=datetime(
            2026,
            1,
            5,
            14,
            25,
            tzinfo=timezone.utc,
        ),
        lookback_rows=2,
        git_commit="b" * 40,
        git_branch="master",
    )


def _result() -> EvaluationRunResult:
    metrics = calculate_evaluation_metrics(
        agent_equity=[0.0, 0.0],
        always_long_equity=[0.0, 0.0],
        always_short_equity=[0.0, 0.0],
        positions=[0, 0],
        shaped_rewards=[0.0, 0.0],
        step_swap_costs=[0.0, 0.0],
        step_swap_events=[0, 0],
        trade_events=[],
        open_position_return_at_end=0.0,
    )

    return EvaluationRunResult(
        metrics=metrics,
        steps_completed=2,
        evaluation_start_index=4,
        evaluation_end_index=6,
        lookback_rows=2,
        policy_trace=EvaluationPolicyTrace(
            execution_indices=(4, 5),
            execution_timestamps=(
                pd.Timestamp(
                    "2026-01-05 14:20:00+00:00"
                ),
                pd.Timestamp(
                    "2026-01-05 14:25:00+00:00"
                ),
            ),
            actions=(0, 0),
            probabilities=(
                (1.0, 0.0),
                (1.0, 0.0),
            ),
            selected_action_probabilities=(
                1.0,
                1.0,
            ),
            threshold_met=(
                None,
                None,
            ),
        ),
    )


def test_creates_pending_evaluation() -> None:
    session = FakeSession()

    state = _create_pending(
        session
    )

    assert state.evaluation_id == 41
    assert state.checkpoint_id == 9
    assert state.status == (
        EvaluationStatus.PENDING
    )
    assert state.steps_expected == 2
    assert state.steps_completed == 0
    assert state.started_at is None
    assert state.finished_at is None

    assert session.evaluation is not None
    assert session.evaluation.steps_expected == 2
    assert session.evaluation.git_branch == "master"

    assert session.commit_calls == 1
    assert session.rollback_calls == 0


def test_rejects_missing_checkpoint() -> None:
    session = FakeSession()
    session.checkpoint = None

    with pytest.raises(
        EvaluationCheckpointNotFoundError,
        match="Checkpoint 9 does not exist",
    ):
        _create_pending(
            session
        )

    assert session.commit_calls == 0
    assert session.rollback_calls == 1


def test_marks_pending_evaluation_running() -> None:
    session = FakeSession()
    _create_pending(session)

    state = mark_evaluation_running(
        FakeSessionFactory(session),
        evaluation_id=41,
    )

    assert state.status == (
        EvaluationStatus.RUNNING
    )
    assert state.started_at is not None
    assert state.finished_at is None
    assert session.commit_calls == 2


def test_rejects_repeated_running_transition() -> None:
    session = FakeSession()
    _create_pending(session)

    mark_evaluation_running(
        FakeSessionFactory(session),
        evaluation_id=41,
    )

    with pytest.raises(
        EvaluationStateTransitionError,
        match="running to running",
    ):
        mark_evaluation_running(
            FakeSessionFactory(session),
            evaluation_id=41,
        )

    assert session.rollback_calls == 1


def test_completes_running_evaluation_with_all_metrics() -> None:
    session = FakeSession()
    _create_pending(session)

    mark_evaluation_running(
        FakeSessionFactory(session),
        evaluation_id=41,
    )

    result = _result()

    state = complete_evaluation(
        FakeSessionFactory(session),
        evaluation_id=41,
        result=result,
    )

    assert state.status == (
        EvaluationStatus.COMPLETED
    )
    assert state.steps_completed == 2
    assert state.finished_at is not None
    assert state.balanced_score == pytest.approx(
        result.metrics.balanced_score
    )

    assert session.evaluation is not None
    assert session.evaluation.agent_return == pytest.approx(
        result.metrics.agent_return
    )
    assert session.evaluation.current_streak_type == (
        EvaluationCurrentStreakType.NONE
    )
    assert session.evaluation.error_type is None
    assert session.evaluation.error_message is None


def test_rejects_result_range_mismatch() -> None:
    session = FakeSession()
    _create_pending(session)

    mark_evaluation_running(
        FakeSessionFactory(session),
        evaluation_id=41,
    )

    mismatched_result = replace(
        _result(),
        evaluation_end_index=7,
    )

    with pytest.raises(
        EvaluationResultMismatchError,
        match="range does not match",
    ):
        complete_evaluation(
            FakeSessionFactory(session),
            evaluation_id=41,
            result=mismatched_result,
        )

    assert session.evaluation is not None
    assert session.evaluation.status == (
        EvaluationStatus.RUNNING
    )
    assert session.rollback_calls == 1


def test_can_fail_pending_evaluation() -> None:
    session = FakeSession()
    _create_pending(session)

    state = fail_evaluation(
        FakeSessionFactory(session),
        evaluation_id=41,
        error=RuntimeError(
            "CUDA evaluation failed"
        ),
        steps_completed=1,
    )

    assert state.status == (
        EvaluationStatus.FAILED
    )
    assert state.started_at is not None
    assert state.finished_at is not None
    assert state.steps_completed == 1
    assert state.error_type == "RuntimeError"
    assert state.error_message == (
        "CUDA evaluation failed"
    )


def test_rejects_invalid_threshold_action() -> None:
    session = FakeSession()

    with pytest.raises(
        EvaluationIdentityError,
        match="must be 1 or null",
    ):
        _create_pending(
            session,
            policy_mode="probability_threshold",
            threshold_action=2,
            probability_threshold=0.60,
        )

    assert session.evaluation is None
    assert session.commit_calls == 0
