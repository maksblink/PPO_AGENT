from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import train_and_eval.evaluation.service as service
from train_and_eval.database.models import (
    Checkpoint,
    EvaluationStatus,
    Run,
)
from train_and_eval.evaluation.persistence import (
    PersistedEvaluationState,
)
from train_and_eval.reproducibility import (
    CleanGitState,
    GitRepositoryDirtyError,
)
from train_and_eval.run_config import (
    load_run_config,
)


class FakeSession:
    def __init__(
        self,
        checkpoint: object,
        run: object,
    ) -> None:
        self.checkpoint = checkpoint
        self.run = run

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
    ) -> object | None:
        if model is Checkpoint:
            if int(identity) == int(
                self.checkpoint.id
            ):
                return self.checkpoint

            return None

        if model is Run:
            if int(identity) == int(
                self.run.id
            ):
                return self.run

            return None

        raise AssertionError(
            f"Unexpected model: {model}"
        )


class FakeSessionFactory:
    def __init__(
        self,
        session: FakeSession,
    ) -> None:
        self.session = session
        self.calls = 0

    def __call__(self) -> FakeSession:
        self.calls += 1
        return self.session


def _normalized_config() -> dict[str, Any]:
    loaded = load_run_config(
        (
            "configs/experiments/"
            "example_5m_timesteps.yml"
        ),
        verify_data=False,
    )

    config = loaded.config.model_dump(
        mode="json"
    )

    config["run"]["seed"] = 123
    config["data"]["path"] = (
        "data/test.parquet"
    )
    config["environment"]["window"] = 2
    config["environment"][
        "position_side"
    ] = "long_only"
    config["ppo"]["device"] = "cuda"
    config["evaluation"]["device"] = "cpu"

    return config


def _session_factory() -> FakeSessionFactory:
    checkpoint = SimpleNamespace(
        id=9,
        run_id=4,
        relative_path=(
            "artifacts/runs/00000004/"
            "checkpoints/model.zip"
        ),
        sha256="b" * 64,
        size_bytes=100,
        model_step=8,
    )

    run = SimpleNamespace(
        id=4,
        seed=123,
        config_schema_version=1,
        normalized_config_json=(
            _normalized_config()
        ),
        data_path="data/test.parquet",
        data_sha256="a" * 64,
        split_index=4,
        train_rows=4,
        validation_rows=4,
        git_commit="c" * 40,
        git_branch="master",
    )

    return FakeSessionFactory(
        FakeSession(
            checkpoint,
            run,
        )
    )


def _market_data(
    *,
    sha256: str = "a" * 64,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "DT": pd.date_range(
                "2026-01-05 14:00:00+00:00",
                periods=8,
                freq="5min",
                tz="UTC",
            ),
        }
    )
    frame.attrs["source_path"] = (
        "data/test.parquet"
    )
    frame.attrs["sha256"] = sha256
    return frame


def _state(
    status: EvaluationStatus,
) -> PersistedEvaluationState:
    now = datetime.now(
        timezone.utc
    )

    return PersistedEvaluationState(
        evaluation_id=17,
        checkpoint_id=9,
        status=status,
        steps_expected=4,
        steps_completed=(
            4
            if status
            == EvaluationStatus.COMPLETED
            else 0
        ),
        started_at=(
            None
            if status
            == EvaluationStatus.PENDING
            else now
        ),
        finished_at=(
            now
            if status
            in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
            }
            else None
        ),
        balanced_score=(
            0.25
            if status
            == EvaluationStatus.COMPLETED
            else None
        ),
        error_type=None,
        error_message=None,
    )


def _patch_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
) -> None:
    monkeypatch.setattr(
        service,
        "require_clean_git",
        lambda root: (
            events.append("clean_git")
            or CleanGitState(
                project_root=Path(root),
                commit="c" * 40,
                branch="master",
            )
        ),
    )

    monkeypatch.setattr(
        service,
        "get_context_definition",
        lambda name: SimpleNamespace(
            required_history_rows=(
                lambda window: 2
            )
        ),
    )

    monkeypatch.setattr(
        service,
        "load_market_data",
        lambda *args, **kwargs: (
            events.append("load_data")
            or _market_data()
        ),
    )


def test_runs_complete_validation_evaluation_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_setup(
        monkeypatch,
        tmp_path,
        events,
    )

    session_factory = (
        _session_factory()
    )
    model = object()
    result = object()

    def create_pending(*args, **kwargs):
        events.append("pending")
        assert kwargs[
            "evaluation_start_index"
        ] == 4
        assert kwargs[
            "evaluation_end_index"
        ] == 8
        assert kwargs["lookback_rows"] == 2
        assert kwargs["git_commit"] == (
            "c" * 40
        )
        return _state(
            EvaluationStatus.PENDING
        )

    monkeypatch.setattr(
        service,
        "create_pending_evaluation",
        create_pending,
    )
    monkeypatch.setattr(
        service,
        "mark_evaluation_running",
        lambda *args, **kwargs: (
            events.append("running")
            or _state(
                EvaluationStatus.RUNNING
            )
        ),
    )
    def load_model(*args, **kwargs):
        events.append("load_model")
        assert kwargs["device"] == "cpu"
        return model

    monkeypatch.setattr(
        service,
        "load_persisted_ppo_checkpoint",
        load_model,
    )

    def run_evaluation(
        received_model,
        market_data,
        environment_config,
        **kwargs,
    ):
        events.append("runner")
        assert received_model is model
        assert kwargs[
            "evaluation_start_index"
        ] == 4
        assert kwargs[
            "evaluation_end_index"
        ] == 8
        assert kwargs["lookback_rows"] == 2
        return result

    monkeypatch.setattr(
        service,
        "run_ppo_evaluation",
        run_evaluation,
    )

    def complete(*args, **kwargs):
        events.append("completed")
        assert kwargs["result"] is result
        return _state(
            EvaluationStatus.COMPLETED
        )

    monkeypatch.setattr(
        service,
        "complete_evaluation",
        complete,
    )

    completed = (
        service
        .evaluate_run_validation_checkpoint(
            session_factory,
            checkpoint_id=9,
            trigger="manual",
            policy_mode=(
                "deterministic_argmax"
            ),
            project_root=tmp_path,
        )
    )

    assert completed.status == (
        EvaluationStatus.COMPLETED
    )
    assert events == [
        "clean_git",
        "load_data",
        "pending",
        "running",
        "load_model",
        "runner",
        "completed",
    ]


def test_records_failed_state_and_reraises_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_setup(
        monkeypatch,
        tmp_path,
        events,
    )

    monkeypatch.setattr(
        service,
        "create_pending_evaluation",
        lambda *args, **kwargs: (
            events.append("pending")
            or _state(
                EvaluationStatus.PENDING
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_evaluation_running",
        lambda *args, **kwargs: (
            events.append("running")
            or _state(
                EvaluationStatus.RUNNING
            )
        ),
    )

    def fail_load(*args, **kwargs):
        events.append("load_model")
        raise RuntimeError(
            "broken checkpoint"
        )

    monkeypatch.setattr(
        service,
        "load_persisted_ppo_checkpoint",
        fail_load,
    )

    def persist_failure(*args, **kwargs):
        events.append("failed")
        assert isinstance(
            kwargs["error"],
            RuntimeError,
        )
        return _state(
            EvaluationStatus.FAILED
        )

    monkeypatch.setattr(
        service,
        "fail_evaluation",
        persist_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="broken checkpoint",
    ):
        (
            service
            .evaluate_run_validation_checkpoint(
                _session_factory(),
                checkpoint_id=9,
                trigger="manual",
                policy_mode=(
                    "deterministic_argmax"
                ),
                project_root=tmp_path,
            )
        )

    assert events == [
        "clean_git",
        "load_data",
        "pending",
        "running",
        "load_model",
        "failed",
    ]


def test_rejects_data_hash_mismatch_before_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        service,
        "require_clean_git",
        lambda root: CleanGitState(
            project_root=Path(root),
            commit="c" * 40,
            branch="master",
        ),
    )
    monkeypatch.setattr(
        service,
        "load_market_data",
        lambda *args, **kwargs: (
            _market_data(
                sha256="f" * 64
            )
        ),
    )

    monkeypatch.setattr(
        service,
        "create_pending_evaluation",
        lambda *args, **kwargs: (
            events.append("pending")
        ),
    )

    with pytest.raises(
        service.EvaluationSourceMismatchError,
        match="SHA-256",
    ):
        (
            service
            .evaluate_run_validation_checkpoint(
                _session_factory(),
                checkpoint_id=9,
                trigger="manual",
                policy_mode=(
                    "deterministic_argmax"
                ),
                project_root=tmp_path,
            )
        )

    assert events == []


def test_dirty_git_is_rejected_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        service,
        "require_clean_git",
        lambda root: (_ for _ in ()).throw(
            GitRepositoryDirtyError(
                "dirty repository"
            )
        ),
    )

    class ForbiddenSessionFactory:
        def __call__(self):
            raise AssertionError(
                "Database must not be accessed."
            )

    with pytest.raises(
        GitRepositoryDirtyError,
        match="dirty repository",
    ):
        (
            service
            .evaluate_run_validation_checkpoint(
                ForbiddenSessionFactory(),
                checkpoint_id=9,
                trigger="manual",
                policy_mode=(
                    "deterministic_argmax"
                ),
                project_root=tmp_path,
            )
        )


def test_preserves_original_error_when_failure_persistence_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_setup(
        monkeypatch,
        tmp_path,
        events,
    )

    monkeypatch.setattr(
        service,
        "create_pending_evaluation",
        lambda *args, **kwargs: (
            events.append("pending")
            or _state(
                EvaluationStatus.PENDING
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_evaluation_running",
        lambda *args, **kwargs: (
            events.append("running")
            or _state(
                EvaluationStatus.RUNNING
            )
        ),
    )

    original_error = RuntimeError(
        "original evaluation failure"
    )

    def fail_model_load(*args, **kwargs):
        events.append("load_model")
        raise original_error

    monkeypatch.setattr(
        service,
        "load_persisted_ppo_checkpoint",
        fail_model_load,
    )

    def fail_failure_persistence(
        *args,
        **kwargs,
    ):
        events.append("failed_persistence")
        assert kwargs["error"] is original_error
        raise OSError(
            "database unavailable"
        )

    monkeypatch.setattr(
        service,
        "fail_evaluation",
        fail_failure_persistence,
    )

    with pytest.raises(
        RuntimeError,
        match="original evaluation failure",
    ) as captured:
        (
            service
            .evaluate_run_validation_checkpoint(
                _session_factory(),
                checkpoint_id=9,
                trigger="manual",
                policy_mode=(
                    "deterministic_argmax"
                ),
                project_root=tmp_path,
            )
        )

    assert captured.value is original_error
    assert events == [
        "clean_git",
        "load_data",
        "pending",
        "running",
        "load_model",
        "failed_persistence",
    ]

    notes = getattr(
        captured.value,
        "__notes__",
        [],
    )

    assert len(notes) == 1
    assert (
        "persisting the failed evaluation "
        "state failed"
    ) in notes[0]
    assert "OSError" in notes[0]
    assert "database unavailable" in notes[0]
