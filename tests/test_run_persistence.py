from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from train_and_eval.database.models import (
    Checkpoint,
    ContinuationMode,
    Run,
    RunStatus,
)
from train_and_eval.market_data.split_market_data import (
    split_market_data_chronologically,
)
from train_and_eval.reproducibility import CleanGitState
from train_and_eval.run_config import (
    LoadedRunConfig,
    ResumeContinuationSection,
    load_run_config,
    normalize_config,
)
from train_and_eval.training.persistence import (
    RunIdentityError,
    RunSourceCheckpointNotFoundError,
    RunStateTransitionError,
    complete_run,
    create_pending_run,
    fail_run,
    mark_run_running,
    update_run_progress,
)


class FakeSession:
    def __init__(self) -> None:
        self.run: Run | None = None
        self.source_checkpoint: object | None = None
        self.source_run: object | None = None
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
        if model is Run:
            if (
                self.run is not None
                and int(self.run.id) == int(identity)
            ):
                return self.run

            if (
                self.source_run is not None
                and int(self.source_run.id) == int(identity)
            ):
                return self.source_run

            return None

        if model is Checkpoint:
            if (
                self.source_checkpoint is not None
                and int(self.source_checkpoint.id) == int(identity)
            ):
                return self.source_checkpoint

            return None

        raise AssertionError(
            f"Unexpected model: {model}"
        )

    def add(self, value: object) -> None:
        assert isinstance(value, Run)
        self.run = value

    def flush(self) -> None:
        self.flush_calls += 1

        if self.run is not None and self.run.id is None:
            self.run.id = 51

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


class FakeSessionFactory:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def __call__(self) -> FakeSession:
        return self.session


def _loaded_config() -> LoadedRunConfig:
    loaded = load_run_config(
        "configs/experiments/example_5m_timesteps.yml",
        verify_data=False,
    )
    total_rows = 8000

    return replace(
        loaded,
        data_manifest_entry={
            "path": loaded.config.data.path,
            "status": "okay",
            "sha256": "a" * 64,
            "rows": total_rows,
        },
    )


def _split(loaded: LoadedRunConfig):
    frame = pd.DataFrame(
        {"row": range(8000)}
    )

    return split_market_data_chronologically(
        frame,
        train_ratio=0.8,
        window=loaded.config.environment.window,
        context=loaded.config.environment.context,
    )


def _git_state() -> CleanGitState:
    return CleanGitState(
        project_root=Path("/tmp/project"),
        commit="b" * 40,
        branch="master",
    )


def _create_pending(session: FakeSession):
    loaded = _loaded_config()
    split = _split(loaded)

    return create_pending_run(
        FakeSessionFactory(session),
        loaded_config=loaded,
        split=split,
        git_state=_git_state(),
    )


def test_creates_pending_run_from_verified_inputs() -> None:
    session = FakeSession()

    state = _create_pending(session)

    assert state.run_id == 51
    assert state.status == RunStatus.PENDING
    assert state.training_steps_requested == 500000
    assert state.training_steps_completed == 0
    assert state.data_epochs_completed == Decimal("0")
    assert state.started_at is None
    assert state.finished_at is None

    assert session.run is not None
    assert session.run.continuation_mode == ContinuationMode.FRESH
    assert session.run.source_checkpoint_id is None
    assert session.run.train_rows == 6400
    assert session.run.validation_rows == 1600
    assert session.run.steps_per_data_epoch == 255
    assert session.run.git_commit == "b" * 40
    assert session.commit_calls == 1
    assert session.rollback_calls == 0


def test_rejects_unverified_loaded_config() -> None:
    session = FakeSession()
    loaded = load_run_config(
        "configs/experiments/example_5m_timesteps.yml",
        verify_data=False,
    )

    with pytest.raises(
        RunIdentityError,
        match="verification enabled",
    ):
        create_pending_run(
            FakeSessionFactory(session),
            loaded_config=loaded,
            split=_split(_loaded_config()),
            git_state=_git_state(),
        )

    assert session.run is None


def test_resume_requires_existing_matching_checkpoint() -> None:
    session = FakeSession()
    loaded = _loaded_config()
    resumed_config = loaded.config.model_copy(
        update={
            "continuation": ResumeContinuationSection(
                mode="resume",
                source_run="source-run",
                checkpoint="best",
            )
        }
    )
    resumed_loaded = replace(
        loaded,
        config=resumed_config,
        normalized_json=normalize_config(resumed_config),
    )

    with pytest.raises(
        RunSourceCheckpointNotFoundError,
        match="does not exist",
    ):
        create_pending_run(
            FakeSessionFactory(session),
            loaded_config=resumed_loaded,
            split=_split(resumed_loaded),
            git_state=_git_state(),
            source_checkpoint_id=8,
        )

    session.source_checkpoint = SimpleNamespace(
        id=8,
        run_id=7,
    )
    session.source_run = SimpleNamespace(
        id=7,
        name="different-run",
    )

    with pytest.raises(
        RunIdentityError,
        match="does not belong",
    ):
        create_pending_run(
            FakeSessionFactory(session),
            loaded_config=resumed_loaded,
            split=_split(resumed_loaded),
            git_state=_git_state(),
            source_checkpoint_id=8,
        )


def test_running_progress_and_completion_lifecycle() -> None:
    session = FakeSession()
    _create_pending(session)

    running = mark_run_running(
        FakeSessionFactory(session),
        run_id=51,
    )

    assert running.status == RunStatus.RUNNING
    assert running.started_at is not None

    progress = update_run_progress(
        FakeSessionFactory(session),
        run_id=51,
        training_steps_completed=510,
    )

    assert progress.training_steps_completed == 510
    assert progress.data_epochs_completed == Decimal("2.00000000")

    with pytest.raises(
        RunStateTransitionError,
        match="before all requested",
    ):
        complete_run(
            FakeSessionFactory(session),
            run_id=51,
        )

    final_progress = update_run_progress(
        FakeSessionFactory(session),
        run_id=51,
        training_steps_completed=500000,
    )
    completed = complete_run(
        FakeSessionFactory(session),
        run_id=51,
    )

    assert final_progress.data_epochs_completed == Decimal(
        "1960.78431373"
    )
    assert completed.status == RunStatus.COMPLETED
    assert completed.finished_at is not None
    assert completed.training_steps_completed == 500000


def test_progress_must_be_monotonic_and_within_request() -> None:
    session = FakeSession()
    _create_pending(session)
    mark_run_running(
        FakeSessionFactory(session),
        run_id=51,
    )
    update_run_progress(
        FakeSessionFactory(session),
        run_id=51,
        training_steps_completed=100,
    )

    with pytest.raises(
        RunIdentityError,
        match="cannot decrease",
    ):
        update_run_progress(
            FakeSessionFactory(session),
            run_id=51,
            training_steps_completed=99,
        )

    with pytest.raises(
        RunIdentityError,
        match="cannot exceed",
    ):
        update_run_progress(
            FakeSessionFactory(session),
            run_id=51,
            training_steps_completed=500001,
        )


def test_failed_run_persists_progress_and_original_error() -> None:
    session = FakeSession()
    _create_pending(session)
    mark_run_running(
        FakeSessionFactory(session),
        run_id=51,
    )

    failed = fail_run(
        FakeSessionFactory(session),
        run_id=51,
        error=RuntimeError("CUDA training failed"),
        training_steps_completed=255,
    )

    assert failed.status == RunStatus.FAILED
    assert failed.training_steps_completed == 255
    assert failed.data_epochs_completed == Decimal("1.00000000")
    assert failed.started_at is not None
    assert failed.finished_at is not None
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "CUDA training failed"
