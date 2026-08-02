from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import train_and_eval.training.service as service
from train_and_eval.checkpoints.persistence import (
    PersistedCheckpoint,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
    RunStatus,
)
from train_and_eval.reproducibility import (
    CleanGitState,
    GitRepositoryDirtyError,
)
from train_and_eval.run_config import (
    LoadedRunConfig,
    RunConfig,
    load_run_config,
    normalize_config,
)
from train_and_eval.training.execution import (
    ExactPPOTrainingResult,
)
from train_and_eval.training.persistence import (
    PersistedRunState,
)


class FakeModel:
    def __init__(
        self,
        *,
        num_timesteps: int,
    ) -> None:
        self.num_timesteps = num_timesteps


def _loaded_config(
    *,
    resume: bool = False,
) -> LoadedRunConfig:
    loaded = load_run_config(
        "configs/experiments/"
        "example_5m_timesteps.yml",
        verify_data=False,
    )
    raw = loaded.config.model_dump(
        mode="json"
    )
    raw["run"]["name"] = (
        "resumed-run"
        if resume
        else "fresh-run"
    )
    raw["run"]["seed"] = 123
    raw["ppo"]["device"] = "cpu"
    raw["training"]["duration_unit"] = (
        "timesteps"
    )
    raw["training"]["duration_amount"] = 10

    if resume:
        raw["continuation"] = {
            "mode": "resume",
            "source_run": "source-run",
            "checkpoint": "final",
        }

    config = RunConfig.model_validate(
        raw
    )

    return replace(
        loaded,
        normalized_json=(
            normalize_config(config)
        ),
        config=config,
        data_manifest_entry={
            "path": config.data.path,
            "status": "okay",
            "sha256": "a" * 64,
            "rows": 8000,
        },
    )


def _split() -> SimpleNamespace:
    return SimpleNamespace(
        train_data=pd.DataFrame(
            {"row": range(7000)}
        ),
        training_start_index=6144,
        train_rows=7000,
        validation_rows=1000,
        split_index=7000,
        steps_per_data_epoch=856,
    )


def _run_state(
    status: RunStatus,
    *,
    requested: int = 10,
    completed: int = 0,
) -> PersistedRunState:
    return PersistedRunState(
        run_id=51,
        name="fresh-run",
        status=status,
        training_steps_requested=requested,
        training_steps_completed=completed,
        data_epochs_completed=Decimal("0"),
        started_at=None,
        finished_at=None,
        error_type=None,
        error_message=None,
    )


def _checkpoint(
    *,
    save_reason: CheckpointSaveReason,
    run_step: int,
    model_step: int,
) -> PersistedCheckpoint:
    return PersistedCheckpoint(
        checkpoint_id=91,
        run_id=51,
        run_step=run_step,
        model_step=model_step,
        save_reason=save_reason,
        relative_path=(
            "artifacts/runs/00000051/"
            "checkpoints/model.zip"
        ),
        absolute_path=Path(
            "/tmp/model.zip"
        ),
        sha256="b" * 64,
        size_bytes=100,
    )


def _training_result(
    *,
    before: int,
    completed: int,
    requested: int = 10,
    stopped_early: bool = False,
) -> ExactPPOTrainingResult:
    return ExactPPOTrainingResult(
        model_steps_before=before,
        model_steps_after=(
            before + completed
        ),
        local_steps_requested=requested,
        local_steps_completed=completed,
        rollout_sizes=(completed,),
        stopped_early=stopped_early,
    )


def _patch_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
    *,
    loaded: LoadedRunConfig | None = None,
) -> LoadedRunConfig:
    resolved_loaded = (
        _loaded_config()
        if loaded is None
        else loaded
    )

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
        "load_run_config",
        lambda *args, **kwargs: (
            events.append("load_config")
            or resolved_loaded
        ),
    )
    monkeypatch.setattr(
        service,
        "_load_and_split_data",
        lambda *args, **kwargs: (
            events.append("load_data")
            or (
                pd.DataFrame(),
                _split(),
            )
        ),
    )

    return resolved_loaded


def test_executes_complete_fresh_training_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
    )
    model = FakeModel(
        num_timesteps=0
    )

    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: (
            events.append("pending")
            or _run_state(
                RunStatus.PENDING
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: (
            events.append("running")
            or _run_state(
                RunStatus.RUNNING
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "TradingEnvironment",
        lambda *args, **kwargs: (
            events.append("environment")
            or object()
        ),
    )
    monkeypatch.setattr(
        service,
        "create_ppo_model",
        lambda *args, **kwargs: (
            events.append("create_model")
            or model
        ),
    )

    def train(*args, **kwargs):
        events.append("train")
        assert kwargs["total_timesteps"] == 10
        model.num_timesteps = 10
        return _training_result(
            before=0,
            completed=10,
        )

    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        train,
    )
    monkeypatch.setattr(
        service,
        "update_run_progress",
        lambda *args, **kwargs: (
            events.append("progress")
            or _run_state(
                RunStatus.RUNNING,
                completed=10,
            )
        ),
    )

    def persist(*args, **kwargs):
        events.append("final_checkpoint")
        assert kwargs["run_step"] == 10
        assert kwargs["save_reason"] == (
            CheckpointSaveReason.FINAL
        )
        return _checkpoint(
            save_reason=(
                CheckpointSaveReason.FINAL
            ),
            run_step=10,
            model_step=10,
        )

    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        persist,
    )
    monkeypatch.setattr(
        service,
        "complete_run",
        lambda *args, **kwargs: (
            events.append("completed")
            or _run_state(
                RunStatus.COMPLETED,
                completed=10,
            )
        ),
    )

    result = service.train_ppo_run(
        object(),
        config_path="run.yml",
        project_root=tmp_path,
        log_interval=None,
    )

    assert result.run.status == (
        RunStatus.COMPLETED
    )
    assert result.checkpoint.run_step == 10
    assert result.source_checkpoint_id is None
    assert events == [
        "clean_git",
        "load_config",
        "load_data",
        "pending",
        "running",
        "environment",
        "create_model",
        "train",
        "progress",
        "final_checkpoint",
        "completed",
    ]


def test_resume_loads_source_with_current_training_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loaded = _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
        loaded=_loaded_config(
            resume=True
        ),
    )
    source = service.ResumeTrainingSource(
        checkpoint_id=88,
        run_id=44,
        relative_path="artifacts/source.zip",
        sha256="d" * 64,
        size_bytes=100,
        model_step=100,
        source_config=loaded.config,
    )
    model = FakeModel(
        num_timesteps=100
    )

    monkeypatch.setattr(
        service,
        "_resolve_resume_source",
        lambda *args, **kwargs: (
            events.append("resolve_source")
            or source
        ),
    )

    def pending(*args, **kwargs):
        events.append("pending")
        assert kwargs[
            "source_checkpoint_id"
        ] == 88
        return _run_state(
            RunStatus.PENDING,
            requested=10,
        )

    monkeypatch.setattr(
        service,
        "create_pending_run",
        pending,
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: (
            events.append("running")
            or _run_state(
                RunStatus.RUNNING
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "TradingEnvironment",
        lambda *args, **kwargs: object(),
    )

    def load_model(*args, **kwargs):
        events.append("load_model")
        assert kwargs[
            "training_config"
        ] == loaded.config.ppo
        assert kwargs["seed"] == 123
        return model

    monkeypatch.setattr(
        service,
        "load_persisted_ppo_checkpoint",
        load_model,
    )

    def train(*args, **kwargs):
        events.append("train")
        model.num_timesteps = 110
        return _training_result(
            before=100,
            completed=10,
        )

    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        train,
    )
    monkeypatch.setattr(
        service,
        "update_run_progress",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING,
            completed=10,
        ),
    )
    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        lambda *args, **kwargs: _checkpoint(
            save_reason=(
                CheckpointSaveReason.FINAL
            ),
            run_step=10,
            model_step=110,
        ),
    )
    monkeypatch.setattr(
        service,
        "complete_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.COMPLETED,
            completed=10,
        ),
    )

    result = service.train_ppo_run(
        object(),
        config_path="resume.yml",
        project_root=tmp_path,
    )

    assert result.source_checkpoint_id == 88
    assert result.training.model_steps_before == 100
    assert result.training.model_steps_after == 110
    assert events[:5] == [
        "clean_git",
        "load_config",
        "load_data",
        "resolve_source",
        "pending",
    ]
    assert "load_model" in events


def test_training_failure_persists_interrupted_checkpoint_and_failed_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
    )
    model = FakeModel(
        num_timesteps=0
    )

    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.PENDING
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING
        ),
    )
    monkeypatch.setattr(
        service,
        "TradingEnvironment",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "create_ppo_model",
        lambda *args, **kwargs: model,
    )

    original_error = RuntimeError(
        "training exploded"
    )

    def fail_training(*args, **kwargs):
        model.num_timesteps = 3
        raise original_error

    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        fail_training,
    )

    def persist(*args, **kwargs):
        events.append("interrupted")
        assert kwargs["run_step"] == 3
        assert kwargs["save_reason"] == (
            CheckpointSaveReason.INTERRUPTED
        )
        return _checkpoint(
            save_reason=(
                CheckpointSaveReason.INTERRUPTED
            ),
            run_step=3,
            model_step=3,
        )

    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        persist,
    )

    def fail_state(*args, **kwargs):
        events.append("failed")
        assert kwargs["error"] is original_error
        assert kwargs[
            "training_steps_completed"
        ] == 3
        return _run_state(
            RunStatus.FAILED,
            completed=3,
        )

    monkeypatch.setattr(
        service,
        "fail_run",
        fail_state,
    )

    with pytest.raises(
        RuntimeError,
        match="training exploded",
    ) as captured:
        service.train_ppo_run(
            object(),
            config_path="run.yml",
            project_root=tmp_path,
        )

    assert captured.value is original_error
    assert events[-2:] == [
        "interrupted",
        "failed",
    ]


def test_does_not_write_interrupted_after_final_checkpoint_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
    )
    model = FakeModel(
        num_timesteps=0
    )

    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.PENDING
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING
        ),
    )
    monkeypatch.setattr(
        service,
        "TradingEnvironment",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "create_ppo_model",
        lambda *args, **kwargs: model,
    )

    def train(*args, **kwargs):
        model.num_timesteps = 10
        return _training_result(
            before=0,
            completed=10,
        )

    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        train,
    )
    monkeypatch.setattr(
        service,
        "update_run_progress",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING,
            completed=10,
        ),
    )

    def persist(*args, **kwargs):
        events.append(
            kwargs["save_reason"].value
        )
        return _checkpoint(
            save_reason=(
                CheckpointSaveReason.FINAL
            ),
            run_step=10,
            model_step=10,
        )

    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        persist,
    )
    completion_error = OSError(
        "database unavailable"
    )
    monkeypatch.setattr(
        service,
        "complete_run",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(completion_error),
    )
    monkeypatch.setattr(
        service,
        "fail_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.FAILED,
            completed=10,
        ),
    )

    with pytest.raises(
        OSError,
        match="database unavailable",
    ):
        service.train_ppo_run(
            object(),
            config_path="run.yml",
            project_root=tmp_path,
        )

    assert events.count("final") == 1
    assert "interrupted" not in events


def test_preserves_original_error_when_recovery_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
    )
    model = FakeModel(
        num_timesteps=0
    )
    original_error = RuntimeError(
        "original failure"
    )

    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.PENDING
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING
        ),
    )
    monkeypatch.setattr(
        service,
        "TradingEnvironment",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "create_ppo_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(original_error),
    )
    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(
            OSError("artifact unavailable")
        ),
    )
    monkeypatch.setattr(
        service,
        "fail_run",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(
            ConnectionError("database unavailable")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="original failure",
    ) as captured:
        service.train_ppo_run(
            object(),
            config_path="run.yml",
            project_root=tmp_path,
        )

    assert captured.value is original_error
    notes = getattr(
        captured.value,
        "__notes__",
        [],
    )
    assert len(notes) == 2
    assert "interrupted PPO checkpoint" in notes[0]
    assert "artifact unavailable" in notes[0]
    assert "failed run state" in notes[1]
    assert "database unavailable" in notes[1]


def test_stopped_early_is_recorded_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
    )
    model = FakeModel(
        num_timesteps=0
    )

    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.PENDING
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING
        ),
    )
    monkeypatch.setattr(
        service,
        "TradingEnvironment",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "create_ppo_model",
        lambda *args, **kwargs: model,
    )

    def train(*args, **kwargs):
        model.num_timesteps = 4
        return _training_result(
            before=0,
            completed=4,
            stopped_early=True,
        )

    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        train,
    )
    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        lambda *args, **kwargs: _checkpoint(
            save_reason=(
                CheckpointSaveReason.INTERRUPTED
            ),
            run_step=4,
            model_step=4,
        ),
    )
    monkeypatch.setattr(
        service,
        "fail_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.FAILED,
            completed=4,
        ),
    )

    with pytest.raises(
        service.TrainingStoppedEarlyError,
        match="stopped before",
    ):
        service.train_ppo_run(
            object(),
            config_path="run.yml",
            project_root=tmp_path,
        )


def test_dirty_git_is_rejected_before_config_and_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        service,
        "require_clean_git",
        lambda root: (
            _ for _ in ()
        ).throw(
            GitRepositoryDirtyError(
                "dirty repository"
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "load_run_config",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(
            AssertionError(
                "Config must not be loaded."
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
        service.train_ppo_run(
            ForbiddenSessionFactory(),
            config_path="run.yml",
            project_root=tmp_path,
        )


class ResumeLookupSession:
    def __init__(
        self,
        *,
        scalar_values: list[object | None],
        checkpoint_by_id: object | None = None,
    ) -> None:
        self.scalar_values = list(
            scalar_values
        )
        self.checkpoint_by_id = (
            checkpoint_by_id
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        return False

    def scalar(self, statement):
        return self.scalar_values.pop(0)

    def get(self, model, identity):
        return self.checkpoint_by_id


class ResumeLookupFactory:
    def __init__(
        self,
        session: ResumeLookupSession,
    ) -> None:
        self.session = session

    def __call__(self):
        return self.session


def _source_run_and_config():
    current = _loaded_config(
        resume=True
    ).config
    raw = current.model_dump(
        mode="json"
    )
    raw["run"]["name"] = "source-run"
    raw["continuation"] = {
        "mode": "fresh",
    }
    source_config = RunConfig.model_validate(
        raw
    )
    source_run = SimpleNamespace(
        id=44,
        normalized_config_json=(
            source_config.model_dump(
                mode="json"
            )
        ),
        config_schema_version=1,
        seed=123,
        data_path=source_config.data.path,
    )

    return current, source_run, source_config


def test_resolves_final_resume_checkpoint() -> None:
    current, source_run, _ = (
        _source_run_and_config()
    )
    checkpoint = SimpleNamespace(
        id=88,
        run_id=44,
        relative_path="artifacts/source.zip",
        sha256="d" * 64,
        size_bytes=100,
        model_step=500,
    )
    session = ResumeLookupSession(
        scalar_values=[
            source_run,
            checkpoint,
        ]
    )

    source = service._resolve_resume_source(
        ResumeLookupFactory(session),
        current_config=current,
    )

    assert source.checkpoint_id == 88
    assert source.run_id == 44
    assert source.model_step == 500
    assert source.source_config.run.name == (
        "source-run"
    )


def test_rejects_missing_resume_source_run() -> None:
    current = _loaded_config(
        resume=True
    ).config
    session = ResumeLookupSession(
        scalar_values=[None]
    )

    with pytest.raises(
        service.TrainingSourceNotFoundError,
        match="Source run",
    ):
        service._resolve_resume_source(
            ResumeLookupFactory(session),
            current_config=current,
        )


def test_rejects_numeric_checkpoint_from_other_run() -> None:
    current, source_run, _ = (
        _source_run_and_config()
    )
    raw = current.model_dump(
        mode="json"
    )
    raw["continuation"]["checkpoint"] = "88"
    numeric_config = RunConfig.model_validate(
        raw
    )
    checkpoint = SimpleNamespace(
        id=88,
        run_id=99,
        relative_path="artifacts/wrong.zip",
        sha256="d" * 64,
        size_bytes=100,
        model_step=500,
    )
    session = ResumeLookupSession(
        scalar_values=[source_run],
        checkpoint_by_id=checkpoint,
    )

    with pytest.raises(
        service.TrainingSourceMismatchError,
        match="does not belong",
    ):
        service._resolve_resume_source(
            ResumeLookupFactory(session),
            current_config=numeric_config,
        )
