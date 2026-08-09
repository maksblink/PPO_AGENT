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
    PPOTrainingUpdate,
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
    checkpoint_every_steps: int = 50_000,
    eval_every_steps: int = 50_000,
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
    raw["evaluation"][
        "checkpoint_every_steps"
    ] = checkpoint_every_steps
    raw["evaluation"][
        "eval_every_steps"
    ] = eval_every_steps

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
        stopped_early=False,
        early_stop_reason=None,
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
    monkeypatch.setattr(
        service,
        "evaluate_run_validation_checkpoint",
        lambda *args, **kwargs: (
            events.append(
                f"{kwargs['trigger']}_evaluation"
            )
            or SimpleNamespace(
                evaluation_id=301,
                checkpoint_id=kwargs[
                    "checkpoint_id"
                ],
                balanced_score=0.25,
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
        "final_evaluation",
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
        assert args[0].id == 88
        assert args[0].checkpoint_id == 88
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
        self.statements: list[object] = []

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
        self.statements.append(statement)
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



def test_resolves_best_resume_checkpoint_by_balanced_score() -> None:
    current, source_run, _ = (
        _source_run_and_config()
    )
    raw = current.model_dump(
        mode="json"
    )
    raw["continuation"]["checkpoint"] = "best"
    best_config = RunConfig.model_validate(raw)
    checkpoint = SimpleNamespace(
        id=77,
        run_id=44,
        relative_path="artifacts/best.zip",
        sha256="e" * 64,
        size_bytes=120,
        model_step=400,
    )
    session = ResumeLookupSession(
        scalar_values=[
            source_run,
            checkpoint,
        ]
    )

    source = service._resolve_resume_source(
        ResumeLookupFactory(session),
        current_config=best_config,
    )

    assert source.checkpoint_id == 77
    query_text = str(session.statements[1]).lower()
    assert "balanced_score desc" in query_text
    assert "evaluations.status" in query_text
    assert "evaluations.data_scope" in query_text

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



def test_executes_periodic_checkpoint_and_evaluation_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
        loaded=_loaded_config(
            checkpoint_every_steps=4,
            eval_every_steps=6,
        ),
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

    trained_segments: list[int] = []

    def train(*args, **kwargs):
        segment = kwargs["total_timesteps"]
        before = model.num_timesteps
        model.num_timesteps += segment
        trained_segments.append(segment)
        return _training_result(
            before=before,
            completed=segment,
            requested=segment,
        )

    monkeypatch.setattr(
        service,
        "learn_ppo_exact_timesteps",
        train,
    )

    progress_steps: list[int] = []

    def progress(*args, **kwargs):
        completed = kwargs[
            "training_steps_completed"
        ]
        progress_steps.append(completed)
        return _run_state(
            RunStatus.RUNNING,
            completed=completed,
        )

    monkeypatch.setattr(
        service,
        "update_run_progress",
        progress,
    )

    checkpoint_steps: list[
        tuple[int, CheckpointSaveReason]
    ] = []

    def persist(*args, **kwargs):
        run_step = kwargs["run_step"]
        reason = kwargs["save_reason"]
        checkpoint_steps.append(
            (run_step, reason)
        )
        return replace(
            _checkpoint(
                save_reason=reason,
                run_step=run_step,
                model_step=model.num_timesteps,
            ),
            checkpoint_id=90 + run_step,
        )

    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        persist,
    )

    evaluation_calls: list[
        tuple[int, str]
    ] = []

    def evaluate(*args, **kwargs):
        evaluation_calls.append(
            (
                kwargs["checkpoint_id"],
                kwargs["trigger"],
            )
        )
        assert kwargs["policy_mode"] == (
            "deterministic_argmax"
        )
        assert kwargs["threshold_action"] is None
        assert (
            kwargs["probability_threshold"]
            is None
        )
        return SimpleNamespace(
            evaluation_id=(
                300 + len(evaluation_calls)
            ),
            checkpoint_id=kwargs["checkpoint_id"],
            balanced_score=0.1,
        )

    monkeypatch.setattr(
        service,
        "evaluate_run_validation_checkpoint",
        evaluate,
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
        config_path="run.yml",
        project_root=tmp_path,
        log_interval=None,
    )

    assert trained_segments == [4, 2, 2, 2]
    assert progress_steps == [4, 6, 8, 10]
    assert checkpoint_steps == [
        (4, CheckpointSaveReason.PERIODIC),
        (6, CheckpointSaveReason.PERIODIC),
        (8, CheckpointSaveReason.PERIODIC),
        (10, CheckpointSaveReason.FINAL),
    ]
    assert evaluation_calls == [
        (96, "scheduled"),
        (100, "final"),
    ]
    assert len(result.checkpoints) == 4
    assert len(result.evaluations) == 2
    assert result.checkpoint.run_step == 10
    assert result.training.rollout_sizes == (
        4,
        2,
        2,
        2,
    )
    assert (
        result.training.local_steps_completed
        == 10
    )


def test_evaluation_failure_uses_existing_checkpoint_without_duplicate_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
        loaded=_loaded_config(
            checkpoint_every_steps=4,
            eval_every_steps=4,
        ),
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
        segment = kwargs["total_timesteps"]
        before = model.num_timesteps
        model.num_timesteps += segment
        return _training_result(
            before=before,
            completed=segment,
            requested=segment,
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
            completed=kwargs[
                "training_steps_completed"
            ],
        ),
    )

    save_reasons: list[
        CheckpointSaveReason
    ] = []

    def persist(*args, **kwargs):
        save_reasons.append(
            kwargs["save_reason"]
        )
        return _checkpoint(
            save_reason=kwargs["save_reason"],
            run_step=kwargs["run_step"],
            model_step=model.num_timesteps,
        )

    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        persist,
    )
    evaluation_error = RuntimeError(
        "validation failed"
    )
    monkeypatch.setattr(
        service,
        "evaluate_run_validation_checkpoint",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(evaluation_error),
    )

    failed_steps: list[int] = []

    def fail_state(*args, **kwargs):
        failed_steps.append(
            kwargs[
                "training_steps_completed"
            ]
        )
        return _run_state(
            RunStatus.FAILED,
            completed=4,
        )

    monkeypatch.setattr(
        service,
        "fail_run",
        fail_state,
    )

    with pytest.raises(
        RuntimeError,
        match="validation failed",
    ):
        service.train_ppo_run(
            object(),
            config_path="run.yml",
            project_root=tmp_path,
        )

    assert save_reasons == [
        CheckpointSaveReason.PERIODIC
    ]
    assert failed_steps == [4]


def test_early_stopping_creates_final_checkpoint_at_same_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loaded = _loaded_config(
        checkpoint_every_steps=100,
        eval_every_steps=4,
    )
    raw = loaded.config.model_dump(
        mode="json"
    )
    raw["training"]["duration_amount"] = 20
    raw["evaluation"][
        "early_stop_patience_evals"
    ] = 2
    early_config = RunConfig.model_validate(raw)
    loaded = replace(
        loaded,
        config=early_config,
        normalized_json=normalize_config(
            early_config
        ),
    )
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
        loaded=loaded,
    )

    model = FakeModel(num_timesteps=0)
    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.PENDING,
            requested=20,
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING,
            requested=20,
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

    trained_segments: list[int] = []

    def train(*args, **kwargs):
        segment = kwargs["total_timesteps"]
        before = model.num_timesteps
        model.num_timesteps += segment
        trained_segments.append(segment)
        return _training_result(
            before=before,
            completed=segment,
            requested=segment,
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
            requested=20,
            completed=kwargs[
                "training_steps_completed"
            ],
        ),
    )

    saved: list[tuple[int, CheckpointSaveReason]] = []

    def persist(*args, **kwargs):
        run_step = kwargs["run_step"]
        reason = kwargs["save_reason"]
        saved.append((run_step, reason))
        return replace(
            _checkpoint(
                save_reason=reason,
                run_step=run_step,
                model_step=model.num_timesteps,
            ),
            checkpoint_id=(
                100 + len(saved)
            ),
        )

    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        persist,
    )

    scores = iter([
        1.0,
        0.9,
        0.8,
        0.8,
    ])
    evaluation_calls: list[tuple[int, str, float]] = []

    def evaluate(*args, **kwargs):
        score = next(scores)
        evaluation_calls.append((
            kwargs["checkpoint_id"],
            kwargs["trigger"],
            score,
        ))
        return SimpleNamespace(
            evaluation_id=(
                300 + len(evaluation_calls)
            ),
            checkpoint_id=kwargs[
                "checkpoint_id"
            ],
            balanced_score=score,
        )

    monkeypatch.setattr(
        service,
        "evaluate_run_validation_checkpoint",
        evaluate,
    )

    completed_calls: list[dict[str, Any]] = []

    def complete(*args, **kwargs):
        completed_calls.append(kwargs)
        return replace(
            _run_state(
                RunStatus.COMPLETED,
                requested=20,
                completed=12,
            ),
            stopped_early=True,
            early_stop_reason=kwargs[
                "early_stop_reason"
            ],
        )

    monkeypatch.setattr(
        service,
        "complete_run",
        complete,
    )

    result = service.train_ppo_run(
        object(),
        config_path="run.yml",
        project_root=tmp_path,
    )

    assert trained_segments == [4, 4, 4]
    assert saved == [
        (4, CheckpointSaveReason.PERIODIC),
        (8, CheckpointSaveReason.PERIODIC),
        (12, CheckpointSaveReason.PERIODIC),
        (12, CheckpointSaveReason.FINAL),
    ]
    assert [
        trigger
        for _, trigger, _ in evaluation_calls
    ] == [
        "scheduled",
        "scheduled",
        "scheduled",
        "final",
    ]
    assert completed_calls[0][
        "stopped_early"
    ] is True
    assert result.training.stopped_early is True
    assert result.training.local_steps_completed == 12
    assert result.training.local_steps_requested == 20
    assert result.checkpoint.save_reason == (
        CheckpointSaveReason.FINAL
    )
    assert result.checkpoint.run_step == 12
    assert result.best_checkpoint.run_step == 4
    assert result.best_evaluation.balanced_score == 1.0
    assert result.early_stop_reason is not None


def test_early_stopping_patience_resets_after_improvement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracker = service.BalancedScoreEarlyStopping(
        patience=2
    )

    decisions = [
        tracker.observe(
            checkpoint_id=index,
            evaluation_id=100 + index,
            balanced_score=score,
        )
        for index, score in enumerate(
            [1.0, 0.9, 1.1, 1.0, 0.9],
            start=1,
        )
    ]

    assert decisions[1].no_improvement_evals == 1
    assert decisions[2].improved is True
    assert decisions[2].no_improvement_evals == 0
    assert decisions[3].should_stop is False
    assert decisions[4].should_stop is True
    assert decisions[4].best_checkpoint_id == 3


def test_final_evaluation_can_become_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The tracker is the same mechanism used by the service for scheduled
    # and final evaluations. A final improvement must replace prior best.
    tracker = service.BalancedScoreEarlyStopping(
        patience=5
    )
    tracker.observe(
        checkpoint_id=1,
        evaluation_id=11,
        balanced_score=0.1,
    )
    final = tracker.observe(
        checkpoint_id=2,
        evaluation_id=12,
        balanced_score=0.2,
    )

    assert final.improved is True
    assert final.best_checkpoint_id == 2
    assert final.best_evaluation_id == 12


def test_live_progress_uses_configured_step_cadence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loaded = _loaded_config(
        checkpoint_every_steps=25_000,
        eval_every_steps=25_000,
    )
    raw = loaded.config.model_dump(mode="json")
    raw["training"]["duration_amount"] = 25_000
    raw["logging"] = {
        "training_progress_every_steps": 10_000,
        "validation_progress_every_steps": 2_000,
    }
    config = RunConfig.model_validate(raw)
    loaded = replace(
        loaded,
        config=config,
        normalized_json=normalize_config(config),
    )
    _patch_preflight(
        monkeypatch,
        tmp_path,
        events,
        loaded=loaded,
    )

    model = FakeModel(num_timesteps=0)
    monkeypatch.setattr(
        service,
        "create_pending_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.PENDING,
            requested=25_000,
        ),
    )
    monkeypatch.setattr(
        service,
        "mark_run_running",
        lambda *args, **kwargs: _run_state(
            RunStatus.RUNNING,
            requested=25_000,
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
        callback = kwargs["update_callback"]
        assert callback is not None

        update_steps = [
            2_048,
            10_240,
            12_288,
            20_480,
            25_000,
        ]
        previous = 0
        for iteration, step in enumerate(
            update_steps,
            start=1,
        ):
            model.num_timesteps = step
            callback(
                PPOTrainingUpdate(
                    model_steps=step,
                    local_steps_completed=step,
                    rollout_iteration=iteration,
                    rollout_size=step - previous,
                    metrics={
                        "n_updates": iteration * 10,
                    },
                )
            )
            previous = step

        return ExactPPOTrainingResult(
            model_steps_before=0,
            model_steps_after=25_000,
            local_steps_requested=25_000,
            local_steps_completed=25_000,
            rollout_sizes=(25_000,),
            stopped_early=False,
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
            requested=25_000,
            completed=kwargs[
                "training_steps_completed"
            ],
        ),
    )
    monkeypatch.setattr(
        service,
        "persist_ppo_checkpoint",
        lambda *args, **kwargs: _checkpoint(
            save_reason=kwargs["save_reason"],
            run_step=kwargs["run_step"],
            model_step=model.num_timesteps,
        ),
    )

    evaluation_intervals: list[int] = []

    def evaluate(*args, **kwargs):
        evaluation_intervals.append(
            kwargs["progress_interval_steps"]
        )
        return SimpleNamespace(
            evaluation_id=301,
            checkpoint_id=kwargs["checkpoint_id"],
            balanced_score=0.25,
        )

    monkeypatch.setattr(
        service,
        "evaluate_run_validation_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        service,
        "_validation_progress_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service,
        "complete_run",
        lambda *args, **kwargs: _run_state(
            RunStatus.COMPLETED,
            requested=25_000,
            completed=25_000,
        ),
    )

    class Reporter:
        def __init__(self) -> None:
            self.training_steps: list[int] = []

        def start(self, **kwargs) -> None:
            pass

        def training_update(self, **kwargs) -> None:
            self.training_steps.append(
                kwargs["completed_steps"]
            )

        def validation_started(self, **kwargs) -> None:
            pass

        def validation_update(self, **kwargs) -> None:
            pass

        def validation_completed(self, *args, **kwargs) -> None:
            pass

        def finish(self, **kwargs) -> None:
            pass

        def fail(self, *args, **kwargs) -> None:
            pass

    reporter = Reporter()

    service.train_ppo_run(
        object(),
        config_path="run.yml",
        project_root=tmp_path,
        progress_reporter=reporter,
    )

    assert reporter.training_steps == [
        10_240,
        20_480,
        25_000,
    ]
    assert evaluation_intervals == [2_000]
