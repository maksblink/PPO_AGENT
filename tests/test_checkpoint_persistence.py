from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from train_and_eval.artifact_storage.storage import (
    ArtifactStorage,
)
from train_and_eval.checkpoints.persistence import (
    CheckpointIdentityError,
    CheckpointRunNotFoundError,
    persist_checkpoint_file,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
)


class FakeSession:
    def __init__(
        self,
        *,
        run_exists: bool = True,
        flush_error: Exception | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.run_exists = run_exists
        self.flush_error = flush_error
        self.commit_error = commit_error

        self.added: list[object] = []
        self.rollback_calls = 0
        self.commit_calls = 0
        self.get_calls: list[
            tuple[type, int]
        ] = []

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
        self.get_calls.append(
            (model, identity)
        )

        if not self.run_exists:
            return None

        return SimpleNamespace(
            id=identity
        )

    def add(
        self,
        value: object,
    ) -> None:
        self.added.append(value)

    def flush(self) -> None:
        if self.flush_error is not None:
            raise self.flush_error

        checkpoint = self.added[-1]
        checkpoint.id = 73

    def commit(self) -> None:
        self.commit_calls += 1

        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        self.rollback_calls += 1


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


def _source_file(
    tmp_path: Path,
    *,
    content: bytes = b"PPO checkpoint",
) -> Path:
    source = tmp_path / "source-model.zip"
    source.write_bytes(content)
    return source


def test_persists_file_and_checkpoint_record(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession()
    factory = FakeSessionFactory(
        session
    )
    storage = ArtifactStorage(
        project_root=project_root,
    )

    result = persist_checkpoint_file(
        factory,
        storage,
        source_path=_source_file(
            tmp_path
        ),
        run_id=12,
        run_step=50_000,
        model_step=150_000,
        save_reason=(
            CheckpointSaveReason.PERIODIC
        ),
    )

    assert result.checkpoint_id == 73
    assert result.run_id == 12
    assert result.run_step == 50_000
    assert result.model_step == 150_000
    assert result.save_reason == (
        CheckpointSaveReason.PERIODIC
    )

    assert result.relative_path == (
        "artifacts/runs/00000012/"
        "checkpoints/"
        "step_000000050000_periodic.zip"
    )
    assert result.absolute_path.is_file()
    assert result.size_bytes > 0
    assert len(result.sha256) == 64

    assert factory.calls == 1
    assert session.commit_calls == 1
    assert session.rollback_calls == 0
    assert len(session.added) == 1

    checkpoint = session.added[0]

    assert checkpoint.run_id == 12
    assert checkpoint.run_step == 50_000
    assert checkpoint.model_step == 150_000
    assert checkpoint.save_reason == (
        CheckpointSaveReason.PERIODIC
    )
    assert checkpoint.relative_path == (
        result.relative_path
    )
    assert checkpoint.sha256 == result.sha256
    assert checkpoint.size_bytes == (
        result.size_bytes
    )


def test_missing_run_does_not_store_artifact(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession(
        run_exists=False
    )
    storage = ArtifactStorage(
        project_root=project_root,
    )

    with pytest.raises(
        CheckpointRunNotFoundError,
        match="does not exist",
    ):
        persist_checkpoint_file(
            FakeSessionFactory(session),
            storage,
            source_path=_source_file(
                tmp_path
            ),
            run_id=999,
            run_step=0,
            model_step=0,
            save_reason="initial",
        )

    assert session.added == []
    assert session.commit_calls == 0

    stored_files = list(
        project_root.glob(
            "artifacts/**/*"
        )
    )
    assert stored_files == []


def test_flush_failure_removes_new_artifact(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession(
        flush_error=RuntimeError(
            "database flush failed"
        )
    )
    storage = ArtifactStorage(
        project_root=project_root,
    )

    expected_path = (
        project_root
        / "artifacts"
        / "runs"
        / "00000001"
        / "checkpoints"
        / "step_000000000100_final.zip"
    )

    with pytest.raises(
        RuntimeError,
        match="database flush failed",
    ):
        persist_checkpoint_file(
            FakeSessionFactory(session),
            storage,
            source_path=_source_file(
                tmp_path
            ),
            run_id=1,
            run_step=100,
            model_step=100,
            save_reason="final",
        )

    assert not expected_path.exists()
    assert session.rollback_calls == 1
    assert session.commit_calls == 0


def test_commit_failure_retains_artifact(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession(
        commit_error=RuntimeError(
            "database commit failed"
        )
    )
    storage = ArtifactStorage(
        project_root=project_root,
    )

    expected_path = (
        project_root
        / "artifacts"
        / "runs"
        / "00000001"
        / "checkpoints"
        / "step_000000000100_final.zip"
    )

    with pytest.raises(
        RuntimeError,
        match="database commit failed",
    ):
        persist_checkpoint_file(
            FakeSessionFactory(session),
            storage,
            source_path=_source_file(
                tmp_path
            ),
            run_id=1,
            run_step=100,
            model_step=100,
            save_reason="final",
        )

    assert expected_path.is_file()
    assert session.rollback_calls == 1
    assert session.commit_calls == 1


def test_rejects_model_step_before_run_step(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession()
    factory = FakeSessionFactory(
        session
    )

    with pytest.raises(
        CheckpointIdentityError,
        match=(
            "model_step must be greater than "
            "or equal to run_step"
        ),
    ):
        persist_checkpoint_file(
            factory,
            ArtifactStorage(
                project_root=project_root,
            ),
            source_path=_source_file(
                tmp_path
            ),
            run_id=1,
            run_step=101,
            model_step=100,
            save_reason="periodic",
        )

    assert factory.calls == 0
    assert session.added == []
