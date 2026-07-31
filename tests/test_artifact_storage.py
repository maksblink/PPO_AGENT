from __future__ import annotations

import hashlib

import pytest

from train_and_eval.artifact_storage.storage import (
    ArtifactAlreadyExistsError,
    ArtifactPathError,
    ArtifactSourceError,
    ArtifactStorage,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_checkpoint_relative_path_is_stable(
    tmp_path,
) -> None:
    storage = ArtifactStorage(
        project_root=tmp_path,
    )

    relative_path = (
        storage.checkpoint_relative_path(
            run_id=42,
            run_step=500_000,
            save_reason="final",
        )
    )

    assert relative_path == (
        "artifacts/runs/00000042/"
        "checkpoints/"
        "step_000000500000_final.zip"
    )


def test_stores_checkpoint_and_returns_metadata(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    source = tmp_path / "source-model.zip"
    content = b"immutable PPO model"
    source.write_bytes(content)

    storage = ArtifactStorage(
        project_root=project_root,
    )

    artifact = storage.store_checkpoint_file(
        source,
        run_id=7,
        run_step=1234,
        save_reason="periodic",
    )

    assert artifact.relative_path == (
        "artifacts/runs/00000007/"
        "checkpoints/"
        "step_000000001234_periodic.zip"
    )
    assert artifact.absolute_path.is_file()
    assert artifact.absolute_path.read_bytes() == content
    assert artifact.sha256 == _sha256(content)
    assert artifact.size_bytes == len(content)

    # The source file remains unchanged.
    assert source.read_bytes() == content

    temporary_files = list(
        artifact.absolute_path.parent.glob(
            ".*.tmp"
        )
    )
    assert temporary_files == []


def test_existing_checkpoint_is_never_overwritten(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    first_source = tmp_path / "first.zip"
    second_source = tmp_path / "second.zip"

    first_source.write_bytes(b"first model")
    second_source.write_bytes(b"second model")

    storage = ArtifactStorage(
        project_root=project_root,
    )

    first_artifact = (
        storage.store_checkpoint_file(
            first_source,
            run_id=1,
            run_step=100,
            save_reason="final",
        )
    )

    with pytest.raises(
        ArtifactAlreadyExistsError,
        match="already exists",
    ):
        storage.store_checkpoint_file(
            second_source,
            run_id=1,
            run_step=100,
            save_reason="final",
        )

    assert (
        first_artifact.absolute_path.read_bytes()
        == b"first model"
    )

    temporary_files = list(
        first_artifact.absolute_path.parent.glob(
            ".*.tmp"
        )
    )
    assert temporary_files == []


@pytest.mark.parametrize(
    (
        "run_id",
        "run_step",
        "save_reason",
        "expected_message",
    ),
    [
        (
            0,
            10,
            "final",
            "run_id must be at least 1",
        ),
        (
            1,
            -1,
            "final",
            "run_step must be at least 0",
        ),
        (
            1,
            10,
            "unknown",
            "Unsupported checkpoint save_reason",
        ),
        (
            True,
            10,
            "final",
            "run_id must be an integer",
        ),
    ],
)
def test_rejects_invalid_checkpoint_identity(
    tmp_path,
    run_id,
    run_step,
    save_reason,
    expected_message,
) -> None:
    storage = ArtifactStorage(
        project_root=tmp_path,
    )

    with pytest.raises(
        ArtifactPathError,
        match=expected_message,
    ):
        storage.checkpoint_relative_path(
            run_id=run_id,
            run_step=run_step,
            save_reason=save_reason,
        )


def test_rejects_missing_source_file(
    tmp_path,
) -> None:
    storage = ArtifactStorage(
        project_root=tmp_path,
    )

    with pytest.raises(
        ArtifactSourceError,
        match="does not exist",
    ):
        storage.store_checkpoint_file(
            tmp_path / "missing.zip",
            run_id=1,
            run_step=0,
            save_reason="initial",
        )


def test_rejects_source_directory(
    tmp_path,
) -> None:
    source_directory = tmp_path / "model"
    source_directory.mkdir()

    storage = ArtifactStorage(
        project_root=tmp_path,
    )

    with pytest.raises(
        ArtifactSourceError,
        match="not a regular file",
    ):
        storage.store_checkpoint_file(
            source_directory,
            run_id=1,
            run_step=0,
            save_reason="initial",
        )
