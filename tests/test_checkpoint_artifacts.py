from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from train_and_eval.checkpoints.artifacts import (
    CheckpointArtifactHashMismatchError,
    CheckpointArtifactMissingError,
    CheckpointArtifactPathError,
    CheckpointArtifactSizeMismatchError,
    resolve_checkpoint_artifact,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _checkpoint(
    *,
    relative_path: str,
    content: bytes,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=17,
        relative_path=relative_path,
        sha256=_sha256(content),
        size_bytes=len(content),
    )


def test_resolves_and_verifies_checkpoint_file(
    tmp_path,
) -> None:
    content = b"immutable PPO checkpoint"
    checkpoint_path = (
        tmp_path
        / "artifacts"
        / "runs"
        / "one"
        / "model.zip"
    )
    checkpoint_path.parent.mkdir(
        parents=True
    )
    checkpoint_path.write_bytes(content)

    artifact = resolve_checkpoint_artifact(
        _checkpoint(
            relative_path=(
                "artifacts/runs/one/model.zip"
            ),
            content=content,
        ),
        project_root=tmp_path,
    )

    assert artifact.checkpoint_id == 17
    assert artifact.relative_path == (
        "artifacts/runs/one/model.zip"
    )
    assert artifact.absolute_path == (
        checkpoint_path.resolve()
    )
    assert artifact.sha256 == _sha256(content)
    assert artifact.size_bytes == len(content)


def test_rejects_missing_checkpoint_file(
    tmp_path,
) -> None:
    with pytest.raises(
        CheckpointArtifactMissingError,
        match="does not exist",
    ):
        resolve_checkpoint_artifact(
            _checkpoint(
                relative_path=(
                    "artifacts/runs/missing.zip"
                ),
                content=b"missing",
            ),
            project_root=tmp_path,
        )


def test_rejects_absolute_checkpoint_path(
    tmp_path,
) -> None:
    checkpoint = _checkpoint(
        relative_path=str(
            tmp_path / "model.zip"
        ),
        content=b"model",
    )

    with pytest.raises(
        CheckpointArtifactPathError,
        match="must not be absolute",
    ):
        resolve_checkpoint_artifact(
            checkpoint,
            project_root=tmp_path,
        )


def test_rejects_checkpoint_path_escape(
    tmp_path,
) -> None:
    with pytest.raises(
        CheckpointArtifactPathError,
        match="escapes the project root",
    ):
        resolve_checkpoint_artifact(
            _checkpoint(
                relative_path="../outside.zip",
                content=b"outside",
            ),
            project_root=tmp_path,
        )


def test_rejects_checkpoint_outside_artifact_storage(
    tmp_path,
) -> None:
    outside_path = tmp_path / "outside.zip"
    content = b"outside artifact storage"
    outside_path.write_bytes(content)

    with pytest.raises(
        CheckpointArtifactPathError,
        match=(
            "escapes the artifact storage directory"
        ),
    ):
        resolve_checkpoint_artifact(
            _checkpoint(
                relative_path="outside.zip",
                content=content,
            ),
            project_root=tmp_path,
        )


def test_rejects_checkpoint_size_mismatch(
    tmp_path,
) -> None:
    content = b"checkpoint"
    checkpoint_path = (
        tmp_path
        / "artifacts"
        / "model.zip"
    )
    checkpoint_path.parent.mkdir(
        parents=True
    )
    checkpoint_path.write_bytes(content)

    checkpoint = _checkpoint(
        relative_path="artifacts/model.zip",
        content=content,
    )
    checkpoint.size_bytes += 1

    with pytest.raises(
        CheckpointArtifactSizeMismatchError,
        match="size mismatch",
    ):
        resolve_checkpoint_artifact(
            checkpoint,
            project_root=tmp_path,
        )


def test_rejects_checkpoint_hash_mismatch(
    tmp_path,
) -> None:
    checkpoint_path = (
        tmp_path
        / "artifacts"
        / "model.zip"
    )
    checkpoint_path.parent.mkdir(
        parents=True
    )
    checkpoint_path.write_bytes(
        b"actual checkpoint"
    )

    checkpoint = _checkpoint(
        relative_path="artifacts/model.zip",
        content=b"different checkpoint",
    )
    checkpoint.size_bytes = (
        checkpoint_path.stat().st_size
    )

    with pytest.raises(
        CheckpointArtifactHashMismatchError,
        match="SHA-256 mismatch",
    ):
        resolve_checkpoint_artifact(
            checkpoint,
            project_root=tmp_path,
        )


def test_stored_checkpoint_can_be_resolved(
    tmp_path,
) -> None:
    from train_and_eval.artifact_storage.storage import (
        ArtifactStorage,
    )

    source_path = tmp_path / "temporary-model.zip"
    source_content = b"complete PPO checkpoint artifact"
    source_path.write_bytes(source_content)

    storage = ArtifactStorage(
        project_root=tmp_path,
    )

    stored = storage.store_checkpoint_file(
        source_path,
        run_id=9,
        run_step=250_000,
        save_reason="final",
    )

    checkpoint = SimpleNamespace(
        id=31,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        size_bytes=stored.size_bytes,
    )

    resolved = resolve_checkpoint_artifact(
        checkpoint,
        project_root=tmp_path,
    )

    assert resolved.checkpoint_id == 31
    assert resolved.relative_path == stored.relative_path
    assert resolved.absolute_path == stored.absolute_path
    assert resolved.sha256 == stored.sha256
    assert resolved.size_bytes == stored.size_bytes
    assert resolved.absolute_path.read_bytes() == source_content
