from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from train_and_eval.artifact_storage.storage import (
    DEFAULT_ARTIFACTS_DIRECTORY,
    DEFAULT_HASH_CHUNK_SIZE,
    calculate_artifact_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHA256_HEX_LENGTH = 64


class CheckpointRecord(Protocol):
    """Database fields needed to resolve one checkpoint artifact."""

    id: int
    relative_path: str
    sha256: str
    size_bytes: int


class CheckpointArtifactError(RuntimeError):
    """Base error for an invalid checkpoint artifact."""


class CheckpointArtifactPathError(
    CheckpointArtifactError
):
    """Raised when a checkpoint path is unsafe or invalid."""


class CheckpointArtifactMissingError(
    CheckpointArtifactError
):
    """Raised when the checkpoint file does not exist."""


class CheckpointArtifactSizeMismatchError(
    CheckpointArtifactError
):
    """Raised when checkpoint size differs from database metadata."""


class CheckpointArtifactHashMismatchError(
    CheckpointArtifactError
):
    """Raised when checkpoint SHA-256 differs from metadata."""


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    checkpoint_id: int
    relative_path: str
    absolute_path: Path
    sha256: str
    size_bytes: int


def calculate_checkpoint_sha256(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Calculate a checkpoint file SHA-256 digest."""
    return calculate_artifact_sha256(
        path,
        chunk_size=chunk_size,
    )


def _validate_expected_sha256(
    value: str,
) -> str:
    if len(value) != SHA256_HEX_LENGTH:
        raise CheckpointArtifactError(
            "Checkpoint SHA-256 must contain "
            "exactly 64 hexadecimal characters."
        )

    if value != value.lower():
        raise CheckpointArtifactError(
            "Checkpoint SHA-256 must be lowercase."
        )

    try:
        int(value, 16)
    except ValueError as error:
        raise CheckpointArtifactError(
            "Checkpoint SHA-256 contains "
            "non-hexadecimal characters."
        ) from error

    return value


def _resolve_safe_path(
    relative_path: str,
    *,
    project_root: str | Path,
    artifacts_directory: str | Path,
) -> tuple[str, Path]:
    if not isinstance(relative_path, str):
        raise CheckpointArtifactPathError(
            "Checkpoint relative_path must be a string."
        )

    normalized_text = relative_path.strip()

    if not normalized_text:
        raise CheckpointArtifactPathError(
            "Checkpoint relative_path must not be empty."
        )

    path = Path(normalized_text)

    if path.is_absolute():
        raise CheckpointArtifactPathError(
            "Checkpoint relative_path must not be absolute."
        )

    root = (
        Path(project_root)
        .expanduser()
        .resolve()
    )

    artifacts_path = Path(
        artifacts_directory
    )

    if artifacts_path.is_absolute():
        raise CheckpointArtifactPathError(
            "artifacts_directory must be relative "
            "to project_root."
        )

    artifacts_root = (
        root / artifacts_path
    ).expanduser().resolve()

    try:
        artifacts_root.relative_to(root)
    except ValueError as error:
        raise CheckpointArtifactPathError(
            "Artifact storage directory escapes "
            "the project root."
        ) from error

    if artifacts_root == root:
        raise CheckpointArtifactPathError(
            "Artifact storage directory must not "
            "be the project root."
        )

    resolved_path = (
        root / path
    ).expanduser().resolve()

    try:
        resolved_path.relative_to(root)
    except ValueError as error:
        raise CheckpointArtifactPathError(
            "Checkpoint path escapes the project root."
        ) from error

    try:
        resolved_path.relative_to(
            artifacts_root
        )
    except ValueError as error:
        raise CheckpointArtifactPathError(
            "Checkpoint path escapes the artifact "
            "storage directory."
        ) from error

    if resolved_path == artifacts_root:
        raise CheckpointArtifactPathError(
            "Checkpoint path must identify a file."
        )

    normalized_relative_path = (
        resolved_path.relative_to(root).as_posix()
    )

    return (
        normalized_relative_path,
        resolved_path,
    )


def resolve_checkpoint_artifact(
    checkpoint: CheckpointRecord,
    *,
    project_root: str | Path = PROJECT_ROOT,
    artifacts_directory: str | Path = (
        DEFAULT_ARTIFACTS_DIRECTORY
    ),
    verify_hash: bool = True,
) -> CheckpointArtifact:
    """
    Resolve and verify one immutable checkpoint artifact.

    The checkpoint must be located inside the project's configured
    artifact storage directory.
    """
    relative_path, absolute_path = _resolve_safe_path(
        checkpoint.relative_path,
        project_root=project_root,
        artifacts_directory=artifacts_directory,
    )

    if not absolute_path.exists():
        raise CheckpointArtifactMissingError(
            "Checkpoint file does not exist: "
            f"{relative_path}"
        )

    if not absolute_path.is_file():
        raise CheckpointArtifactPathError(
            "Checkpoint path does not identify "
            f"a regular file: {relative_path}"
        )

    expected_size = int(
        checkpoint.size_bytes
    )

    if expected_size < 1:
        raise CheckpointArtifactError(
            "Checkpoint size_bytes must be positive."
        )

    actual_size = absolute_path.stat().st_size

    if actual_size != expected_size:
        raise CheckpointArtifactSizeMismatchError(
            "Checkpoint size mismatch for "
            f"{relative_path}: expected "
            f"{expected_size}, found {actual_size}."
        )

    expected_sha256 = _validate_expected_sha256(
        str(checkpoint.sha256)
    )

    if verify_hash:
        actual_sha256 = (
            calculate_checkpoint_sha256(
                absolute_path
            )
        )

        if actual_sha256 != expected_sha256:
            raise CheckpointArtifactHashMismatchError(
                "Checkpoint SHA-256 mismatch for "
                f"{relative_path}: expected "
                f"{expected_sha256}, found "
                f"{actual_sha256}."
            )

    return CheckpointArtifact(
        checkpoint_id=int(checkpoint.id),
        relative_path=relative_path,
        absolute_path=absolute_path,
        sha256=expected_sha256,
        size_bytes=actual_size,
    )
