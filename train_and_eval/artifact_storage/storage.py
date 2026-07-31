from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIRECTORY = "artifacts"
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024

_ALLOWED_CHECKPOINT_SAVE_REASONS = frozenset(
    {
        "initial",
        "periodic",
        "final",
        "manual",
        "interrupted",
    }
)


class ArtifactStorageError(RuntimeError):
    """Base error for artifact storage failures."""


class ArtifactPathError(ArtifactStorageError):
    """Raised when an artifact path is unsafe or invalid."""


class ArtifactSourceError(ArtifactStorageError):
    """Raised when a source artifact cannot be used."""


class ArtifactAlreadyExistsError(
    ArtifactStorageError
):
    """Raised when immutable artifact already exists."""


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    relative_path: str
    absolute_path: Path
    sha256: str
    size_bytes: int


def calculate_artifact_sha256(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be at least 1."
        )

    digest = hashlib.sha256()

    with Path(path).open("rb") as artifact_file:
        while chunk := artifact_file.read(
            chunk_size
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _integer_value(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise ArtifactPathError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ArtifactPathError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise ArtifactPathError(
            f"{name} must be an integer."
        )

    if result < minimum:
        raise ArtifactPathError(
            f"{name} must be at least {minimum}."
        )

    return result


def _save_reason_value(
    save_reason: Any,
) -> str:
    raw_value = getattr(
        save_reason,
        "value",
        save_reason,
    )

    if not isinstance(raw_value, str):
        raise ArtifactPathError(
            "save_reason must be a string or enum value."
        )

    value = raw_value.strip()

    if value not in _ALLOWED_CHECKPOINT_SAVE_REASONS:
        allowed = ", ".join(
            sorted(
                _ALLOWED_CHECKPOINT_SAVE_REASONS
            )
        )

        raise ArtifactPathError(
            "Unsupported checkpoint save_reason "
            f"{value!r}. Allowed values: {allowed}."
        )

    return value


class ArtifactStorage:
    """
    Store immutable project artifacts under one directory.

    Paths returned by this class are relative to the project root
    and may be stored directly in the database.
    """

    def __init__(
        self,
        *,
        project_root: str | Path = PROJECT_ROOT,
        artifacts_directory: str | Path = (
            DEFAULT_ARTIFACTS_DIRECTORY
        ),
    ) -> None:
        self.project_root = (
            Path(project_root)
            .expanduser()
            .resolve()
        )

        directory = Path(
            artifacts_directory
        )

        if directory.is_absolute():
            raise ArtifactPathError(
                "artifacts_directory must be relative "
                "to project_root."
            )

        self.artifacts_root = (
            self.project_root / directory
        ).expanduser().resolve()

        try:
            relative_directory = (
                self.artifacts_root.relative_to(
                    self.project_root
                )
            )
        except ValueError as error:
            raise ArtifactPathError(
                "artifacts_directory escapes "
                "the project root."
            ) from error

        if relative_directory.as_posix() == ".":
            raise ArtifactPathError(
                "artifacts_directory must not be "
                "the project root itself."
            )

        self.artifacts_relative_path = (
            relative_directory.as_posix()
        )

    def checkpoint_relative_path(
        self,
        *,
        run_id: int,
        run_step: int,
        save_reason: Any,
    ) -> str:
        resolved_run_id = _integer_value(
            run_id,
            name="run_id",
            minimum=1,
        )
        resolved_run_step = _integer_value(
            run_step,
            name="run_step",
            minimum=0,
        )
        resolved_reason = _save_reason_value(
            save_reason
        )

        return (
            f"{self.artifacts_relative_path}/"
            f"runs/{resolved_run_id:08d}/"
            "checkpoints/"
            f"step_{resolved_run_step:012d}_"
            f"{resolved_reason}.zip"
        )

    def _resolve_target(
        self,
        relative_path: str,
    ) -> tuple[str, Path]:
        if not isinstance(relative_path, str):
            raise ArtifactPathError(
                "Artifact relative path must be a string."
            )

        normalized_text = relative_path.strip()

        if not normalized_text:
            raise ArtifactPathError(
                "Artifact relative path must not be empty."
            )

        path = Path(normalized_text)

        if path.is_absolute():
            raise ArtifactPathError(
                "Artifact relative path must not "
                "be absolute."
            )

        absolute_path = (
            self.project_root / path
        ).expanduser().resolve()

        try:
            absolute_path.relative_to(
                self.artifacts_root
            )
        except ValueError as error:
            raise ArtifactPathError(
                "Artifact path escapes the artifact "
                "storage directory."
            ) from error

        if absolute_path == self.artifacts_root:
            raise ArtifactPathError(
                "Artifact path must identify a file."
            )

        normalized_relative_path = (
            absolute_path.relative_to(
                self.project_root
            ).as_posix()
        )

        return (
            normalized_relative_path,
            absolute_path,
        )

    @staticmethod
    def _validate_source(
        source_path: str | Path,
    ) -> Path:
        source = (
            Path(source_path)
            .expanduser()
            .resolve()
        )

        if not source.exists():
            raise ArtifactSourceError(
                "Source artifact does not exist: "
                f"{source}"
            )

        if not source.is_file():
            raise ArtifactSourceError(
                "Source artifact is not a regular file: "
                f"{source}"
            )

        return source

    def store_file(
        self,
        source_path: str | Path,
        *,
        relative_path: str,
    ) -> StoredArtifact:
        """
        Copy a completed file into immutable artifact storage.

        The copy is written to a temporary file in the destination
        directory and then linked atomically to its final name.
        Existing artifacts are never overwritten.
        """
        source = self._validate_source(
            source_path
        )
        (
            normalized_relative_path,
            target,
        ) = self._resolve_target(
            relative_path
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if target.exists():
            raise ArtifactAlreadyExistsError(
                "Artifact already exists: "
                f"{normalized_relative_path}"
            )

        file_descriptor, temporary_name = (
            tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
        )

        temporary_path = Path(
            temporary_name
        )

        try:
            with (
                source.open("rb") as source_file,
                os.fdopen(
                    file_descriptor,
                    "wb",
                ) as temporary_file,
            ):
                shutil.copyfileobj(
                    source_file,
                    temporary_file,
                    length=DEFAULT_HASH_CHUNK_SIZE,
                )
                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            try:
                os.link(
                    temporary_path,
                    target,
                )
            except FileExistsError as error:
                raise ArtifactAlreadyExistsError(
                    "Artifact already exists: "
                    f"{normalized_relative_path}"
                ) from error

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

        try:
            size_bytes = target.stat().st_size

            if size_bytes < 1:
                raise ArtifactSourceError(
                    "Stored artifact must not be empty."
                )

            sha256 = calculate_artifact_sha256(
                target
            )

        except Exception:
            target.unlink(
                missing_ok=True
            )
            raise

        return StoredArtifact(
            relative_path=normalized_relative_path,
            absolute_path=target,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def store_checkpoint_file(
        self,
        source_path: str | Path,
        *,
        run_id: int,
        run_step: int,
        save_reason: Any,
    ) -> StoredArtifact:
        relative_path = (
            self.checkpoint_relative_path(
                run_id=run_id,
                run_step=run_step,
                save_reason=save_reason,
            )
        )

        return self.store_file(
            source_path,
            relative_path=relative_path,
        )
