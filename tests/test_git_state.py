from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from train_and_eval.reproducibility.git_state import (
    GitRepositoryDirtyError,
    GitRepositoryNotFoundError,
    require_clean_git,
)


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_repository(
    tmp_path: Path,
) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()

    _git(
        repository,
        "init",
    )
    _git(
        repository,
        "config",
        "user.name",
        "PPO Agent Test",
    )
    _git(
        repository,
        "config",
        "user.email",
        "ppo-agent@example.invalid",
    )

    tracked_file = repository / "tracked.txt"
    tracked_file.write_text(
        "initial\n",
        encoding="utf-8",
    )

    _git(
        repository,
        "add",
        "tracked.txt",
    )
    _git(
        repository,
        "commit",
        "-m",
        "initial commit",
    )

    return repository


def test_returns_clean_repository_identity(
    tmp_path: Path,
) -> None:
    repository = _clean_repository(
        tmp_path
    )

    state = require_clean_git(
        repository
    )

    assert state.project_root == (
        repository.resolve()
    )
    assert state.commit == _git(
        repository,
        "rev-parse",
        "HEAD",
    )
    assert len(state.commit) == 40
    assert state.branch


def test_rejects_modified_tracked_file(
    tmp_path: Path,
) -> None:
    repository = _clean_repository(
        tmp_path
    )

    (
        repository / "tracked.txt"
    ).write_text(
        "modified\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GitRepositoryDirtyError,
        match="working tree must be clean",
    ):
        require_clean_git(
            repository
        )


def test_rejects_untracked_file(
    tmp_path: Path,
) -> None:
    repository = _clean_repository(
        tmp_path
    )

    (
        repository / "untracked.txt"
    ).write_text(
        "untracked\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GitRepositoryDirtyError,
        match="untracked.txt",
    ):
        require_clean_git(
            repository
        )


def test_rejects_non_repository(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "not-a-repository"
    directory.mkdir()

    with pytest.raises(
        GitRepositoryNotFoundError,
        match="Not a Git repository",
    ):
        require_clean_git(
            directory
        )
