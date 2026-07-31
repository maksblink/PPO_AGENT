from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GIT_COMMIT_PATTERN = re.compile(
    r"^[0-9a-f]{40}$"
)


class GitStateError(RuntimeError):
    """Base error for Git reproducibility checks."""


class GitRepositoryNotFoundError(
    GitStateError
):
    """Raised when the supplied path is not a Git repository."""


class GitRepositoryDirtyError(
    GitStateError
):
    """Raised when tracked or untracked changes are present."""


@dataclass(frozen=True, slots=True)
class CleanGitState:
    project_root: Path
    commit: str
    branch: str


def _run_git(
    project_root: Path,
    arguments: Sequence[str],
) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise GitStateError(
            "Git executable is not available."
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = (
            error.stderr.strip()
            if error.stderr
            else ""
        )

        message = (
            "Git command failed: "
            + " ".join(arguments)
        )

        if stderr:
            message += f"\n{stderr}"

        raise GitStateError(
            message
        ) from error

    return completed.stdout.strip()


def require_clean_git(
    project_root: str | Path = PROJECT_ROOT,
) -> CleanGitState:
    """
    Require one clean Git repository and return its identity.

    Both tracked changes and untracked files make the repository dirty.
    A detached HEAD is allowed and is recorded as ``detached``.
    """
    root = (
        Path(project_root)
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise GitRepositoryNotFoundError(
            f"Project root is not a directory: {root}"
        )

    try:
        repository_root = Path(
            _run_git(
                root,
                [
                    "rev-parse",
                    "--show-toplevel",
                ],
            )
        ).resolve()
    except GitStateError as error:
        raise GitRepositoryNotFoundError(
            f"Not a Git repository: {root}"
        ) from error

    if repository_root != root:
        raise GitRepositoryNotFoundError(
            "project_root must be the top-level "
            "Git repository directory.\n"
            f"Provided: {root}\n"
            f"Detected: {repository_root}"
        )

    status = _run_git(
        root,
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
    )

    if status:
        raise GitRepositoryDirtyError(
            "Git working tree must be clean before "
            "training or evaluation.\n"
            "Commit, stash, or remove these changes:\n"
            f"{status}"
        )

    commit = _run_git(
        root,
        [
            "rev-parse",
            "--verify",
            "HEAD",
        ],
    ).lower()

    if not _GIT_COMMIT_PATTERN.fullmatch(
        commit
    ):
        raise GitStateError(
            "Git HEAD is not a valid full commit hash."
        )

    branch = _run_git(
        root,
        [
            "branch",
            "--show-current",
        ],
    )

    if not branch:
        branch = "detached"

    return CleanGitState(
        project_root=root,
        commit=commit,
        branch=branch,
    )
