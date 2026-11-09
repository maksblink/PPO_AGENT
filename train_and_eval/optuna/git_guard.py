"""Require a committed project state before running the proposal CLI."""
import subprocess
from pathlib import Path


def assert_git_clean(root: Path) -> str:
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode:
            raise RuntimeError("Could not check Git state:\n" + result.stderr.strip())
        return result.stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none")
    if dirty:
        raise RuntimeError(
            "Git working tree is not clean.\n"
            "Commit or discard changes before running Optuna.\n\n" + dirty
        )
    return git("rev-parse", "--verify", "HEAD")
