import subprocess
from pathlib import Path

import pytest

from train_and_eval.optuna.git_guard import assert_git_clean


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("configs/\nartifacts/\n")
    (tmp_path / "tracked.txt").write_text("original\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "baseline")
    return tmp_path


def test_clean_and_ignored_config(repo):
    (repo / "configs").mkdir()
    (repo / "configs/search.yml").write_text("changed local settings")
    assert assert_git_clean(repo) == git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("kind", ["modified", "staged", "deleted", "untracked"])
def test_dirty_states_block(repo, kind):
    p = repo / "tracked.txt"
    if kind == "deleted":
        p.unlink()
    elif kind == "untracked":
        (repo / "new.py").write_text("new")
    else:
        p.write_text("changed")
        if kind == "staged":
            git(repo, "add", "tracked.txt")
    with pytest.raises(RuntimeError, match="not clean"):
        assert_git_clean(repo)


def test_non_repo_blocks(tmp_path):
    with pytest.raises(RuntimeError, match="Could not check"):
        assert_git_clean(tmp_path)


@pytest.mark.parametrize("validation", [False, True])
def test_cli_blocks_before_config_db_and_reports(monkeypatch, validation):
    from train_and_eval.optuna import __main__ as cli
    def blocked(root):
        assert isinstance(root, Path)
        raise RuntimeError("dirty project")
    monkeypatch.setattr(cli, "assert_git_clean", blocked)
    for name in ("load", "snapshot", "save"):
        monkeypatch.setattr(cli, name, lambda *a, **k: pytest.fail("Side effect before Git guard"))
    args = ["--config", "missing.yml"] + (["--validate-config"] if validation else [])
    with pytest.raises(RuntimeError, match="dirty project"):
        cli.main(args)
