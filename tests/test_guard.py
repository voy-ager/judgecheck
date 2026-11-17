"""Tests for the Claude Code guard hook in .claude/hooks/guard.py."""

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

GUARD = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load()


def _repo(path: Path, branch: str) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", branch, str(path)], check=True)  # noqa: S603, S607
    return path


@pytest.fixture
def on_main(tmp_path: Path) -> Path:
    return _repo(tmp_path / "main_repo", "main")


@pytest.fixture
def on_feature(tmp_path: Path) -> Path:
    return _repo(tmp_path / "feature_repo", "p0/1-x")


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 3 --squash",
        "git push origin main",
        "git push -f origin p0/1-x",
        "git push --force-with-lease",
        "git commit --no-verify -m x",
        'git config user.email "a@b.c"',
        "gh repo delete x/y",
        "gh api -X DELETE repos/x/y/rulesets/1",
        "gh api repos/x/y/rulesets/1 -X PUT --input a",
        "git commit -m x",
        "git push",
        "git --git-dir=other/.git commit -m x",
    ],
)
def test_blocks_on_main(command: str, on_main: Path) -> None:
    assert guard.check(command, str(on_main)) is not None


@pytest.mark.parametrize(
    "command",
    [
        "git push -u origin p0/1-charter",
        "git config --local --list",
        "gh pr create --fill",
        "gh api repos/x/y/rulesets --input r.json",
        "git status",
        "uv run pytest",
    ],
)
def test_allows_on_main(command: str, on_main: Path) -> None:
    assert guard.check(command, str(on_main)) is None


def test_commit_allowed_on_feature_branch(on_feature: Path) -> None:
    assert guard.check("git commit -m x", str(on_feature)) is None


def test_follows_cd_into_feature_worktree(on_main: Path, on_feature: Path) -> None:
    assert guard.check(f'cd "{on_feature}" && git commit -m x', str(on_main)) is None


def test_follows_git_c_into_feature_worktree(on_main: Path, on_feature: Path) -> None:
    assert guard.check(f'git -C "{on_feature}" commit -m x', str(on_main)) is None


def test_follows_cd_into_main_checkout(on_main: Path, on_feature: Path) -> None:
    assert guard.check(f'cd "{on_main}" && git commit -m x', str(on_feature)) is not None


def test_follows_git_c_into_main_checkout(on_main: Path, on_feature: Path) -> None:
    assert guard.check(f'git -C "{on_main}" commit -m x', str(on_feature)) is not None
