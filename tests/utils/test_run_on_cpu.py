import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from soe.utils.run import RunEfficiency, create_run, make_run_id, read_run_info
from soe.utils.run_card import write_run_card

EXAMPLE = Path("examples/grpo_trainer/m1_dev_qwen35-2b.yaml")


def git_repo(path: Path) -> Path:
    path.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (path / "f.txt").write_text("x")
    git("add", ".")
    git("commit", "-qm", "init")
    return path


def test_make_run_id() -> None:
    now = datetime(2026, 10, 26, 13, 5, 9, tzinfo=UTC)
    assert make_run_id("grpo_dev", 3, now) == "20261026-130509-grpo_dev-s3"


def test_create_run_records_provenance(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    run_dir = create_run(EXAMPLE, seed=2, runs_root=tmp_path / "runs", repo_root=repo)
    info = read_run_info(run_dir)
    assert info.seed == 2 and info.run_id.endswith("grpo_m1_dev_qwen35-2b-s2")
    assert len(info.git_commit) == 40 and info.git_dirty is False
    assert (run_dir / "config.yaml").read_bytes() == EXAMPLE.read_bytes()
    assert (run_dir / "checkpoints").is_dir()


def test_create_run_flags_dirty_tree(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    (repo / "f.txt").write_text("changed")
    run_dir = create_run(EXAMPLE, seed=1, runs_root=tmp_path / "runs", repo_root=repo)
    assert read_run_info(run_dir).git_dirty is True


def test_invalid_config_creates_nothing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    shutil.copyfile(EXAMPLE, bad)
    bad.write_text(bad.read_text() + "typo_key: 1\n")
    with pytest.raises(ValueError):
        create_run(bad, seed=1, runs_root=tmp_path / "runs", repo_root=git_repo(tmp_path / "r"))
    assert not (tmp_path / "runs").exists()


def test_run_card_includes_accounting(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    run_dir = create_run(EXAMPLE, seed=1, runs_root=tmp_path / "runs", repo_root=repo)
    efficiency = RunEfficiency(rollout_tokens=2_000_000, newly_solved_heldout=5)
    (run_dir / "efficiency.json").write_text(efficiency.model_dump_json())
    card = write_run_card(run_dir).read_text()
    assert "grpo_m1_dev_qwen35-2b" in card
    assert "excluded from conclusions" in card
    assert "per million rollout tokens: 2.500" in card
