import json
from pathlib import Path

import pytest

from soe.trainer.config.eval_suite import file_sha256
from soe.trainer.main_rl import checked_data_file, frozen_overrides, train_to_stop


def test_data_file_must_match_its_hash(tmp_path: Path) -> None:
    (tmp_path / "train.parquet").write_bytes(b"rows")
    good = file_sha256(tmp_path / "train.parquet")
    assert checked_data_file(tmp_path, "train.parquet", good) == tmp_path / "train.parquet"
    with pytest.raises(ValueError, match="sha256"):
        checked_data_file(tmp_path, "train.parquet", "0" * 64)
    with pytest.raises(ValueError, match="missing"):
        checked_data_file(tmp_path, "nope.parquet", good)


def test_resumed_job_must_rebuild_the_same_overrides(tmp_path: Path) -> None:
    first = ["a=1", "b=2"]
    assert frozen_overrides(tmp_path, first) == first
    assert json.loads((tmp_path / "verl_overrides.json").read_text()) == first
    assert frozen_overrides(tmp_path, list(first)) == first
    with pytest.raises(ValueError, match="differ"):
        frozen_overrides(tmp_path, ["a=1", "b=3"])


FROZEN = ["trainer.val_before_train=true", "trainer.total_training_steps=null", "x=1"]


class FakeVerl:
    """Stands in for one verl launch: trains to its total, saves the last step, and leaves the
    stop estimate it would have written at that save (`estimates`: save step -> stop)."""

    def __init__(self, run_dir: Path, estimates: dict[int, int]) -> None:
        self.run_dir = run_dir
        self.estimates = estimates
        self.launches: list[list[str]] = []
        (run_dir / "checkpoints").mkdir(parents=True)

    def __call__(self, overrides: list[str]) -> None:
        self.launches.append(overrides)
        total = dict(o.split("=", 1) for o in overrides)["trainer.total_training_steps"]
        last = int(total) if total != "null" else max(self.estimates)
        (self.run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(f"{last}")
        stop = self.estimates.get(last, last)
        (self.run_dir / "stop_step.json").write_text(json.dumps({"stop_step": stop}))


def test_a_run_that_reaches_its_stop_launches_verl_once(tmp_path: Path) -> None:
    verl = FakeVerl(tmp_path, {14: 14})
    train_to_stop(FROZEN, tmp_path, verl)
    assert len(verl.launches) == 1
    assert verl.launches[0] == FROZEN  # fresh run: validation before training, verl's own total


def test_a_stop_raised_at_the_last_step_relaunches_verl_from_that_save(tmp_path: Path) -> None:
    # M1.11 B-L: verl had marked step 12 as last; its save re-estimated the stop at 13.
    (tmp_path / "stop_step.json").write_text(json.dumps({"stop_step": 12}))
    verl = FakeVerl(tmp_path, {12: 13, 13: 13})
    (tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("6")
    train_to_stop(FROZEN, tmp_path, verl)
    assert len(verl.launches) == 2
    assert "trainer.total_training_steps=12" in verl.launches[0]
    second = verl.launches[1]
    assert "trainer.total_training_steps=13" in second
    assert "trainer.val_before_train=false" in second  # step 12 was validated when saved


def test_a_launch_without_progress_is_an_error(tmp_path: Path) -> None:
    def stuck(overrides: list[str]) -> None:
        (tmp_path / "stop_step.json").write_text(json.dumps({"stop_step": 9}))

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("6")
    with pytest.raises(RuntimeError, match="without progress"):
        train_to_stop(FROZEN, tmp_path, stuck)
