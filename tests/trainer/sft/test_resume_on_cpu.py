"""SFT resume bookkeeping: the newest complete checkpoint, and the example order across segments."""

import json
from pathlib import Path

from soe.trainer.sft.resume import (
    STATE_FILE,
    continuity,
    example_key,
    last_complete_checkpoint,
    log_path,
)


def test_the_newest_checkpoint_with_its_trainer_state_is_chosen(tmp_path: Path) -> None:
    assert last_complete_checkpoint(tmp_path / "trainer") is None
    for step, complete in [(6, True), (12, True), (18, False)]:
        path = tmp_path / "trainer" / f"checkpoint-{step}"
        path.mkdir(parents=True)
        if complete:
            (path / STATE_FILE).write_text("{}")
    (tmp_path / "trainer" / "runs").mkdir()
    found = last_complete_checkpoint(tmp_path / "trainer")
    assert found is not None and found.step == 12 and found.path.name == "checkpoint-12"


def log(run_dir: Path, segment: str, rank: int, rows: list[tuple[int, list[int]]]) -> None:
    path = log_path(run_dir, segment, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for step, examples in rows:
            f.write(json.dumps({"step": step, "examples": examples}) + "\n")


def test_redone_steps_must_repeat_the_same_examples(tmp_path: Path) -> None:
    # Job 1 saved at step 2 and was killed during step 4 after one of its two micro-batches.
    log(tmp_path, "101", 0, [(1, [5]), (1, [9]), (2, [3]), (2, [7]), (3, [1]), (3, [4]), (4, [8])])
    log(tmp_path, "102", 0, [(3, [1]), (3, [4]), (4, [8]), (4, [2]), (5, [6]), (5, [0])])
    result = continuity(tmp_path)
    assert result.redone_steps == [3, 4] and result.mismatched_steps == []
    log(tmp_path, "102", 1, [(3, [11])])
    log(tmp_path, "101", 1, [(3, [12])])
    assert continuity(tmp_path).mismatched_steps == [3]


def test_example_keys_are_stable_and_distinct() -> None:
    assert example_key([1, 2, 3]) == example_key((1, 2, 3))
    assert example_key([1, 2, 3]) != example_key([1, 2, 4])
    assert example_key([1, 2]) != example_key([1, 2, 0])
