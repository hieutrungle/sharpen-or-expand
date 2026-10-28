import json
from pathlib import Path

import pytest

from soe.trainer.rl.rollout_tokens import TOKENS_ENV, cumulative_rollout_tokens, record_rollout


def write(path: Path, entries: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"step": s, "tokens": t}) + "\n" for s, t in entries))


def test_agent_loop_records_append_to_the_segment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "segment-7.jsonl"
    monkeypatch.setenv(TOKENS_ENV, str(path))
    record_rollout(step=1, tokens=10)
    record_rollout(step=1, tokens=16384)
    assert cumulative_rollout_tokens(tmp_path, through_step=1, responses_per_step=2) == 16394


def test_a_replayed_step_counts_once_from_the_latest_segment(tmp_path: Path) -> None:
    # Job 100 ran steps 1-3 and was killed during step 3; job 101 resumed from step 2's checkpoint
    # and ran step 3 again.
    write(tmp_path / "segment-100.jsonl", [(1, 5), (1, 5), (2, 7), (2, 7), (3, 1)])
    write(tmp_path / "segment-101.jsonl", [(3, 9), (3, 9)])
    assert cumulative_rollout_tokens(tmp_path, through_step=3, responses_per_step=2) == 10 + 14 + 18


def test_a_lost_rollout_or_step_fails_loudly(tmp_path: Path) -> None:
    write(tmp_path / "segment-1.jsonl", [(1, 5), (1, 5), (2, 7)])
    with pytest.raises(ValueError, match="step 2 has 1 rollouts"):
        cumulative_rollout_tokens(tmp_path, through_step=2, responses_per_step=2)
    write(tmp_path / "segment-1.jsonl", [(1, 5), (1, 5), (2, 7), (2, 7)])
    with pytest.raises(ValueError, match="no rollout-token records for step 3"):
        cumulative_rollout_tokens(tmp_path, through_step=3, responses_per_step=2)
