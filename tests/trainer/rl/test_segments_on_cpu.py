import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from soe.trainer.main_rl import latest_checkpoint_step, segment_overrides
from soe.trainer.rl.segments import SegmentEvent, append_event, merged_metrics, training_hours

T0 = datetime(2026, 10, 29, 0, 0, tzinfo=UTC)


def event(kind: str, segment: str, hours: float, **extra: object) -> SegmentEvent:
    return SegmentEvent(
        event=kind,  # type: ignore[arg-type]
        segment=segment,
        at=T0 + timedelta(hours=hours),
        node="fc096",
        hardware="2 NVIDIA A40, 46068 MiB",
        git_commit="abc",
        **extra,  # type: ignore[arg-type]
    )


def write_metrics(path: Path, records: list[tuple[int, dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"step": s, "data": d}) + "\n" for s, d in records))


def test_a_replayed_step_replaces_the_killed_jobs_record(tmp_path: Path) -> None:
    write_metrics(
        tmp_path / "segment-100.jsonl",
        [
            (0, {"val": 0.1}),
            (1, {"r": 1.0}),
            (6, {"r": 6.0, "val": 0.6}),
            (7, {"r": 7.0}),
            (8, {"r": 8.0}),
        ],
    )
    write_metrics(
        tmp_path / "segment-101.jsonl", [(7, {"r": 7.5}), (8, {"r": 8.5}), (9, {"r": 9.5})]
    )
    merged = merged_metrics(tmp_path)
    assert list(merged) == [0, 1, 6, 7, 8, 9]
    assert merged[0] == {"val": 0.1}  # validation before training, then step records combine
    assert merged[6] == {"r": 6.0, "val": 0.6}  # the resumed job did not re-validate step 6
    assert (merged[7], merged[8]) == ({"r": 7.5}, {"r": 8.5})


def test_training_hours_sum_finished_killed_and_current_segments(tmp_path: Path) -> None:
    append_event(tmp_path, event("start", "100", 0.0, resumed_from_step=0))
    # job 100 was killed: no end event; it ends at its last rollout-token record (2 h)
    tokens = tmp_path / "rollout_tokens" / "segment-100.jsonl"
    tokens.parent.mkdir()
    tokens.write_text("{}\n")
    stamp = (T0 + timedelta(hours=2)).timestamp()
    os.utime(tokens, (stamp, stamp))
    append_event(tmp_path, event("start", "101", 3.0, resumed_from_step=6))
    assert training_hours(tmp_path, T0 + timedelta(hours=4)) == pytest.approx(3.0)


def test_this_jobs_overrides_replace_frozen_entries(tmp_path: Path) -> None:
    frozen = ["a=1", "trainer.val_before_train=true", "trainer.total_training_steps=null"]
    assert segment_overrides(frozen, 0, None) == frozen
    assert segment_overrides(frozen, 6, 40) == [
        "a=1",
        "trainer.val_before_train=false",
        "trainer.total_training_steps=40",
    ]
    with pytest.raises(ValueError, match="not in the frozen list"):
        segment_overrides(["a=1"], 6, None)
    assert latest_checkpoint_step(tmp_path) == 0
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6")
    assert latest_checkpoint_step(tmp_path) == 6
