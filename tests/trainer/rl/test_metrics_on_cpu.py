import json
from pathlib import Path

import pytest

from soe.trainer.rl.metrics import (
    OUTCOMES_ENV,
    SOE_KEYS,
    RolloutOutcome,
    distinct_answers,
    outcomes_by_step,
    read_step_metrics,
    record_outcome,
    step_metrics,
    write_step_metrics,
)
from soe.utils.metric.pass_at_k import same_string


def outcome(problem: str, score: float, answer: str | None = "1", **extra: bool) -> RolloutOutcome:
    fields = {"truncated": False, "has_box": answer is not None, "turn_end": True} | extra
    return RolloutOutcome(step=1, problem_id=problem, score=score, answer=answer, **fields)


def test_step_metrics_of_two_groups() -> None:
    outcomes = [
        # p1: two right, one wrong answer, one truncated (score 0, no final answer)
        outcome("p1", 1.0, "1"),
        outcome("p1", 1.0, " 1"),
        outcome("p1", 0.0, "2", turn_end=False),
        outcome("p1", 0.0, "9", truncated=True),
        # p2: all wrong; one without an answer
        outcome("p2", 0.0, "3"),
        outcome("p2", 0.0, "4"),
        outcome("p2", 0.0, "4"),
        outcome("p2", 0.0, None),
    ]
    lengths = [100, 200, 300, 16384, 50, 60, 70, 80]
    m = step_metrics(outcomes, lengths, prompts=2, group_size=4, same=same_string)
    assert set(m) == set(SOE_KEYS)
    assert m["soe/reward/mean"] == 0.25
    assert m["soe/group/reward_var_mean"] == pytest.approx((0.25 + 0.0) / 2)
    assert m["soe/group/zero_var_frac"] == 0.5  # p2
    assert (m["soe/group/all_correct_frac"], m["soe/group/all_wrong_frac"]) == (0.0, 0.5)
    assert m["soe/rollout/truncated_frac"] == 1 / 8
    assert m["soe/rollout/boxed_frac"] == 7 / 8
    assert m["soe/rollout/stop_turn_end_frac"] == 6 / 8
    assert m["soe/rollout/stop_other_frac"] == 1 / 8
    # numpy.percentile(lengths, 50) and (lengths, 95)
    assert (m["soe/length/p50"], m["soe/length/p95"]) == pytest.approx((90.0, 10754.6))
    assert m["soe/length/max"] == 16384.0
    # p1: {1, 2} (the truncated "9" is not a final answer); p2: {3, 4}
    assert m["soe/answers/distinct_per_group_mean"] == 2.0
    assert m["soe/answers/none_frac"] == 1 / 7


def test_step_metrics_refuse_a_lost_rollout() -> None:
    full = [outcome("p1", 1.0)] * 4 + [outcome("p2", 0.0)] * 4
    with pytest.raises(ValueError, match="groups of 4"):
        step_metrics(full[:-1], [1] * 7, prompts=2, group_size=4, same=same_string)
    with pytest.raises(ValueError, match="length records"):
        step_metrics(full, [1] * 7, prompts=2, group_size=4, same=same_string)
    with pytest.raises(ValueError, match="expected 3 groups"):
        step_metrics(full, [1] * 8, prompts=3, group_size=4, same=same_string)


def test_distinct_answers_groups_like_maj_at_k() -> None:
    def same(a: str, b: str) -> bool:
        return a.strip("0.") == b.strip("0.")

    assert distinct_answers(["1", "1.0", "2", "01"], same) == 2
    assert distinct_answers([], same) == 0


def test_outcomes_are_appended_and_a_replayed_step_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "segment-100.jsonl", tmp_path / "segment-101.jsonl"
    monkeypatch.setenv(OUTCOMES_ENV, str(first))
    record_outcome(outcome("p1", 1.0))
    record_outcome(outcome("p1", 0.0).model_copy(update={"step": 2}))
    monkeypatch.setenv(OUTCOMES_ENV, str(second))
    record_outcome(outcome("p9", 1.0).model_copy(update={"step": 2}))  # the resumed job
    by_step = outcomes_by_step(tmp_path)
    assert [o.problem_id for o in by_step[1]] == ["p1"]
    assert [o.problem_id for o in by_step[2]] == ["p9"]


def test_step_metrics_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rollout_metrics.jsonl"
    write_step_metrics(path, {2: {"a": 0.5}, 1: {"a": 1.0}})
    assert [json.loads(x)["step"] for x in path.read_text().splitlines()] == [1, 2]
    assert read_step_metrics(path) == {1: {"a": 1.0}, 2: {"a": 0.5}}
    assert read_step_metrics(tmp_path / "missing.jsonl") == {}
