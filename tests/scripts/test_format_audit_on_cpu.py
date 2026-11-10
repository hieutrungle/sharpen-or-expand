"""scripts/format_audit.py and scripts/rollout_groups.py on synthetic samples (M2.11 audits)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(f"scripts/{name}.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fa = load("format_audit")
rg = load("rollout_groups")

GOOD = "think about it</think>\n\nSo \\(x=2\\).\n\n\\boxed{\\frac{1}{2}}"


def test_boxed_contents_match_nested_braces() -> None:
    assert [b for _, b in fa.boxed_contents("a \\boxed{\\frac{1}{2}} b \\boxed{3}")] == [
        "\\frac{1}{2}",
        "3",
    ]
    assert fa.boxed_contents("no box") == []


def test_audit_counts_each_format_part() -> None:
    samples = [
        fa.Sample(completion=GOOD, stopped=True),
        fa.Sample(completion="a</think>b</think>\\boxed{1}", stopped=True),  # two closes
        fa.Sample(completion="x</think>\\boxed{1} or \\boxed{2}", stopped=True),
        fa.Sample(completion="still thinking " * 400, stopped=False),  # loop, cut in thinking
        fa.Sample(completion="t</think>answer\nuser\nnext question", stopped=False),
    ]
    r = fa.audit(samples)
    assert r["samples"] == 5 and r["stop_rate"] == pytest.approx(0.6)
    assert r["when_stopped"]["one_close"] == pytest.approx(2 / 3)
    assert r["when_one_close"] == {"box_ge1": 1.0, "box_eq1": 0.5, "boxes_distinct_gt1": 0.5}
    assert r["chars_after_last_box_p50_p90"] == [0, 0]
    assert r["when_truncated"] == {"cut_in_thinking": 0.5, "cut_in_answer": 0.5, "cut_loop": 0.5}
    assert r["all"]["leaked_turn"] == pytest.approx(0.2)


def test_demos_count_as_stopped(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"problem_id": "p", "user": "q", "completion": GOOD}) + "\n")
    assert fa.demo_samples(path) == [fa.Sample(completion=GOOD, stopped=True)]


def rollout(step: int, pid: str, score: float, truncated: bool = False) -> dict[str, object]:
    return dict(
        step=step,
        problem_id=pid,
        score=score,
        truncated=truncated,
        has_box=not truncated,
        turn_end=not truncated,
        answer="1",
    )


def test_rollout_groups_split_signal_from_zero_variance(tmp_path: Path) -> None:
    rows = (
        [rollout(1, "a", 1.0)] * 2  # all correct
        + [rollout(1, "b", 0.0, truncated=True)] * 2  # all wrong, truncated
        + [rollout(1, "c", 1.0), rollout(1, "c", 0.0)]  # mixed
        + [rollout(2, "a", 0.0)] * 2  # same problem, new step: a new group
    )
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rollouts = rg.read_rollouts(path)
    s1 = rg.summary([r for r in rollouts if r.step == 1])
    assert (s1["groups"], s1["group_sizes"]) == (3, [2])
    assert s1["groups_all_correct"] == s1["groups_all_wrong"] == s1["groups_mixed"] == 1 / 3
    assert s1["truncated_in_all_wrong_groups"] == 1.0
    assert s1["truncated_in_mixed_groups"] == 0.0
    assert s1["im_end_when_stopped"] == 1.0
    assert rg.summary(rollouts)["groups"] == 4
