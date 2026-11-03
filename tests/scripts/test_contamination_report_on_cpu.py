"""contamination_report lists pool/reference pairs; `--references rl` checks against the RL pool."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from soe.utils.dataset.problems import Level, Problem
from soe.utils.jsonl import write_jsonl

SHARED = (
    "a triangle has sides of length 3 and 4 and 5 units so find the radius of the circle around it"
)


def script() -> ModuleType:
    path = Path("scripts/contamination_report.py")
    spec = importlib.util.spec_from_file_location("contamination_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem(pid: str, question: str, answer: str = "1") -> Problem:
    return Problem(problem_id=pid, source="x", level=Level.TRAIN, question=question, answer=answer)


def test_rl_references_flag_copies_of_rl_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = script()
    rl = tmp_path / "datasets/soe/v3.1/rl"
    rl.mkdir(parents=True)
    write_jsonl(rl / "train.jsonl", [problem("dapo/t1", SHARED + " circumscribed", "5/2")])
    write_jsonl(rl / "val.jsonl", [problem("dapo/v1", "an unrelated question about primes")])
    pool = tmp_path / "pool.jsonl"
    write_jsonl(
        pool,
        [
            problem("deepmath/copy", SHARED + " circumscribed", "\\frac{5}{2}"),
            problem("deepmath/other", "compute the derivative of x squared at two"),
        ],
    )
    out = tmp_path / "report.json"
    monkeypatch.setenv("SOE_DATA", str(tmp_path))
    argv = ["x", "--train", "pool.jsonl", "--references", "rl", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()
    report = json.loads(out.read_text())
    assert set(report["references"]) == {"rl_train", "rl_val"}
    assert [(p["reference"], p["train"]) for p in report["pairs"]] == [("dapo/t1", "deepmath/copy")]
    pair = report["pairs"][0]
    assert pair["level"] == "rl_train" and pair["category"] == "E" and pair["same_answer"]
    assert report["residual_at_frozen_rule"]["single_problem"] == ["dapo/t1"]


def test_min_share_lists_weaker_matches() -> None:
    m = script()
    words = " ".join(f"w{i}" for i in range(30))
    reference = problem("c/1", words)
    weak = problem("d/1", " ".join(f"w{i}" for i in range(11)) + " other words entirely here")
    assert m.shares([reference], [weak], 8, frozenset()) == {}
    found = m.shares([reference], [weak], 8, frozenset(), min_share=0.1)
    assert list(found) == [("c/1", "d/1")] and 0.1 <= found[("c/1", "d/1")] < 0.2
