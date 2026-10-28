"""compare_eval pairs problems and reports the mean difference with its standard error."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from soe.utils.dataset.problems import Level
from soe.utils.metric.eval_tables import Counts, ProblemResult


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_eval", Path("scripts/compare_eval.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(pid: str, correct: int) -> ProblemResult:
    counts = Counts(
        n=16,
        correct=correct,
        truncated=16 - correct,
        boxed=correct,
        stopped=correct,
        boxed_when_stopped=correct,
        completion_tokens=16 * 100,
    )
    return ProblemResult(
        problem_id=pid,
        problem_set="s",
        level=Level.A,
        hard=False,
        main=counts,
        all=counts,
        maj={},
    )


def test_paired_difference_and_standard_error() -> None:
    m = script()
    out = m.paired([0.5, 0.5, 0.5], [0.5, 0.75, 1.0])
    assert out["diff"] == pytest.approx(0.25) and out["se"] == pytest.approx(0.25 / 3**0.5)


def test_compare_uses_common_problems_per_level() -> None:
    m = script()
    a = {"x": result("x", 8), "y": result("y", 0), "z": result("z", 16)}
    b = {"x": result("x", 12), "y": result("y", 4)}
    table = m.compare(a, b, main_k=16, hard_k=64)
    row = table["level A"]["pass@1"]
    assert row["problems"] == 2 and row["diff"] == pytest.approx(0.25)
    assert "hard tier" not in table
