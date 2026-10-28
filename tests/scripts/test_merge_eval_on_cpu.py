"""merge_eval keeps results only for problems that are identical in both suite versions."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from soe.trainer.config.eval_suite import EvalSuite, file_sha256
from soe.utils.dataset.problems import Level, Problem
from soe.utils.jsonl import write_jsonl
from soe.utils.metric.eval_tables import Counts, ProblemResult

SUITE = Path("soe/trainer/config/eval_suite.yaml")


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("merge_eval", Path("scripts/merge_eval.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_with(tmp: Path, name: str, questions: list[str]) -> EvalSuite:
    path = tmp / f"{name}.jsonl"
    write_jsonl(
        path,
        [
            Problem(problem_id=f"b/{i}", source="dapo", level=Level.B, question=q, answer="1")
            for i, q in enumerate(questions)
        ],
    )
    data: dict[str, Any] = yaml.safe_load(SUITE.read_text())
    data |= {
        "name": name,
        "hard_subset": None,
        "excluded_problems": {},
        "problem_sets": [
            {"name": "level_b", "level": "B", "path": path.name, "sha256": file_sha256(path)}
        ],
    }
    return EvalSuite.model_validate(data)


def result(pid: str, correct: int) -> ProblemResult:
    counts = Counts(
        n=16,
        correct=correct,
        truncated=0,
        boxed=16,
        stopped=16,
        boxed_when_stopped=16,
        completion_tokens=160,
    )
    return ProblemResult(
        problem_id=pid,
        problem_set="level_b",
        level=Level.B,
        hard=False,
        main=counts,
        all=counts,
        maj={k: correct > 8 for k in (1, 4, 8, 16)},
    )


def test_unchanged_problems_reuse_old_results_and_changed_ones_take_the_patch(
    tmp_path: Path,
) -> None:
    m = script()
    old = suite_with(tmp_path, "old", ["same", "broken \x0crac{1}{2}"])
    new = suite_with(tmp_path, "new", ["same", "broken \\frac{1}{2}"])
    old_results = {"b/0": result("b/0", 16), "b/1": result("b/1", 0)}
    merged, counts = m.merge(old, new, old_results, {"b/1": result("b/1", 12)}, tmp_path)
    assert counts == {"reused": 1, "patched": 1, "excluded": 0}
    assert [(r.problem_id, r.main.correct) for r in merged] == [("b/0", 16), ("b/1", 12)]


def test_a_changed_problem_without_a_patch_is_refused(tmp_path: Path) -> None:
    m = script()
    old = suite_with(tmp_path, "old", ["same", "broken \x0crac{1}{2}"])
    new = suite_with(tmp_path, "new", ["same", "broken \\frac{1}{2}"])
    with pytest.raises(SystemExit, match="not re-evaluated"):
        m.merge(old, new, {"b/0": result("b/0", 16), "b/1": result("b/1", 0)}, {}, tmp_path)
