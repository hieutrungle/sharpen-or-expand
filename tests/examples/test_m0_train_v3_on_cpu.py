"""Train v3 drops exactly the reviewed copies and refuses ids it cannot find."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from soe.utils.dataset.problems import Level, Problem


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m0_train_v3.py")
    spec = importlib.util.spec_from_file_location("m0_train_v3", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem(pid: str) -> Problem:
    return Problem(problem_id=pid, source="dapo", level=Level.TRAIN, question="q", answer="1")


def test_a_train_problem_goes_if_any_of_its_pairs_says_drop() -> None:
    m = script()
    review = {
        "pairs": [
            {"train": "t/1", "verdict": "keep"},
            {"train": "t/1", "verdict": "drop"},  # the same problem as another reference
            {"train": "t/2", "verdict": "keep"},
        ]
    }
    drops = m.reviewed_drops(review)
    kept, dropped = m.split([problem("t/1"), problem("t/2"), problem("t/3")], drops)
    assert [p.problem_id for p in kept] == ["t/2", "t/3"] and [p.problem_id for p in dropped] == [
        "t/1"
    ]
    with pytest.raises(ValueError, match="not in the train pool"):
        m.split([problem("t/2")], drops)
    with pytest.raises(ValueError, match="unknown verdicts"):
        m.reviewed_drops({"pairs": [{"train": "t/1", "verdict": "maybe"}]})


def test_committed_review_drops_108_train_problems() -> None:
    import json

    review = json.loads(Path("docs/reports/m0_train_v3_review.json").read_text())
    assert len(review["pairs"]) == 121 and len(script().reviewed_drops(review)) == 108
