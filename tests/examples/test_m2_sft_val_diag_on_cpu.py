"""M2.11 diagnostic on the SFT validation problems: the suite and the problem selection."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.sft.data import SftExample
from soe.utils.dataset.problems import Level, Problem


def load() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_val_problems.py")
    spec = importlib.util.spec_from_file_location("m2_sft_val_problems", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step = load()


def problem(pid: str) -> Problem:
    return Problem(problem_id=pid, source="deepmath", level=Level.TRAIN, question="q", answer="1")


def example(pid: str) -> SftExample:
    return SftExample(problem_id=pid, user="u", completion="c</think>\\boxed{1}")


def test_validation_problems_once_each_sorted() -> None:
    problems = [problem(p) for p in ("c", "a", "b", "d")]
    chosen = step.val_problems(problems, [example("c"), example("a"), example("c")])
    assert [p.problem_id for p in chosen] == ["a", "c"]


def test_a_missing_validation_problem_is_refused() -> None:
    with pytest.raises(SystemExit, match="1 validation problems"):
        step.val_problems([problem("a")], [example("a"), example("z")])


def test_suite_is_soe_eval_v4_with_four_samples() -> None:
    suite = load_suite(Path("examples/eval/m2_sft_val_diag_suite.yaml"))
    v4 = load_suite(Path("soe/trainer/config/eval_suite.yaml"))
    for part in ("sampling", "prompt", "scoring", "hard"):
        assert getattr(suite, part) == getattr(v4, part)
    assert suite.main.samples_per_problem == 4 and suite.hard_subset is None
    (sft_val,) = suite.problem_sets
    assert sft_val.path == "datasets/soe/v1/sft/val_problems.jsonl"
