"""m2_sft_pool_clean drops every pool problem a contamination report lists, round by round."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from soe.utils.dataset.problems import Level, Problem


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_pool_clean.py")
    spec = importlib.util.spec_from_file_location("m2_sft_pool_clean", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_listed_problem_is_dropped() -> None:
    m = script()
    report = {"pairs": [{"reference": "b/1", "train": "d/1"}, {"reference": "b/2", "train": "d/3"}]}
    problems = [
        Problem(problem_id=f"d/{i}", source="d", level=Level.TRAIN, question="q", answer="1")
        for i in range(5)
    ]
    assert [p.problem_id for p in m.clean(problems, m.listed(report))] == ["d/0", "d/2", "d/4"]
    with pytest.raises(ValueError, match="not in the pool"):
        m.clean(problems, {"d/9"})


def test_committed_rounds_end_at_a_check_that_lists_nothing() -> None:
    config = script().CleanConfig.model_validate(
        yaml.safe_load(Path("examples/data_preprocess/m2_sft_pool_clean.yaml").read_text())
    )
    clean = json.loads(config.report.read_text())["clean"]
    finals = sorted(Path("docs/reports").glob("m2_contamination_sft*_clean.json"))
    assert len(finals) == 3  # Levels A/B/C, Level C at 10%, RL pool
    for path in finals:
        report = json.loads(path.read_text())
        assert report["train"]["sha256"] == clean["sha256"], path
        assert report["candidate_pairs"] == 0, path
        assert report["residual_at_frozen_rule"] == {"single_problem": [], "covered_by_union": []}
