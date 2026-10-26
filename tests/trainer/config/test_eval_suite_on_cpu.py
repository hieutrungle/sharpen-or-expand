import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from soe.trainer.config.eval_suite import EvalSuite, file_sha256, hash_mismatches, load_suite

SUITE = Path("soe/trainer/config/eval_suite.yaml")


def suite_dict() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(SUITE.read_text())
    return data


def test_frozen_suite_loads_with_decision_004_settings() -> None:
    suite = load_suite(SUITE)
    assert (suite.sampling.temperature, suite.sampling.top_p) == (0.6, 0.95)
    assert suite.sampling.max_tokens == 16384 and suite.sampling.top_k == -1
    assert suite.main.samples_per_problem == 16 and suite.hard.samples_per_problem == 64
    assert suite.scoring.maj_grouping == "verifier_equivalence"
    assert suite.scoring.transition_solved_threshold == 0.5


def test_frozen_suite_hashes_match_the_committed_data_build_report() -> None:
    suite = load_suite(SUITE)
    report = json.loads(Path(suite.data_build_report).read_text())
    for problem_set in suite.problem_sets:
        key = "rl/level_b" if problem_set.name == "level_b" else f"eval/{problem_set.name}"
        assert problem_set.sha256 == report["sha256"][key], problem_set.name
    covered = {p.name for p in suite.problem_sets}
    built = {k.split("/", 1)[1] for k in report["sha256"] if k.startswith("eval/")} | {"level_b"}
    assert covered == built  # every built evaluation file is in the suite, and nothing else


def test_prompt_renders_problem_and_boxed_instruction() -> None:
    prompt = load_suite(SUITE).prompt.render("What is 1+1?")
    assert prompt.startswith("What is 1+1?\n\n")
    assert prompt.endswith("put your final answer within \\boxed{}.")


def test_prompt_needs_exactly_one_placeholder() -> None:
    data = suite_dict()
    data["prompt"]["template"] = "no placeholder"
    with pytest.raises(ValidationError, match="exactly once"):
        EvalSuite.model_validate(data)


def test_k_spec_rules() -> None:
    data = suite_dict()
    data["main"]["ks"] = [4, 1]
    with pytest.raises(ValidationError, match="strictly increasing"):
        EvalSuite.model_validate(data)
    data = suite_dict()
    data["main"]["transition_ks"] = [32]
    with pytest.raises(ValidationError, match="subset"):
        EvalSuite.model_validate(data)
    data = suite_dict()
    data["hard"] = {"ks": [1, 4], "transition_ks": [1, 4]}
    with pytest.raises(ValidationError, match="at least as many samples"):
        EvalSuite.model_validate(data)


def test_unknown_scoring_rule_rejected() -> None:
    data = suite_dict()
    data["scoring"]["truncated_response"] = "ignored"
    with pytest.raises(ValidationError):
        EvalSuite.model_validate(data)


def test_hash_check_detects_drift(tmp_path: Path) -> None:
    problems = tmp_path / "toy.jsonl"
    problems.write_text('{"q": 1}\n')
    data = suite_dict()
    data["problem_sets"] = [
        {"name": "toy", "level": "A", "path": "toy.jsonl", "sha256": file_sha256(problems)}
    ]
    suite = EvalSuite.model_validate(data)
    assert hash_mismatches(suite, tmp_path) == []
    problems.write_text('{"q": 2}\n')
    assert hash_mismatches(suite, tmp_path) == ["toy"]
    problems.unlink()
    assert hash_mismatches(suite, tmp_path) == ["toy"]
