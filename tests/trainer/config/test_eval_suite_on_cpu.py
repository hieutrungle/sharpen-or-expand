from pathlib import Path

import pytest
import yaml

from soe.trainer.config.eval_suite import EvalSuite, file_sha256, hash_mismatches, load_suite


def write_suite(tmp_path: Path, problems: Path) -> Path:
    suite = {
        "name": "test-suite",
        "sampling": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096, "ks": [1, 4, 8, 16]},
        "problem_sets": [
            {"name": "toy", "level": "A", "path": problems.name, "sha256": file_sha256(problems)}
        ],
        "hard_subset": None,
        "hard_subset_sampling": None,
        "transition_solved_threshold": 0.5,
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return path


def test_suite_hash_detects_drift(tmp_path: Path) -> None:
    problems = tmp_path / "toy.jsonl"
    problems.write_text('{"q": 1}\n')
    suite = load_suite(write_suite(tmp_path, problems))
    assert suite.sampling.samples_per_problem == 16
    assert hash_mismatches(suite, tmp_path) == []
    problems.write_text('{"q": 2}\n')
    assert hash_mismatches(suite, tmp_path) == ["toy"]
    problems.unlink()
    assert hash_mismatches(suite, tmp_path) == ["toy"]


def test_suite_rejects_unsorted_ks(tmp_path: Path) -> None:
    problems = tmp_path / "toy.jsonl"
    problems.write_text("{}\n")
    data = yaml.safe_load(write_suite(tmp_path, problems).read_text())
    data["sampling"]["ks"] = [4, 1]
    with pytest.raises(ValueError):
        EvalSuite.model_validate(data)
