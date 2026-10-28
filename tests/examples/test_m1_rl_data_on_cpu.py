"""RL data v3.1: the validation split leaks nothing into train and drops nothing else; verl rows
carry the frozen prompt and the gold answer."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Level, Problem

REPORT = Path("docs/reports/m1_datasets_v3.1.json")


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m1_rl_data.py")
    spec = importlib.util.spec_from_file_location("m1_rl_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem(pid: str, question: str) -> Problem:
    return Problem(problem_id=pid, source="dapo", level=Level.TRAIN, question=question, answer="7")


WORDS = [f"w{i}" for i in range(40)]


def test_shared_fraction_counts_ngrams_found_in_other_problems() -> None:
    m = script()
    original = problem("p/1", " ".join(WORDS[:20]))
    copy = problem("p/2", " ".join(WORDS[:20]) + " extra")  # same text plus one word
    unique = problem("p/3", " ".join(WORDS[20:]))
    fractions = m.shared_fraction([original, copy, unique], n=13, ignore=frozenset())
    assert fractions["p/1"] == 1.0 and fractions["p/3"] == 0.0
    assert fractions["p/2"] == pytest.approx(8 / 9)  # 8 of its 9 13-grams are in p/1


def test_carve_validation_takes_only_uncovered_problems_and_keeps_the_rest() -> None:
    m = script()
    problems = [problem(f"p/{i}", f"q{i}") for i in range(10)]
    fractions = {p.problem_id: (0.9 if i < 4 else 0.1) for i, p in enumerate(problems)}
    train, val, eligible = m.carve_validation(problems, fractions, size=3, seed=1, min_fraction=0.5)
    assert eligible == 6 and len(val) == 3 and len(train) == 7
    assert all(fractions[p.problem_id] < 0.5 for p in val)
    assert [p.problem_id for p in train] == [p.problem_id for p in problems if p not in val]
    again = m.carve_validation(problems, fractions, size=3, seed=1, min_fraction=0.5)
    assert again[1] == val
    with pytest.raises(ValueError, match="eligible"):
        m.carve_validation(problems, fractions, size=7, seed=1, min_fraction=0.5)


def test_unchecked_problems_never_enter_validation() -> None:
    m = script()
    problems = [problem("p/1", "a"), problem("p/2", "b")]
    with pytest.raises(ValueError, match="only 1 eligible"):
        m.carve_validation(problems, {"p/1": None, "p/2": 0.0}, size=2, seed=0, min_fraction=0.5)


def test_verl_row_holds_the_frozen_prompt_and_gold() -> None:
    m = script()
    prompt = load_suite(Path("soe/trainer/config/eval_suite.yaml")).prompt
    row = m.verl_row(problem("dapo/x", "What is 3+4?"), prompt, split="val", index=5)
    assert row["prompt"] == [{"role": "user", "content": prompt.render("What is 3+4?")}]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "7"}
    assert row["extra_info"] == {"index": 5, "problem_id": "dapo/x", "split": "val"}
    assert row["data_source"] == "soe/dapo"


def test_parquet_round_trip_and_stable_hashes(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    m = script()
    prompt = load_suite(Path("soe/trainer/config/eval_suite.yaml")).prompt
    rows = [
        m.verl_row(problem(f"p/{i}", f"q {i}"), prompt, split="train", index=i) for i in range(3)
    ]
    m.write_parquet(rows, tmp_path / "a.parquet")
    m.write_parquet(rows, tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()
    assert pq.read_table(tmp_path / "a.parquet").to_pylist() == rows
    assert m.rows_sha256(rows) == m.rows_sha256(json.loads(json.dumps(rows)))


def test_committed_report_partitions_train_v3() -> None:
    report = json.loads(REPORT.read_text())
    v3 = json.loads(Path("docs/reports/m0_datasets_v3.json").read_text())
    assert report["derived_from"]["sha256"] == v3["sha256"]["rl/train"]
    train, val = report["files"]["train"]["problems"], report["files"]["val"]["problems"]
    assert (train, val) == (11868, 256) and train + val == v3["counts"]["rl/train"]
    assert len(set(report["validation_ids"])) == 256
    assert report["max_shared_fraction_in_val"] < report["near_duplicate"]["min_fraction"]


def test_dev_config_points_at_the_built_files() -> None:
    import yaml

    report = json.loads(REPORT.read_text())
    data = yaml.safe_load(Path("examples/grpo_trainer/m1_dev_qwen35-2b.yaml").read_text())["data"]
    for split in ("train", "val"):
        assert data[f"{split}_file"] == report["files"][split]["parquet"]
        assert data[f"{split}_sha256"] == report["files"][split]["parquet_sha256"]
