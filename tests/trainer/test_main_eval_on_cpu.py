"""main_eval on a tiny suite: units, sampling calls, the plan guard and the report stage."""

import json
from pathlib import Path

import pytest
import yaml

from soe.trainer import main_eval
from soe.trainer.config.eval_suite import EvalSuite, file_sha256, load_suite
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.shards import mark_done, shard_path

SUITE = Path("soe/trainer/config/eval_suite.yaml")
CHECKPOINT = "Qwen/Qwen3.5-2B-Base"  # staged Hub id, so its revision comes from stage_assets.yaml


def tiny_suite(tmp_path: Path) -> tuple[Path, EvalSuite]:
    """Sets "a" (Level A, 3 problems; a/0 is in the hard subset) and "b" (Level B, 2 problems)."""
    sets = {"a": (Level.A, 3), "b": (Level.B, 2)}
    entries = []
    for name, (level, size) in sets.items():
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(
            path,
            [
                Problem(
                    problem_id=f"{name}/{i}", source=name, level=level, question="q", answer="1"
                )
                for i in range(size)
            ],
        )
        entries.append(
            {"name": name, "level": level.value, "path": path.name, "sha256": file_sha256(path)}
        )
    hard = tmp_path / "hard.jsonl"
    write_jsonl(hard, [read_jsonl(tmp_path / "a.jsonl", Problem)[0]])
    data = yaml.safe_load(SUITE.read_text())
    data["problem_sets"] = entries
    data["hard_subset"] = {
        "name": "hard",
        "level": "A",
        "path": hard.name,
        "sha256": file_sha256(hard),
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(data))
    return suite_path, load_suite(suite_path)


def test_units_give_the_hard_subset_more_samples(tmp_path: Path) -> None:
    _, suite = tiny_suite(tmp_path)
    units = main_eval.eval_units(suite, tmp_path, limit_per_set=None)
    assert [u.problem.problem_id for u in units] == ["a/0", "a/1", "a/2", "b/0", "b/1"]
    assert [u.n for u in units] == [64, 16, 16, 16, 16] and units[0].hard
    assert [len(c) for c in main_eval.calls(units)] == [4, 1]  # one n per vLLM call
    limited = main_eval.eval_units(suite, tmp_path, limit_per_set=1)
    assert [u.problem.problem_id for u in limited] == ["a/0", "b/0"]


def test_plan_guard_refuses_a_different_evaluation(tmp_path: Path) -> None:
    suite_path, suite = tiny_suite(tmp_path)
    units = main_eval.eval_units(suite, tmp_path, None)
    plan = main_eval.make_plan(
        suite_path, suite, CHECKPOINT, units, num_shards=2, limit_per_set=None
    )
    assert plan.samples == 64 + 4 * 16 and plan.revision is not None
    main_eval.write_or_check_plan(tmp_path / "out", plan)
    main_eval.write_or_check_plan(tmp_path / "out", plan)  # the same plan is fine
    with pytest.raises(SystemExit, match="different evaluation"):
        main_eval.write_or_check_plan(tmp_path / "out", plan.model_copy(update={"num_shards": 3}))


def test_report_stage_builds_tables_from_shards(tmp_path: Path) -> None:
    suite_path, suite = tiny_suite(tmp_path)
    units = main_eval.eval_units(suite, tmp_path, None)
    out = tmp_path / "out"
    plan = main_eval.make_plan(
        suite_path, suite, CHECKPOINT, units, num_shards=2, limit_per_set=None
    )
    main_eval.write_or_check_plan(out, plan)
    cli = main_eval.EvaluateCli(
        suite=suite_path, checkpoint=CHECKPOINT, output=out, stage="report", num_shards=2
    )
    for k in range(2):
        records = [
            SampleRecord(
                problem_id=u.problem.problem_id,
                sample_index=i,
                checkpoint=CHECKPOINT,
                completion="",
                extracted_answer="1" if i % 2 == 0 else "2",
                correct=i % 2 == 0,
                completion_tokens=5,
                finish_reason="stop",
            )
            for u in units[k::2]
            for i in range(u.n)
        ]
        write_jsonl(shard_path(out, k, 2), records)
    mark_done(out, 0)
    assert main_eval.run_report(cli, suite, units, plan) == 1  # shard 1 not finished
    mark_done(out, 1)
    assert main_eval.run_report(cli, suite, units, plan) == 0
    report = json.loads((out / "report.json").read_text())
    assert set(report["main"]["per_set"]) == {"a", "b"}
    assert report["main"]["per_level"]["A"]["problems"] == 3
    assert (
        report["main"]["per_set"]["a"]["pass@1"] == 0.5
        and report["main"]["per_set"]["a"]["pass@16"] == 1.0
    )
    assert report["main"]["per_set"]["b"]["maj@16"] == 1.0  # 8-8 tie: the first group, "1", wins
    assert report["hard"]["all"]["problems"] == 1 and report["hard"]["all"]["pass@64"] == 1.0
    per_problem = [
        json.loads(line) for line in (out / "per_problem.jsonl").read_text().splitlines()
    ]
    assert [p["all"]["n"] for p in per_problem] == [64, 16, 16, 16, 16]
