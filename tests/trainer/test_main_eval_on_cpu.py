"""main_eval on a tiny suite: units, sampling calls, the plan guard and the report stage."""

import json
from pathlib import Path
from typing import Any

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
    data["excluded_problems"] = {}
    data["removed_from_hard_subset"] = {}
    data["hard_subset"] = {
        "name": "hard",
        "path": hard.name,
        "sha256": file_sha256(hard),
        "selection_report": "unused",
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
                completion="\\boxed{1}" if i % 2 == 0 else "\\boxed{2}",
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


def test_plan_is_never_seen_half_written(tmp_path: Path) -> None:
    suite_path, suite = tiny_suite(tmp_path)
    units = main_eval.eval_units(suite, tmp_path, None)
    plan = main_eval.make_plan(
        suite_path, suite, CHECKPOINT, units, num_shards=2, limit_per_set=None
    )
    out = tmp_path / "out"
    main_eval.write_or_check_plan(out, plan)
    assert [p.name for p in out.iterdir()] == ["plan.json"]  # no private file left behind
    (out / "plan.json").write_text("")  # what a racing reader used to see
    with pytest.raises(ValueError):
        main_eval.write_or_check_plan(out, plan)


def test_only_problems_selects_listed_ids_and_rejects_unknown_ones(tmp_path: Path) -> None:
    _, suite = tiny_suite(tmp_path)
    units = main_eval.eval_units(suite, tmp_path, None, only=frozenset({"a/0", "b/1"}))
    assert [(u.problem.problem_id, u.n) for u in units] == [("a/0", 64), ("b/1", 16)]
    with pytest.raises(ValueError, match="not in the suite"):
        main_eval.eval_units(suite, tmp_path, None, only=frozenset({"a/0", "zzz/9"}))


def test_excluded_problems_are_skipped_and_leave_the_hard_tier(tmp_path: Path) -> None:
    _, suite = tiny_suite(tmp_path)
    excluded = suite.model_copy(
        update={"excluded_problems": {"a/0": "bad gold", "b/1": "ambiguous"}}
    )
    units = main_eval.eval_units(excluded, tmp_path, None)
    assert [u.problem.problem_id for u in units] == ["a/1", "a/2", "b/0"]  # a/0 was the hard one
    unknown = suite.model_copy(update={"excluded_problems": {"zzz/1": "typo"}})
    with pytest.raises(ValueError, match="excluded problems not in the suite"):
        main_eval.eval_units(unknown, tmp_path, None)


def test_removed_from_hard_subset_keeps_the_problem_at_main_tier(tmp_path: Path) -> None:
    _, suite = tiny_suite(tmp_path)
    removed = suite.model_copy(update={"removed_from_hard_subset": {"a/0": "verifier"}})
    units = main_eval.eval_units(removed, tmp_path, None)
    assert [(u.problem.problem_id, u.n, u.hard) for u in units][0] == ("a/0", 16, False)
    not_hard = suite.model_copy(update={"removed_from_hard_subset": {"a/1": "typo"}})
    with pytest.raises(ValueError, match="not in it"):
        main_eval.eval_units(not_hard, tmp_path, None)


def test_report_rescores_stored_verdicts_with_the_current_verifier() -> None:
    record = SampleRecord(
        problem_id="a/0",
        sample_index=0,
        checkpoint=CHECKPOINT,
        completion="so \\boxed{10\\%}",
        extracted_answer="10\\%",
        correct=False,  # as an older verifier judged it
        completion_tokens=5,
        finish_reason="stop",
    )
    assert main_eval.rescored(record, "10").correct
    truncated = record.model_copy(update={"finish_reason": "length"})
    assert not main_eval.rescored(truncated, "10").correct  # truncated stays incorrect
    unboxed = record.model_copy(update={"completion": "10", "correct": True})
    assert not main_eval.rescored(unboxed, "10").correct


def sampled_evaluation(tmp_path: Path) -> tuple[Path, EvalSuite, Path]:
    """The tiny suite fully sampled into tmp_path/old (2 shards, every sample correct)."""
    suite_path, suite = tiny_suite(tmp_path)
    units = main_eval.eval_units(suite, tmp_path, None)
    old = tmp_path / "old"
    plan = main_eval.make_plan(suite_path, suite, CHECKPOINT, units, 2, None)
    main_eval.write_or_check_plan(old, plan)
    for k in range(2):
        records = [
            SampleRecord(
                problem_id=u.problem.problem_id,
                sample_index=i,
                checkpoint=CHECKPOINT,
                completion="\\boxed{1}",
                extracted_answer="1",
                correct=True,
                completion_tokens=5,
                finish_reason="stop",
            )
            for u in units[k::2]
            for i in range(u.n)
        ]
        write_jsonl(shard_path(old, k, 2), records)
        mark_done(old, k)
    return suite_path, suite, old


def revised(tmp_path: Path, suite_path: Path, **changes: Any) -> tuple[Path, EvalSuite]:
    data = yaml.safe_load(suite_path.read_text()) | {"name": "tiny-v2"} | changes
    path = tmp_path / "suite_v2.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, load_suite(path)


def rescore_cli(new_path: Path, old_path: Path, old: Path, out: Path) -> main_eval.EvaluateCli:
    return main_eval.EvaluateCli(
        suite=new_path,
        checkpoint=CHECKPOINT,
        output=out,
        stage="report",
        rescore_from=old,
        rescore_suite=old_path,
    )


def test_rescore_drops_newly_excluded_samples_and_records_provenance(tmp_path: Path) -> None:
    old_path, _, old = sampled_evaluation(tmp_path)
    new_path, new = revised(tmp_path, old_path, excluded_problems={"b/1": "impossible gold"})
    units = main_eval.eval_units(new, tmp_path, None)
    out = tmp_path / "new"
    assert main_eval.run_rescore(rescore_cli(new_path, old_path, old, out), new, units) == 0
    report = json.loads((out / "report.json").read_text())
    source = report["plan"]["rescored_from"]
    assert source["dropped_problems"] == ["b/1"] and source["dropped_samples"] == 16
    assert source["suite_file_sha256"] == file_sha256(old_path)
    assert set(source["shard_files_sha256"]) == {"samples-0-of-2.jsonl", "samples-1-of-2.jsonl"}
    assert report["plan"]["suite_name"] == "tiny-v2" and report["plan"]["samples"] == 64 + 3 * 16
    assert report["main"]["per_level"]["B"]["problems"] == 1
    assert report["main"]["per_level"]["A"]["problems"] == 3  # untouched
    assert not list(out.glob("samples-*"))  # nothing is copied or re-sampled


def test_rescore_refuses_a_suite_that_changes_what_is_sampled(tmp_path: Path) -> None:
    old_path, old_suite, _ = sampled_evaluation(tmp_path)
    sampling = old_suite.sampling.model_dump() | {"temperature": 1.0}
    _, new = revised(tmp_path, old_path, sampling=sampling, excluded_problems={"b/1": "x"})
    with pytest.raises(ValueError, match=r"differ in \['sampling'\]"):
        main_eval.newly_excluded(old_suite, new)


def test_rescore_refuses_a_suite_that_drops_an_exclusion(tmp_path: Path) -> None:
    old_path, old_suite, _ = sampled_evaluation(tmp_path)
    earlier = old_suite.model_copy(update={"excluded_problems": {"a/1": "x"}})
    _, new = revised(tmp_path, old_path)
    with pytest.raises(ValueError, match="drops exclusions"):
        main_eval.newly_excluded(earlier, new)


def test_rescore_refuses_the_wrong_source_suite_or_unfinished_shards(tmp_path: Path) -> None:
    old_path, _, old = sampled_evaluation(tmp_path)
    new_path, new = revised(tmp_path, old_path, excluded_problems={"b/1": "x"})
    units = main_eval.eval_units(new, tmp_path, None)
    with pytest.raises(SystemExit, match="is not the suite"):  # new passed as the old suite
        main_eval.run_rescore(rescore_cli(new_path, new_path, old, tmp_path / "o1"), new, units)
    (old / "claims" / "shard-1" / "done").unlink()
    with pytest.raises(SystemExit, match=r"shards \[1\] not finished"):
        main_eval.run_rescore(rescore_cli(new_path, old_path, old, tmp_path / "o2"), new, units)
