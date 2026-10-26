"""The committed M0.6 config holds the pre-registered values (decision 005), and the select stage
turns sample shards into a subset and a report."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord, SamplingRunInfo
from soe.utils.jsonl import read_jsonl, write_jsonl

SCRIPT = Path("examples/data_preprocess/m0_hard_subset.py")
CONFIG = Path("examples/data_preprocess/m0_hard_subset.yaml")


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m0_hard_subset", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_config_holds_the_decision_005_values() -> None:
    m = script()
    config = m.load_config(CONFIG)
    m.check_against_suite(config, load_suite(config.suite))
    assert config.selector == "Qwen/Qwen3.5-4B-Base" and config.sampling_seed == 1
    assert config.candidate_sets == ["math500", "aime24", "aime25", "olympiadbench_en", "level_b"]
    rule = config.rule
    assert (rule.max_correct, rule.fallback_max_correct) == (4, 8)
    assert (rule.min_size, rule.target_size, rule.draw_seed) == (200, 250, 20261026)
    assert config.min_answer_rate == 0.8


def test_level_c_and_the_evaluation_seed_are_rejected() -> None:
    m = script()
    config = m.load_config(CONFIG)
    suite = load_suite(config.suite)
    with pytest.raises(ValueError, match="Level C"):
        m.check_against_suite(config.model_copy(update={"candidate_sets": ["aime26"]}), suite)
    with pytest.raises(ValueError, match="seed"):
        m.check_against_suite(config.model_copy(update={"sampling_seed": 0}), suite)


def fake_data(  # type: ignore[no-untyped-def]
    tmp_path: Path, m: ModuleType, correct_by_set: dict[str, list[int]], answer: str | None = "1"
):
    """Problem files at the suite's paths plus one sample shard with the given correct counts;
    every candidate set not named gets one problem solved 16/16."""
    config = m.load_config(CONFIG)
    suite = load_suite(config.suite)
    paths = {p.name: p.path for p in suite.problem_sets}
    records = []
    for name in config.candidate_sets:
        level = Level.B if name == "level_b" else Level.A
        source = "dapo" if name == "level_b" else name
        counts = correct_by_set.get(name, [16])
        problems = [
            Problem(problem_id=f"{name}/{i}", source=source, level=level, question="q", answer="1")
            for i in range(len(counts))
        ]
        write_jsonl(tmp_path / paths[name], problems)
        for problem, c in zip(problems, counts, strict=True):
            for k in range(16):
                records.append(
                    SampleRecord(
                        problem_id=problem.problem_id,
                        sample_index=k,
                        checkpoint="base",
                        completion="",
                        extracted_answer=answer,
                        correct=k < c,
                        completion_tokens=10,
                        finish_reason="stop",
                    )
                )
    work = tmp_path / config.work_dir
    write_jsonl(m.shard_path(work, 0, 1), records)
    info = SamplingRunInfo(
        checkpoint="base",
        revision=None,
        suite_name=suite.name,
        problem_sets=config.candidate_sets,
        samples_per_problem=16,
        seed=1,
        chat_template_kwargs={},
        stop_token_ids=[1, 2],
        git_commit="abc",
        hostname="h",
        slurm_job_id=None,
        gpus=["A40"],
        library_versions={},
    )
    write_jsonl(work / "runinfo-0-of-1.jsonl", [info])
    rule = config.rule.model_copy(update={"min_size": 3, "target_size": 4})
    return config.model_copy(update={"rule": rule, "report": tmp_path / "report.json"})


def test_select_stage_writes_the_subset_and_report(tmp_path: Path) -> None:
    m = script()
    config = fake_data(
        tmp_path, m, {"math500": [0, 2, 9, 16], "aime24": [0, 1], "level_b": [3, 4, 5, 12]}
    )
    assert m.run_select(config, tmp_path) == 0
    report = json.loads(config.report.read_text())
    assert report["eligible"] == 6 and report["selected"] == 4 and report["max_correct_used"] == 4
    assert {k: v["selected"] for k, v in report["per_set"].items()} == {
        "math500": 1,
        "aime24": 2,
        "aime25": 0,
        "olympiadbench_en": 0,
        "level_b": 1,
    }  # quotas 4 * (2, 2, 2) / 6 = 1.33 each; the leftover goes to the name that sorts first
    chosen = read_jsonl(tmp_path / config.output, Problem)
    assert [p.problem_id for p in chosen] == report["selected_problem_ids"]


def test_select_stage_halts_on_a_low_answer_rate(tmp_path: Path) -> None:
    m = script()
    config = fake_data(tmp_path, m, {"math500": [0, 0, 0, 0]}, answer=None)
    assert m.run_select(config, tmp_path) == 1
    assert json.loads(config.report.read_text())["halted_low_answer_rate"] is True
    assert not (tmp_path / config.output).exists()
