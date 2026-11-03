"""m2_sft_sample: disjoint seeded draws, and yield statistics from kept completions."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from soe.trainer.config.eval_suite import file_sha256
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import write_jsonl


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_sample.py")
    spec = importlib.util.spec_from_file_location("m2_sft_sample", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(m: ModuleType, tmp_path: Path, start: int, stop: int) -> Any:
    pool = tmp_path / "pool.jsonl"
    if not pool.exists():
        write_jsonl(
            pool,
            [
                Problem(
                    problem_id=f"d/{i:02d}", source="d", level=Level.TRAIN, question="q", answer="1"
                )
                for i in range(20)
            ],
        )
    return m.SampleConfig(
        version="t",
        suite=Path("soe/trainer/config/eval_suite.yaml"),
        model="Qwen/Qwen3.5-4B-Base",
        problems="pool.jsonl",
        problems_sha256=file_sha256(pool),
        meta="meta.jsonl",
        draw_seed=7,
        draw_start=start,
        draw_stop=stop,
        samples_per_problem=4,
        max_kept_per_problem=2,
        sampling_seed=2,
        num_shards=2,
        work_dir="work",
    )


def test_draws_are_seeded_and_the_pilot_and_full_run_never_overlap(tmp_path: Path) -> None:
    m = script()
    pilot = m.drawn(config(m, tmp_path, 0, 5), tmp_path)
    full = m.drawn(config(m, tmp_path, 5, 20), tmp_path)
    assert {p.problem_id for p in pilot}.isdisjoint(p.problem_id for p in full)
    assert len({p.problem_id for p in pilot + full}) == 20
    assert m.drawn(config(m, tmp_path, 0, 5), tmp_path) == pilot  # deterministic
    with pytest.raises(SystemExit, match="outside"):
        m.drawn(config(m, tmp_path, 0, 21), tmp_path)


def record(pid: str, i: int, correct: bool, stop: bool = True) -> SampleRecord:
    return SampleRecord(
        problem_id=pid,
        sample_index=i,
        checkpoint="m",
        completion=f"r{i}</think> \\boxed{{1}}" if stop else f"r{i}",
        extracted_answer="1" if stop else None,
        correct=correct and stop,
        completion_tokens=1000 * (i + 1) if stop else 16384,
        finish_reason="stop" if stop else "length",
    )


def test_yield_stats_counts_kept_completions_per_problem() -> None:
    m = script()
    by_problem = {
        "a": [record("a", i, True) for i in range(4)],  # 4 correct -> 2 kept
        "b": [record("b", 0, True), *[record("b", i, False, stop=False) for i in (1, 2, 3)]],
        "c": [record("c", i, False) for i in range(4)],  # none kept
    }
    stats = m.yield_stats(by_problem, {"a": 5.0, "b": 5.0, "c": 8.0}, max_kept=2, seed=0)
    assert stats["kept"] == 3 and stats["problems_with_a_kept"] == 2
    assert stats["kept_per_problem"]["mean"] == 1.0
    low, high = stats["kept_per_problem"]["ci95"]
    assert 0.0 <= low <= 1.0 <= high <= 2.0
    assert stats["truncation_rate"] == 3 / 12 and stats["stopped_boxed_rate"] == 1.0
    assert stats["kept_tokens"]["histogram_upper_edges"]["2048"] == 3  # 1,000 / 2,000 / 1,000
    assert stats["kept_per_problem_by_difficulty"]["5"] == {"problems": 2, "mean": 1.5}


def test_pilot_config_samples_the_clean_pool_as_roadmap_d4() -> None:
    import json

    import yaml

    m = script()
    pilot = m.SampleConfig.model_validate(
        yaml.safe_load(Path("examples/data_preprocess/m2_sft_pilot.yaml").read_text())
    )
    clean = json.loads(Path("docs/reports/m2_sft_pool_clean.json").read_text())["clean"]
    assert (pilot.problems, pilot.problems_sha256) == (clean["path"], clean["sha256"])
    assert (pilot.draw_start, pilot.draw_stop) == (0, 1000) and pilot.draw_stop <= clean["problems"]
    assert (pilot.samples_per_problem, pilot.max_kept_per_problem, pilot.sampling_seed) == (4, 2, 2)
    assert m.pinned_revision(pilot.model) == "1001bb4d826a52d1f399e183466143f4da7b741b"


def test_full_config_continues_the_pilot_permutation() -> None:
    import yaml

    m = script()
    pilot, full = (
        m.SampleConfig.model_validate(
            yaml.safe_load(Path(f"examples/data_preprocess/m2_sft_{name}.yaml").read_text())
        )
        for name in ("pilot", "full")
    )
    same = ("suite", "model", "problems", "problems_sha256", "meta", "draw_seed", "sampling_seed")
    assert all(getattr(pilot, f) == getattr(full, f) for f in same)
    assert (pilot.samples_per_problem, pilot.max_kept_per_problem) == (4, 2)
    assert (full.samples_per_problem, full.max_kept_per_problem) == (4, 2)
    assert (full.draw_start, full.draw_stop) == (pilot.draw_stop, 16000)  # M2.4's size
    assert full.work_dir != pilot.work_dir
