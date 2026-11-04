"""scripts/sft_run2_check.py on synthetic evaluations and training logs (M2.11, decision 029)."""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.records import SampleRecord

GOOD = "reasoning</think>so \\boxed{1}"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sft_run2_check", Path("scripts/sft_run2_check.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load()


def make_eval(
    directory: Path,
    correct: Callable[[int, int], bool],
    truncated: Callable[[int, int], bool] = lambda p, s: False,
    problems: int = 256,
) -> Path:
    """problem p, sample s: correct(p, s) and truncated(p, s) decide the record."""
    directory.mkdir(parents=True)
    plan = {"suite_name": check.SUITE_NAME, "checkpoint": directory.name, "num_shards": 2}
    (directory / "plan.json").write_text(json.dumps(plan))
    shards: list[list[str]] = [[], []]
    for p in range(problems):
        for s in range(16):
            cut = truncated(p, s)
            record = SampleRecord(
                problem_id=f"q{p:03d}",
                sample_index=s,
                checkpoint=directory.name,
                completion="reasoning that never ends" if cut else GOOD,
                extracted_answer=None if cut else "1",
                correct=correct(p, s) and not cut,
                completion_tokens=16384 if cut else 100,
                finish_reason="length" if cut else "stop",
            )
            shards[p % 2].append(record.model_dump_json())
    for k, lines in enumerate(shards):
        (directory / f"samples-{k}-of-2.jsonl").write_text("\n".join(lines) + "\n")
    return directory


def make_run(run: Path, val: tuple[float, float, float] = (0.40, 0.38, 0.37)) -> Path:
    (run / "metrics").mkdir(parents=True)
    lines = [{"step": s, "data": {"loss": 1.0 - 0.05 * s, "grad_norm": 1.0}} for s in range(1, 9)]
    lines += [{"step": s, "data": {"eval_loss": v}} for s, v in zip((4, 6, 8), val, strict=True)]
    (run / "metrics" / "segment-1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return run


def base_correct(p: int, s: int) -> bool:
    return p % 2 == 0 and s < 4  # half the problems, 4 of 16 samples each


def base_truncated(p: int, s: int) -> bool:
    return s >= 12  # a quarter of the samples


def test_a_healthy_sft_run_needs_no_run_2(tmp_path: Path) -> None:
    base = make_eval(tmp_path / "base", base_correct, base_truncated)
    sft = make_eval(tmp_path / "sft", lambda p, s: p % 2 == 0 and s < 8)
    out = check.check(base, sft, make_run(tmp_path / "run"))
    assert out["triggered"] == [] and out["run2_changes"] == []
    assert out["verdict"].startswith("no run #2")
    assert out["base"]["pass@1"] == pytest.approx(0.125) and out["sft"]["pass@1"] == 0.25
    assert out["base"]["truncation_rate"] == 0.25 and out["sft"]["truncation_rate"] == 0.0
    assert out["criteria"]["4"]["pass@16"]["drop"] == 0.0
    assert out["criteria"]["1"]["first_quarter_mean_loss"] == pytest.approx(0.925)
    assert out["criteria"]["1"]["last_quarter_mean_loss"] == pytest.approx(0.625)


def test_lost_coverage_and_kept_truncation_trigger_both_changes(tmp_path: Path) -> None:
    base = make_eval(tmp_path / "base", base_correct, base_truncated)
    # SFT solves only a quarter of the problems the base solves; truncation as the base.
    sft = make_eval(tmp_path / "sft", lambda p, s: p % 8 == 0 and s < 4, base_truncated)
    out = check.check(base, sft, make_run(tmp_path / "run"))
    assert out["triggered"] == [3, 4] and out["verdict"] == "run #2"
    drop = out["criteria"]["4"]["pass@16"]
    assert drop["drop"] == pytest.approx(0.375) and drop["holds"] and drop["se"] > 0
    assert out["run2_changes"] == sorted([check.CHANGES[3], check.CHANGES[4]])


def test_validation_loss_rising_twice_is_a_training_failure(tmp_path: Path) -> None:
    base = make_eval(tmp_path / "base", base_correct, base_truncated)
    sft = make_eval(tmp_path / "sft", base_correct)
    once = check.check(base, sft, make_run(tmp_path / "once", (0.40, 0.41, 0.39)))
    assert not once["criteria"]["1"]["holds"]
    twice = check.check(base, sft, make_run(tmp_path / "twice", (0.40, 0.41, 0.42)))
    assert twice["triggered"] == [1] and twice["run2_changes"] == [check.CHANGES[1]]


def test_format_rule_needs_one_think_close_and_a_box_after_it() -> None:
    def record(text: str) -> SampleRecord:
        return SampleRecord(
            problem_id="q",
            sample_index=0,
            checkpoint="c",
            completion=text,
            extracted_answer="1" if "boxed" in text else None,
            correct=True,
            completion_tokens=5,
            finish_reason="stop",
        )

    assert check.well_formed(record(GOOD))
    assert not check.well_formed(record("a</think>b</think>\\boxed{1}"))
    assert not check.well_formed(record("\\boxed{1}</think>no box"))
    assert not check.well_formed(record("no think \\boxed{1}"))


def test_an_incomplete_evaluation_is_refused(tmp_path: Path) -> None:
    short = make_eval(tmp_path / "short", base_correct, problems=255)
    with pytest.raises(SystemExit, match="256 problems"):
        check.read_eval(short)


def test_check_suite_is_soe_eval_v4_on_the_rl_validation_split() -> None:
    suite = load_suite(Path("examples/eval/m2_run2_check_suite.yaml"))
    v4 = load_suite(Path("soe/trainer/config/eval_suite.yaml"))
    assert suite.name == check.SUITE_NAME
    for part in ("sampling", "prompt", "scoring", "main", "hard"):
        assert getattr(suite, part) == getattr(v4, part)
    (rl_val,) = suite.problem_sets
    report = json.loads(Path(suite.data_build_report).read_text())
    assert rl_val.path == "datasets/soe/v3.1/rl/val.jsonl"
    assert rl_val.sha256 in json.dumps(report) and suite.hard_subset is None
