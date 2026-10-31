"""The M1.9 acceptance script applies the four declared tests and their thresholds."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_spec = importlib.util.spec_from_file_location(
    "dev_run_acceptance", Path("scripts/dev_run_acceptance.py")
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
acceptance, eval_tests, reward_trend = _module.acceptance, _module.eval_tests, _module.reward_trend


def level(pass1: float, truncation: float, boxed: float = 1.0) -> dict[str, float]:
    return {"pass@1": pass1, "truncation_rate": truncation, "boxed_rate_when_stopped": boxed}


def report(
    levels: dict[str, dict[str, float]], sha: str = "s", checkpoint: str = "/c/step_4/hf"
) -> dict[str, Any]:
    return {
        "plan": {"checkpoint": checkpoint, "only_problems_sha256": sha},
        "main": {"per_level": levels},
    }


def test_reward_trend_windows_are_20_percent_and_compare_to_2_se() -> None:
    rewards = [0.30, 0.32] * 5 + [0.40, 0.42] * 5  # 20 steps: windows of 4
    t = reward_trend(rewards)
    assert t["window_steps"] == 4
    assert t["first_mean"] == pytest.approx(0.31)
    assert t["last_mean"] == pytest.approx(0.41)
    assert t["se"] == pytest.approx((2 * 0.0001333333 / 4) ** 0.5, rel=1e-4)
    assert t["pass"]


def test_reward_trend_within_noise_fails() -> None:
    t = reward_trend([0.3, 0.5, 0.3, 0.5, 0.4, 0.4, 0.5, 0.3, 0.5, 0.31])
    assert not t["pass"]


def test_reward_trend_needs_two_windows() -> None:
    with pytest.raises(ValueError):
        reward_trend([0.1, 0.2, 0.3])


def test_eval_tests_apply_each_threshold_per_level() -> None:
    base = report({"A": level(0.40, 0.50), "B": level(0.40, 0.50)})
    ok = report({"A": level(0.351, 0.60, 0.95), "B": level(0.50, 0.20)})
    bad = report({"A": level(0.34, 0.61, 0.94), "B": level(0.40, 0.50)})
    good = eval_tests(base, ok)
    assert all(good["A"][k] for k in ("pass@1_ok", "boxed_ok", "truncation_ok"))
    failed = eval_tests(base, bad)["A"]
    assert not (failed["pass@1_ok"] or failed["boxed_ok"] or failed["truncation_ok"])


def test_acceptance_rejects_a_different_problem_list(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lines = [{"step": s, "data": {"soe/reward/mean": 0.1 * s}} for s in range(1, 11)]
    (run / "rollout_metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    for name, sha in (("base", "s"), ("ck", "other")):
        (tmp_path / name).mkdir()
        levels = {"A": level(0.4, 0.5)}
        (tmp_path / name / "report.json").write_text(json.dumps(report(levels, sha)))
    with pytest.raises(ValueError, match="different problem list"):
        acceptance(run, tmp_path / "base", [tmp_path / "ck"])


def test_cli_takes_one_checkpoints_flag_per_evaluation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lines = [{"step": s, "data": {"soe/reward/mean": 0.1 * s}} for s in range(1, 11)]
    (run / "rollout_metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    for name in ("base", "a", "b"):
        (tmp_path / name).mkdir()
        levels = {"A": level(0.4, 0.5)}
        (tmp_path / name / "report.json").write_text(json.dumps(report(levels)))
    out = tmp_path / "out.json"
    args = ["--run-dir", str(run), "--base", str(tmp_path / "base"), "--out", str(out)]
    args += ["--checkpoints", str(tmp_path / "a"), "--checkpoints", str(tmp_path / "b")]
    _module.CliApp.run(_module.DevRunAcceptanceCli, cli_args=args)
    assert len(json.loads(out.read_text())["evaluations"]) == 2
