"""M1.9 acceptance: the four tests declared in docs/roadmap/M1.md before the dev run started.

    python scripts/dev_run_acceptance.py --run-dir runs/<id> --base $SOE_DATA/eval/<base> \\
        --checkpoints $SOE_DATA/eval/<step a> --checkpoints $SOE_DATA/eval/<step b> ... \\
        --out docs/reports/m1_dev_qwen35-2b_acceptance.json

1. Training reward: the mean of the per-step reward means over the last 20% of steps exceeds the
   first 20% by more than 2 standard errors (SE of the difference of the two window means, from
   the spread of the per-step means in each window).
2. On the dev subset, no level's pass@1 drops by more than 5 points against the base model.
3. The boxed rate among samples that stopped on their own (not truncated) is >= 95% per level.
4. The truncation rate rises by no more than 10 points per level against the base model.
Evaluation tests are applied to every evaluated checkpoint; the run passes only if all do.
Inputs are `rollout_metrics.jsonl` (soe step metrics, one record per trained step, the replayed
steps of a resumed run already overwritten) and each evaluation's `report.json` (main_eval).
"""

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.rl.metrics import STEP_METRICS, read_step_metrics

REWARD_KEY = "soe/reward/mean"
WINDOW_FRACTION = 0.2
MIN_SE_MULTIPLE = 2.0
MAX_PASS1_DROP = 0.05
MIN_BOXED_WHEN_STOPPED = 0.95
MAX_TRUNCATION_RISE = 0.10


def reward_trend(rewards: list[float]) -> dict[str, float | int | bool]:
    """Compare the first and last 20% of per-step reward means (at least 2 steps per window)."""
    k = max(2, round(WINDOW_FRACTION * len(rewards)))
    if 2 * k > len(rewards):
        raise ValueError(f"{len(rewards)} steps are too few for two windows of {k}")
    first, last = rewards[:k], rewards[-k:]
    se = math.sqrt(statistics.variance(first) / k + statistics.variance(last) / k)
    diff = statistics.fmean(last) - statistics.fmean(first)
    return {
        "steps": len(rewards),
        "window_steps": k,
        "first_mean": statistics.fmean(first),
        "last_mean": statistics.fmean(last),
        "difference": diff,
        "se": se,
        "se_multiple": diff / se if se > 0 else math.inf,
        "pass": diff > MIN_SE_MULTIPLE * se,
    }


def eval_tests(base: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-level tests 2-4 of one checkpoint report against the base report."""
    out = {}
    for level, b in base["main"]["per_level"].items():
        c = checkpoint["main"]["per_level"][level]
        pass1_change = c["pass@1"] - b["pass@1"]
        truncation_change = c["truncation_rate"] - b["truncation_rate"]
        out[level] = {
            "pass@1_base": b["pass@1"],
            "pass@1": c["pass@1"],
            "pass@1_change": pass1_change,
            "boxed_rate_when_stopped": c["boxed_rate_when_stopped"],
            "truncation_rate_base": b["truncation_rate"],
            "truncation_rate": c["truncation_rate"],
            "truncation_change": truncation_change,
            "pass@1_ok": pass1_change >= -MAX_PASS1_DROP,
            "boxed_ok": c["boxed_rate_when_stopped"] >= MIN_BOXED_WHEN_STOPPED,
            "truncation_ok": truncation_change <= MAX_TRUNCATION_RISE,
        }
    return out


def read_report(eval_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((eval_dir / "report.json").read_text())
    return report


def acceptance(run_dir: Path, base_dir: Path, checkpoint_dirs: list[Path]) -> dict[str, Any]:
    by_step = read_step_metrics(run_dir / STEP_METRICS)
    steps = sorted(by_step)
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(f"{run_dir / STEP_METRICS} does not hold steps 1..N exactly once")
    base = read_report(base_dir)
    evaluations = {}
    for eval_dir in checkpoint_dirs:
        report = read_report(eval_dir)
        if report["plan"]["only_problems_sha256"] != base["plan"]["only_problems_sha256"]:
            raise ValueError(f"{eval_dir} used a different problem list from the base")
        evaluations[str(eval_dir)] = {
            "checkpoint": report["plan"]["checkpoint"],
            "levels": eval_tests(base, report),
        }
    trend = reward_trend([by_step[s][REWARD_KEY] for s in steps])
    eval_ok = all(
        level["pass@1_ok"] and level["boxed_ok"] and level["truncation_ok"]
        for e in evaluations.values()
        for level in e["levels"].values()
    )
    return {
        "run_dir": str(run_dir),
        "base": {"eval_dir": str(base_dir), "checkpoint": base["plan"]["checkpoint"]},
        "only_problems_sha256": base["plan"]["only_problems_sha256"],
        "thresholds": {
            "window_fraction": WINDOW_FRACTION,
            "min_se_multiple": MIN_SE_MULTIPLE,
            "max_pass@1_drop": MAX_PASS1_DROP,
            "min_boxed_when_stopped": MIN_BOXED_WHEN_STOPPED,
            "max_truncation_rise": MAX_TRUNCATION_RISE,
        },
        "reward_trend": trend,
        "evaluations": evaluations,
        "pass": bool(trend["pass"]) and eval_ok,
    }


class DevRunAcceptanceCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/dev_run_acceptance.py", cli_kebab_case=True
    )

    run_dir: Path
    base: Path
    checkpoints: list[Path]
    out: Path

    def cli_cmd(self) -> None:
        result = acceptance(self.run_dir, self.base, self.checkpoints)
        self.out.write_text(json.dumps(result, indent=2) + "\n")
        t = result["reward_trend"]
        print(
            f"reward {t['first_mean']:.4f} -> {t['last_mean']:.4f} "
            f"({t['se_multiple']:.2f} SE, {'ok' if t['pass'] else 'FAIL'})"
        )
        for e in result["evaluations"].values():
            for level, x in e["levels"].items():
                step = Path(e["checkpoint"]).parent.name
                print(
                    f"{step} {level}: pass@1 {x['pass@1_change']:+.3f}, "
                    f"boxed|stopped {x['boxed_rate_when_stopped']:.3f}, "
                    f"truncation {x['truncation_change']:+.3f}"
                )
        print(f"acceptance: {'PASS' if result['pass'] else 'FAIL'}")
        if not result["pass"]:
            sys.exit(1)


if __name__ == "__main__":
    CliApp.run(DevRunAcceptanceCli)
