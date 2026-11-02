"""M1.11: effective rollout-token rate of the throughput runs, and wall-clock hours for the budgets.

    python scripts/throughput_report.py --runs runs/<B-F0 run> --runs runs/<B-L run> \\
        --out docs/reports/m1_throughput.json

Per run (docs/roadmap/M1.md, M1.11):
- R = rollout tokens / wall-clock hours of every job segment (start-up, the step-0 validation,
  every step, its saves and exports, every validation pass and the shutdown; queue time is not
  in a segment). The spread is the coefficient of variation of the per-step rates of steps 2..N
  (rollout tokens / verl step seconds; step 1 warms up), applied to R.
- For 30, 50, 100 and 200M tokens: hours = B / R (with the spread), jobs = ceil(hours / 66)
  (72 h jobs minus start-up and lost-interval slack), and hours x 1.25 for queueing and restarts.
- Sensitivity to length: a least-squares line of step seconds on the step's mean response length
  (steps 2..N), so R at length L is R x rate(L) / rate(L_run), rate(L) = L / (a + b L). It is
  evaluated at 0.5, 1 and 1.5 times the base model's length (step 1's mean; both runs start from
  the base). The fit spans only the run's own lengths, so `extrapolated` marks a factor whose
  length lies outside them; read those values as a model, not a measurement.
"""

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.rl.rollout_tokens import tokens_by_step
from soe.trainer.rl.segments import merged_metrics, read_events

BUDGETS = (30_000_000, 50_000_000, 100_000_000, 200_000_000)
JOB_HOURS = 66.0
QUEUE_FACTOR = 1.25
LENGTH_FACTORS = (0.5, 1.0, 1.5)


def wall_hours(run_dir: Path) -> float:
    """Hours of every job segment, from its start event to its end event."""
    starts: dict[str, datetime] = {}
    ends: dict[str, datetime] = {}
    for event in read_events(run_dir):
        (starts if event.event == "start" else ends)[event.segment] = event.at
    missing = sorted(set(starts) - set(ends))
    if missing:
        raise ValueError(f"{run_dir}: segments without an end event: {missing}")
    return sum((ends[s] - starts[s]).total_seconds() for s in starts) / 3600


def line_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Least squares y = a + b x, with the standard error of b."""
    n, mx, my = len(xs), statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    a = my - b * mx
    residual = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys, strict=True))
    se_b = math.sqrt(residual / (n - 2) / sxx) if n > 2 else math.nan
    return {"a": a, "b": b, "se_b": se_b}


def budget_table(rate: float, spread: float) -> list[dict[str, Any]]:
    """Hours and jobs per budget at R tokens/h, with R x (1 -/+ spread) as the range."""
    rows = []
    for budget in BUDGETS:
        hours = budget / rate
        rows.append(
            {
                "rollout_tokens": budget,
                "hours": hours,
                "hours_range": [budget / (rate * (1 + spread)), budget / (rate * (1 - spread))],
                "jobs": math.ceil(hours / JOB_HOURS),
                "hours_with_queue_factor": hours * QUEUE_FACTOR,
            }
        )
    return rows


def run_throughput(run_dir: Path) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    assert config.hardware is not None
    metrics = {s: d for s, d in merged_metrics(run_dir / "metrics").items() if s >= 1}
    tokens = {s: sum(t) for s, t in tokens_by_step(run_dir / "rollout_tokens").items()}
    steps = sorted(s for s in metrics if s in tokens and "timing_s/step" in metrics[s])
    if len(steps) < 4 or steps != list(range(1, steps[-1] + 1)):
        raise ValueError(f"{run_dir}: need steps 1..N (N >= 4) with timings, got {steps}")
    hours = wall_hours(run_dir)
    total_tokens = sum(tokens[s] for s in steps)
    rate = total_tokens / hours
    steady = steps[1:]
    per_step = [tokens[s] / metrics[s]["timing_s/step"] * 3600 for s in steady]
    spread = statistics.stdev(per_step) / statistics.fmean(per_step)
    step_s = sum(metrics[s]["timing_s/step"] for s in steps)
    validation_s = {
        s: metrics[s]["timing_s/testing"] for s in steps if "timing_s/testing" in metrics[s]
    }
    save_s = {
        s: metrics[s]["timing_s/save_checkpoint"]
        for s in steps
        if "timing_s/save_checkpoint" in metrics[s]
    }
    lengths = [metrics[s]["response_length/mean"] for s in steady]
    fit = line_fit(lengths, [metrics[s]["timing_s/step"] for s in steady])
    base_length = metrics[1]["response_length/mean"]
    run_length = statistics.fmean(lengths)

    def step_rate(length: float) -> float:
        return length / (fit["a"] + fit["b"] * length)

    sensitivity = []
    for factor in LENGTH_FACTORS:
        length = factor * base_length
        scaled = rate * step_rate(length) / step_rate(run_length)
        sensitivity.append(
            {
                "length_factor": factor,
                "mean_response_length": length,
                "rate_tokens_per_hour": scaled,
                "extrapolated": not min(lengths) <= length <= max(lengths),
                "budgets": budget_table(scaled, spread),
            }
        )
    starts = [e for e in read_events(run_dir) if e.event == "start"]
    return {
        "run_dir": str(run_dir),
        "config_name": config.name,
        "training_mode": config.training_mode.value,
        "gpus": (run_dir / "hardware.txt").read_text().strip(),
        "node": starts[0].node,
        "commit": starts[0].git_commit,
        "segments": len(starts),
        "steps": len(steps),
        "rollout_tokens": total_tokens,
        "wall_clock_hours": hours,
        "rate_tokens_per_hour": rate,
        "rate_spread": spread,
        "per_step_rate_tokens_per_hour": dict(zip(steady, per_step, strict=True)),
        "time_split_hours": {
            "steps": step_s / 3600,  # verl's step timer: generate ... update, weight sync, save
            "validation": sum(validation_s.values()) / 3600,
            # start-up, the step-0 validation, the end-of-run export sweep and shutdown
            "other": hours - (step_s + sum(validation_s.values())) / 3600,
        },
        "validation_passes_s": validation_s,
        "save_s": save_s,
        "base_response_length": base_length,
        "run_response_length": run_length,
        "length_fit": fit,
        "budgets": budget_table(rate, spread),
        "length_sensitivity": sensitivity,
    }


class ThroughputCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/throughput_report.py", cli_kebab_case=True
    )

    runs: list[Path]
    out: Path

    def cli_cmd(self) -> None:
        runs = [run_throughput(r) for r in self.runs]
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "job_hours": JOB_HOURS,
            "queue_factor": QUEUE_FACTOR,
            "runs": runs,
        }
        self.out.write_text(json.dumps(report, indent=2) + "\n")
        for run in runs:
            print(
                f"{run['config_name']}: R = {run['rate_tokens_per_hour'] / 1e6:.2f}M tokens/h "
                f"± {run['rate_spread']:.1%} over {run['wall_clock_hours']:.2f} h"
            )
            for row in run["budgets"]:
                print(
                    f"  {row['rollout_tokens'] / 1e6:.0f}M: {row['hours']:.1f} h, "
                    f"{row['jobs']} job(s), {row['hours_with_queue_factor']:.1f} h with queueing"
                )


if __name__ == "__main__":
    CliApp.run(ThroughputCli)
