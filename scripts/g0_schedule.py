"""The G0 schedule (M1.12): Lane A hours and Lane B evaluation days per milestone, from M1's rates.

    python scripts/g0_schedule.py --inputs scripts/g0_schedule.yaml \\
        --out docs/reports/g0_schedule.json

Lane A (4xA40, fc098): an RL run takes budget / R hours, with R the M1.11 effective rate over its
length band (lowest and highest of R x (1 -/+ spread) and the 0.5x / 1.5x length rescalings); SFT
and DPO take training tokens / the M1.10 update rate (forward and backward over every token, the
same work as an SFT step). Hours are also given x the queue factor.
Lane B (1-GPU workers): every save of an RL run (one per `save_every_hours`) gets one full
evaluation, costing what M0's 4B reference cost in A40-hours (its completion tokens / M0's A40
rate); preference sampling costs tokens / the A40 rate. A pool of GPUs counts as A40-equivalents
(GPUs x its type's M0 rate / the A40 rate), and days assume the pool busy around the clock. Each
pool in `lane_b_pools` is one scenario.
The critical path adds, per milestone, the longer of the two lanes: a gate waits for the
evaluations of its runs. It leaves out SFT data curation, human review and write-up time.
"""

import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

Range = tuple[float, float]


class Milestone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rl_runs: tuple[int, int] = (0, 0)
    rl_tokens: Range = (0.0, 0.0)
    train_tokens: Range = (0.0, 0.0)
    generated_samples: Range = (0.0, 0.0)
    speed_factor: Range = (1.0, 1.0)
    extra_evals: int = 0


class ScheduleInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    throughput_report: Path
    throughput_run: str
    memory_report: Path
    memory_cell: str
    eval_rates_report: Path
    queue_factor: float
    save_every_hours: float
    evals_per_run_cap: int
    lane_b_pools: dict[str, dict[str, int]]  # scenario -> GPU type -> GPUs
    mean_response_tokens: Range
    milestones: list[Milestone]


def rl_rate_band(throughput: dict[str, Any], run_name: str) -> dict[str, float]:
    """R (rollout tokens per hour) and the lowest and highest rate the M1.11 report supports."""
    (run,) = [r for r in throughput["runs"] if r["config_name"] == run_name]
    rate, spread = run["rate_tokens_per_hour"], run["rate_spread"]
    candidates = [rate * (1 - spread), rate * (1 + spread)]
    candidates += [s["rate_tokens_per_hour"] for s in run["length_sensitivity"]]
    return {"rate": rate, "low": min(candidates), "high": max(candidates)}


def update_rate(memory: dict[str, Any], cell: str) -> float:
    """Training tokens per hour of the update phase (steps 2..N of the measured cell)."""
    (entry,) = [c for c in memory["cells"] if c["cell"] == cell]
    (measured,) = entry["measured"]
    steps = measured["steps"].values()
    tokens = sum(float(s["training_tokens"]) for s in steps)
    return 3600 * tokens / sum(float(s["update_actor_s"]) for s in steps)


def a40_rate(rates: dict[str, Any]) -> float:
    """M0's A40 sampling rate, tokens per hour."""
    return 3600 * float(rates["by_gpu"]["A40"]["tokens_per_second"])


def pool_a40_equivalents(rates: dict[str, Any], pool: dict[str, int]) -> float:
    a40 = float(rates["by_gpu"]["A40"]["tokens_per_second"])
    return sum(
        n * float(rates["by_gpu"][gpu]["tokens_per_second"]) / a40 for gpu, n in pool.items()
    )


def milestone_plan(
    m: Milestone,
    inputs: ScheduleInputs,
    rl: dict[str, float],
    train_rate: float,
    rates: dict[str, Any],
) -> dict[str, Any]:
    eval_h = float(rates["completion_tokens"]) / a40_rate(rates)
    pools = {name: pool_a40_equivalents(rates, p) for name, p in inputs.lane_b_pools.items()}

    def lane_a(i: int) -> tuple[float, float]:  # i = 0: best case, 1: worst case
        rate = rl["high"] if i == 0 else rl["low"]
        run_hours = m.rl_tokens[i] / rate if m.rl_runs[i] else 0.0
        hours = m.speed_factor[i] * (m.rl_runs[i] * run_hours + m.train_tokens[i] / train_rate)
        return hours, m.speed_factor[i] * run_hours

    def lane_b(i: int, run_hours: float, cap: int | None) -> dict[str, Any]:
        per_run = math.ceil(run_hours / inputs.save_every_hours) if m.rl_runs[i] else 0
        if cap is not None:
            per_run = min(per_run, cap)
        evals = m.rl_runs[i] * per_run + m.extra_evals
        sampling_tokens = m.generated_samples[i] * inputs.mean_response_tokens[i]
        a40_h = evals * eval_h + sampling_tokens / a40_rate(rates)
        return {
            "evaluations": evals,
            "a40_hours": a40_h,
            "days": {name: a40_h / eq / 24 for name, eq in pools.items()},
        }

    out: dict[str, Any] = {"name": m.name}
    for label, i in (("low", 0), ("high", 1)):
        hours, run_hours = lane_a(i)
        out[label] = {
            "lane_a_hours": hours,
            "lane_a_hours_with_queue_factor": hours * inputs.queue_factor,
            "lane_a_days_with_queue_factor": hours * inputs.queue_factor / 24,
            "hours_per_rl_run": run_hours,
            "lane_b_every_save": lane_b(i, run_hours, None),
            "lane_b_capped": lane_b(i, run_hours, inputs.evals_per_run_cap),
        }
    return out


def schedule(inputs: ScheduleInputs) -> dict[str, Any]:
    rl = rl_rate_band(json.loads(inputs.throughput_report.read_text()), inputs.throughput_run)
    train = update_rate(json.loads(inputs.memory_report.read_text()), inputs.memory_cell)
    rates = json.loads(inputs.eval_rates_report.read_text())
    rows = [milestone_plan(m, inputs, rl, train, rates) for m in inputs.milestones]
    totals: dict[str, Any] = {}
    for label in ("low", "high"):
        a = sum(r[label]["lane_a_days_with_queue_factor"] for r in rows)
        totals[label] = {"lane_a_days_with_queue_factor": a}
        for policy in ("lane_b_every_save", "lane_b_capped"):
            days = {
                pool: [r[label][policy]["days"][pool] for r in rows] for pool in inputs.lane_b_pools
            }
            totals[label][policy] = {
                "evaluations": sum(r[label][policy]["evaluations"] for r in rows),
                "a40_hours": sum(r[label][policy]["a40_hours"] for r in rows),
                "lane_b_days": {pool: sum(d) for pool, d in days.items()},
                "critical_path_days": {
                    pool: sum(
                        max(r[label]["lane_a_days_with_queue_factor"], b)
                        for r, b in zip(rows, d, strict=True)
                    )
                    for pool, d in days.items()
                },
            }
    return {
        "inputs": json.loads(inputs.model_dump_json()),
        "rl_rate_tokens_per_hour": rl,
        "train_tokens_per_hour": train,
        "eval_a40_hours_per_checkpoint": float(rates["completion_tokens"]) / a40_rate(rates),
        "lane_b_pool_a40_equivalents": {
            name: pool_a40_equivalents(rates, p) for name, p in inputs.lane_b_pools.items()
        },
        "milestones": rows,
        "totals": totals,
    }


class ScheduleCli(BaseSettings):
    model_config = SettingsConfigDict(cli_prog_name="scripts/g0_schedule.py", cli_kebab_case=True)

    inputs: Path
    out: Path

    def cli_cmd(self) -> None:
        inputs = ScheduleInputs.model_validate(yaml.safe_load(self.inputs.read_text()))
        result = schedule(inputs)
        self.out.write_text(json.dumps(result, indent=2) + "\n")
        for label, t in result["totals"].items():
            every, capped = t["lane_b_every_save"], t["lane_b_capped"]
            print(f"{label}: Lane A {t['lane_a_days_with_queue_factor']:.1f} d")
            for pool in inputs.lane_b_pools:
                print(
                    f"  {pool}: Lane B {every['lane_b_days'][pool]:.1f} d "
                    f"({every['evaluations']} evals), capped {capped['lane_b_days'][pool]:.1f} d; "
                    f"critical path {every['critical_path_days'][pool]:.1f} d / "
                    f"{capped['critical_path_days'][pool]:.1f} d capped"
                )


if __name__ == "__main__":
    CliApp.run(ScheduleCli)
