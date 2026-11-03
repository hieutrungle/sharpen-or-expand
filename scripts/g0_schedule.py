"""The G0 schedule (M1.12): Lane A hours and Lane B evaluation days per milestone, from M1's rates.

    python scripts/g0_schedule.py --inputs scripts/g0_schedule.yaml \\
        --out docs/reports/g0_schedule.json

Lane A (4xA40, fc098): an RL run takes budget / R hours, with R the M1.11 effective rate over its
length band (lowest and highest of R x (1 -/+ spread) and the 0.5x / 1.5x length rescalings); SFT
and DPO take training tokens / the M1.10 update rate (forward and backward over every token, the
same work as an SFT step). Hours are also given x the queue factor.
Lane B (1-GPU workers): every save of an RL run (one per `save_every_hours`) gets one full
evaluation, costing what M0's 4B reference cost; preference sampling costs tokens / M0's sampling
rate. Days assume `lane_b_workers` workers busy around the clock.
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
    eval_compute_report: Path
    eval_family: str
    eval_samples: int
    queue_factor: float
    save_every_hours: float
    evals_per_run_cap: int
    lane_b_workers: int
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


def eval_gpu_hours(compute: dict[str, Any], family: str) -> float:
    return float(compute["families"][family]["gpu_hours"])


def milestone_plan(
    m: Milestone, inputs: ScheduleInputs, rl: dict[str, float], train_rate: float, eval_h: float
) -> dict[str, Any]:
    # M0's sampling rate: its samples at M0's mean length (the high end of mean_response_tokens)
    tokens_per_gpu_hour = inputs.eval_samples * inputs.mean_response_tokens[1] / eval_h

    def lane_a(i: int) -> tuple[float, float]:  # i = 0: best case, 1: worst case
        rate = rl["high"] if i == 0 else rl["low"]
        run_hours = m.rl_tokens[i] / rate if m.rl_runs[i] else 0.0
        hours = m.speed_factor[i] * (m.rl_runs[i] * run_hours + m.train_tokens[i] / train_rate)
        return hours, m.speed_factor[i] * run_hours

    def lane_b(i: int, run_hours: float, cap: int | None) -> dict[str, float]:
        per_run = math.ceil(run_hours / inputs.save_every_hours) if m.rl_runs[i] else 0
        if cap is not None:
            per_run = min(per_run, cap)
        evals = m.rl_runs[i] * per_run + m.extra_evals
        sampling_h = m.generated_samples[i] * inputs.mean_response_tokens[i]
        gpu_h = evals * eval_h + sampling_h / tokens_per_gpu_hour
        return {
            "evaluations": evals,
            "gpu_hours": gpu_h,
            "days": gpu_h / inputs.lane_b_workers / 24,
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
    eval_h = eval_gpu_hours(json.loads(inputs.eval_compute_report.read_text()), inputs.eval_family)
    rows = [milestone_plan(m, inputs, rl, train, eval_h) for m in inputs.milestones]
    totals: dict[str, Any] = {}
    for label in ("low", "high"):
        a = sum(r[label]["lane_a_days_with_queue_factor"] for r in rows)
        totals[label] = {"lane_a_days_with_queue_factor": a}
        for policy in ("lane_b_every_save", "lane_b_capped"):
            b = sum(r[label][policy]["days"] for r in rows)
            path = sum(
                max(r[label]["lane_a_days_with_queue_factor"], r[label][policy]["days"])
                for r in rows
            )
            totals[label][policy] = {
                "evaluations": sum(r[label][policy]["evaluations"] for r in rows),
                "lane_b_days": b,
                "critical_path_days": path,
            }
    return {
        "inputs": json.loads(inputs.model_dump_json()),
        "rl_rate_tokens_per_hour": rl,
        "train_tokens_per_hour": train,
        "eval_gpu_hours_per_checkpoint": eval_h,
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
            print(
                f"{label}: Lane A {t['lane_a_days_with_queue_factor']:.1f} d; Lane B "
                f"{every['lane_b_days']:.1f} d ({every['evaluations']} evals), capped "
                f"{capped['lane_b_days']:.1f} d; critical path {every['critical_path_days']:.1f} d"
                f" / {capped['critical_path_days']:.1f} d capped"
            )


if __name__ == "__main__":
    CliApp.run(ScheduleCli)
