"""M2.11 run-#2 check: decision 029's four criteria for an SFT run and the verdict.

    python scripts/sft_run2_check.py --base $SOE_DATA/eval/m2.11-base-rl-val \\
        --sft $SOE_DATA/eval/m2.11-sft-run1-rl-val --run runs/<SFT run> \\
        --report docs/reports/m2_sft_run2_check.json

--base and --sft are main_eval outputs on examples/eval/m2_run2_check_suite.yaml (the RL
validation split, 256 problems x 16 samples); --run is the SFT run directory (its metrics).
Criteria (decision 029 item 5), each reported with its values, threshold and whether it holds:
1. training failure: NaN/Inf loss or grad norm; last-quarter mean loss not below the first
   quarter's; or validation loss rising at both validations after the first;
2. format failure: under 98% of SFT's stopped samples have exactly one `</think>` and a
   `\\boxed{}` answer after it;
3. truncation not reduced: SFT's truncation rate not below the base model's;
4. collapse: base minus SFT pass@1, or pass@16, above 2 bootstrap standard errors (2,000 paired
   resamples of the problems, seed 0).
Run #2 happens if any holds and applies every triggered change (item 6).
"""

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.rl.segments import merged_metrics
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl
from soe.utils.metric.pass_at_k import pass_at_k
from soe.utils.shards import shard_path

SUITE_NAME = "soe-eval-v4-rl-val"  # examples/eval/m2_run2_check_suite.yaml
PROBLEMS, SAMPLES = 256, 16
FORMAT_MIN = 0.98
SE_MULTIPLE = 2.0
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 0
CHANGES = {
    1: "learning rate 1e-5 / 3 (3.33e-6), everything else unchanged",
    2: "fix the data rule that let the format through, rebuild with M2.5's build otherwise",
    3: "drop train examples above the 90th percentile of train token lengths; val unchanged",
}
CHANGES[4] = CHANGES[1]


def read_eval(directory: Path) -> tuple[str, dict[str, list[SampleRecord]]]:
    """The checkpoint and its samples by problem; refuses another suite or an incomplete run."""
    plan = json.loads((directory / "plan.json").read_text())
    if plan["suite_name"] != SUITE_NAME:
        raise SystemExit(f"{directory}: suite {plan['suite_name']}, expected {SUITE_NAME}")
    by_problem: dict[str, list[SampleRecord]] = defaultdict(list)
    for k in range(plan["num_shards"]):
        for record in read_jsonl(shard_path(directory, k, plan["num_shards"]), SampleRecord):
            by_problem[record.problem_id].append(record)
    indices = {p: sorted(r.sample_index for r in rs) for p, rs in by_problem.items()}
    if len(indices) != PROBLEMS or any(v != list(range(SAMPLES)) for v in indices.values()):
        raise SystemExit(f"{directory}: not {PROBLEMS} problems x {SAMPLES} samples")
    return plan["checkpoint"], dict(by_problem)


def well_formed(record: SampleRecord) -> bool:
    if record.completion.count("</think>") != 1 or record.extracted_answer is None:
        return False
    return "\\boxed{" in record.completion.split("</think>", 1)[1]


def summary(by_problem: dict[str, list[SampleRecord]]) -> dict[str, float]:
    records = [r for rs in by_problem.values() for r in rs]
    stopped = [r for r in records if r.finish_reason == "stop"]
    return {
        "pass@1": per_problem_mean(by_problem, 1),
        "pass@16": per_problem_mean(by_problem, SAMPLES),
        "truncation_rate": sum(r.finish_reason == "length" for r in records) / len(records),
        "well_formed_when_stopped": sum(map(well_formed, stopped)) / len(stopped),
        "mean_completion_tokens": statistics.mean(r.completion_tokens for r in records),
    }


def per_problem(by_problem: dict[str, list[SampleRecord]], k: int) -> dict[str, float]:
    return {p: pass_at_k(len(rs), sum(r.correct for r in rs), k) for p, rs in by_problem.items()}


def per_problem_mean(by_problem: dict[str, list[SampleRecord]], k: int) -> float:
    return statistics.mean(per_problem(by_problem, k).values())


def paired_drop(base: dict[str, float], sft: dict[str, float]) -> dict[str, float]:
    """Base minus SFT, and the standard error of that difference over paired resamples."""
    ids = sorted(base)
    diffs = [base[p] - sft[p] for p in ids]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {"drop": statistics.mean(diffs), "se": statistics.stdev(draws)}


def training(run_dir: Path) -> dict[str, Any]:
    steps = merged_metrics(run_dir / "metrics")
    train = {s: d for s, d in steps.items() if "loss" in d}
    last = max(train)
    quarter = math.ceil(last / 4)
    first = statistics.mean(train[s]["loss"] for s in train if s <= quarter)
    final = statistics.mean(train[s]["loss"] for s in train if s > last - quarter)
    values = [d[k] for d in train.values() for k in ("loss", "grad_norm") if k in d]
    val = [(s, d["eval_loss"]) for s, d in steps.items() if "eval_loss" in d]
    if len(val) < 3:
        raise SystemExit(f"{run_dir}: {len(val)} validations, decision 029 needs 3")
    later = [v for _, v in val]
    rising = all(b > a for a, b in zip(later, later[1:], strict=False))
    non_finite = sum(not math.isfinite(v) for v in values)
    return {
        "steps": last,
        "non_finite_values": non_finite,
        "first_quarter_mean_loss": first,
        "last_quarter_mean_loss": final,
        "validation_loss": dict(val),
        "holds": non_finite > 0 or final >= first or rising,
    }


def check(base_dir: Path, sft_dir: Path, run_dir: Path) -> dict[str, Any]:
    base_checkpoint, base = read_eval(base_dir)
    sft_checkpoint, sft = read_eval(sft_dir)
    if set(base) != set(sft):
        raise SystemExit("base and SFT sampled different problems")
    base_summary, sft_summary = summary(base), summary(sft)
    collapse = {
        f"pass@{k}": paired_drop(per_problem(base, k), per_problem(sft, k)) for k in (1, SAMPLES)
    }
    for value in collapse.values():
        value["holds"] = value["drop"] > SE_MULTIPLE * value["se"]
    criteria: dict[int, dict[str, Any]] = {
        1: training(run_dir),
        2: {
            "well_formed_when_stopped": sft_summary["well_formed_when_stopped"],
            "threshold": FORMAT_MIN,
            "holds": sft_summary["well_formed_when_stopped"] < FORMAT_MIN,
        },
        3: {
            "sft": sft_summary["truncation_rate"],
            "base": base_summary["truncation_rate"],
            "holds": sft_summary["truncation_rate"] >= base_summary["truncation_rate"],
        },
        4: {**collapse, "holds": any(v["holds"] for v in collapse.values())},
    }
    triggered = [c for c, v in criteria.items() if v["holds"]]
    changes = sorted({CHANGES[c] for c in triggered})
    return {
        "base": {"checkpoint": base_checkpoint, **base_summary},
        "sft": {"checkpoint": sft_checkpoint, "run": run_dir.name, **sft_summary},
        "criteria": {str(c): v for c, v in criteria.items()},
        "triggered": triggered,
        "verdict": "run #2" if triggered else "no run #2: run #1's final export is frozen",
        "run2_changes": changes,
    }


class Run2CheckCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/sft_run2_check.py", cli_kebab_case=True
    )

    base: Path
    sft: Path
    run: Path
    report: Path

    def cli_cmd(self) -> None:
        result = check(self.base, self.sft, self.run)
        self.report.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    CliApp.run(Run2CheckCli)
