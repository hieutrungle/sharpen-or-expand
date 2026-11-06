"""M2.11 diagnostic: where SFT run #1 differs from the base model (user, 2026-11-05).

    E=$SOE_DATA/eval; python scripts/sft_diagnostic.py \\
        --pair rl_val:$E/m2.11-base-rl-val:$E/m2.11-sft-run1-rl-val \\
        --pair sft_val:$E/m2.11diag-base-sft-val:$E/m2.11diag-sft-run1-sft-val \\
        --meta $SOE_DATA/datasets/soe/v1/sft/pool_meta.jsonl \\
        --report docs/reports/m2_sft_diagnostic.json --table docs/reports/m2_sft_diagnostic.csv

Each `--pair` is `split:base dir:SFT dir`, the base model's and the SFT checkpoint's main_eval
outputs on the same problems. Per split and model: pass@1 and pass@n (n samples per problem,
the unbiased estimator), truncation and stop rates, accuracy when stopped, the share of samples
whose last 4,000 characters compress below 15% (a heuristic for repetition loops), and lengths.
Then SFT minus base, paired by
problem, with the standard error over 2,000 bootstrap resamples of the problems (seed 0), and the
pass@1 change split into a stopping part and an accuracy-when-stopped part. Problems with a
difficulty in `--meta` (DeepMath) are also broken down by difficulty. The table has one row per
split and problem. Outside the run-#2 rule; nothing here chooses anything. Ids and numbers only.
"""

import csv
import json
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl
from soe.utils.metric.pass_at_k import pass_at_k
from soe.utils.shards import shard_path

TAIL_CHARS, LOOP_RATIO = 4000, 0.15
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 0
DIFFICULTY_BINS = (("<=5", 0.0, 5.0), ("5.5-6.5", 5.5, 6.5), (">=7", 7.0, 10.0))


def read_eval(directory: Path) -> tuple[str, str, dict[str, list[SampleRecord]]]:
    """Suite name, checkpoint and samples by problem; refuses an incomplete evaluation."""
    plan = json.loads((directory / "plan.json").read_text())
    by_problem: dict[str, list[SampleRecord]] = defaultdict(list)
    for k in range(plan["num_shards"]):
        for record in read_jsonl(shard_path(directory, k, plan["num_shards"]), SampleRecord):
            by_problem[record.problem_id].append(record)
    counts = {len(rs) for rs in by_problem.values()}
    if len(counts) != 1:
        raise SystemExit(f"{directory}: problems have different sample counts {sorted(counts)}")
    return plan["suite_name"], plan["checkpoint"], dict(by_problem)


def loops(record: SampleRecord) -> bool:
    tail = record.completion[-TAIL_CHARS:].encode()
    return len(zlib.compress(tail)) / max(1, len(tail)) < LOOP_RATIO


def per_problem(records: list[SampleRecord]) -> dict[str, float]:
    n = len(records)
    correct = sum(r.correct for r in records)
    return {
        "n": n,
        "correct": correct,
        "pass@1": correct / n,
        "pass@n": pass_at_k(n, correct, n),
        "truncated": sum(r.finish_reason == "length" for r in records) / n,
        "loops": sum(map(loops, records)) / n,
    }


def summary(by_problem: dict[str, list[SampleRecord]]) -> dict[str, float]:
    records = [r for rs in by_problem.values() for r in rs]
    n = len(next(iter(by_problem.values())))
    stopped = [r for r in records if r.finish_reason == "stop"]
    truncated = [r for r in records if r.finish_reason == "length"]
    return {
        "problems": len(by_problem),
        "samples_per_problem": n,
        "pass@1": sum(r.correct for r in records) / len(records),
        f"pass@{n}": statistics.mean(
            pass_at_k(n, sum(r.correct for r in rs), n) for rs in by_problem.values()
        ),
        "truncation_rate": len(truncated) / len(records),
        "stop_rate": len(stopped) / len(records),
        "accuracy_when_stopped": sum(r.correct for r in stopped) / max(1, len(stopped)),
        "loop_rate": sum(map(loops, records)) / len(records),
        "loops_among_truncated": sum(map(loops, truncated)) / max(1, len(truncated)),
        "mean_completion_tokens": statistics.mean(r.completion_tokens for r in records),
        "median_stopped_tokens": statistics.median(r.completion_tokens for r in stopped)
        if stopped
        else 0.0,
    }


def paired(base: list[float], sft: list[float]) -> dict[str, float]:
    """Mean SFT minus base over problems and its bootstrap standard error."""
    diffs = [s - b for b, s in zip(base, sft, strict=True)]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {"sft_minus_base": statistics.mean(diffs), "se": statistics.stdev(draws)}


def decompose(base: dict[str, float], sft: dict[str, float]) -> dict[str, float]:
    """pass@1 = stop rate x accuracy when stopped (truncation scores 0), so its change splits
    exactly into a stopping part and an accuracy part."""
    stopping = (sft["stop_rate"] - base["stop_rate"]) * base["accuracy_when_stopped"]
    accuracy = sft["stop_rate"] * (sft["accuracy_when_stopped"] - base["accuracy_when_stopped"])
    return {"total": sft["pass@1"] - base["pass@1"], "stopping": stopping, "accuracy": accuracy}


def difficulty_bin(value: float) -> str | None:
    for name, low, high in DIFFICULTY_BINS:
        if low <= value <= high:
            return name
    return None


def compare(
    base: dict[str, list[SampleRecord]],
    sft: dict[str, list[SampleRecord]],
    difficulty: dict[str, float],
) -> dict[str, Any]:
    if set(base) != set(sft):
        raise SystemExit("base and SFT sampled different problems")
    ids = sorted(base)
    rows_b = {p: per_problem(base[p]) for p in ids}
    rows_s = {p: per_problem(sft[p]) for p in ids}
    metrics = ("pass@1", "pass@n", "truncated", "loops")
    out: dict[str, Any] = {"base": summary(base), "sft": summary(sft)}
    out["paired"] = {
        m: paired([rows_b[p][m] for p in ids], [rows_s[p][m] for p in ids]) for m in metrics
    }
    out["pass@1_change"] = decompose(out["base"], out["sft"])
    bins: dict[str, list[str]] = defaultdict(list)
    for p in ids:
        if p in difficulty and (name := difficulty_bin(difficulty[p])) is not None:
            bins[name].append(p)
    out["by_difficulty"] = {
        name: {
            "problems": len(bins[name]),
            "base": summary({p: base[p] for p in bins[name]}),
            "sft": summary({p: sft[p] for p in bins[name]}),
            "paired_pass@1": paired(
                [rows_b[p]["pass@1"] for p in bins[name]], [rows_s[p]["pass@1"] for p in bins[name]]
            ),
        }
        for name, _, _ in DIFFICULTY_BINS
        if bins[name]
    }
    out["rows"] = [
        {"problem_id": p, "n": rows_b[p]["n"], "difficulty": difficulty.get(p)}
        | {f"base_{m}": rows_b[p][m] for m in metrics}
        | {f"sft_{m}": rows_s[p][m] for m in metrics}
        for p in ids
    ]
    return out


def read_difficulty(meta: Path | None) -> dict[str, float]:
    if meta is None:
        return {}
    rows = (json.loads(line) for line in meta.read_text().splitlines() if line.strip())
    return {
        r["problem_id"]: float(r["difficulty"]) for r in rows if r.get("difficulty") is not None
    }


class SftDiagnosticCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/sft_diagnostic.py", cli_kebab_case=True
    )

    pair: list[str]  # "split:base eval dir:SFT eval dir"
    meta: Path | None = None
    report: Path
    table: Path

    def cli_cmd(self) -> None:
        difficulty = read_difficulty(self.meta)
        report: dict[str, Any] = {}
        table: list[dict[str, Any]] = []
        for item in self.pair:
            split, base_dir, sft_dir = item.split(":")
            base_suite, base_ckpt, base = read_eval(Path(base_dir))
            sft_suite, sft_ckpt, sft = read_eval(Path(sft_dir))
            if base_suite != sft_suite:
                raise SystemExit(f"{split}: suites differ ({base_suite} vs {sft_suite})")
            result = compare(base, sft, difficulty)
            table += [{"split": split} | row for row in result.pop("rows")]
            report[split] = {
                "suite": base_suite,
                "base_checkpoint": base_ckpt,
                "sft_checkpoint": sft_ckpt,
            } | result
        self.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        with self.table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        for split, r in report.items():
            print(split, json.dumps({k: r[k] for k in ("base", "sft", "pass@1_change")}))


if __name__ == "__main__":
    CliApp.run(SftDiagnosticCli)
