"""Markdown tables for the M1 report chapter, built only from committed reports.

    python scripts/report_m1_tables.py > /tmp/m1_tables.md
Each table is printed under a `<!-- table: name -->` marker; docs/report/m1.md embeds them.
"""

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPORTS = Path("docs/reports")
DEV_STEPS = (12, 24, 37)
BUDGETS_M = (30, 50, 100, 200)
# Job families counted in M0's account (m0_compute.json) that also fall in M1's window.
M0_FAMILIES = {"soe-eval-v3", "soe-eval-v2b", "soe-fp8-probe"}
# Slurm job family -> M1 task. `soe-train` is train.sbatch's default name, which the chained
# successor jobs of M1.5 (3222962) and M1.9 (3223952) kept, so Slurm cannot split it by task.
TASK_OF_FAMILY = {
    "soe-m1-smoke": "M1.3 verl GRPO smoke",
    "soe-m1-eval-smoke": "M1.4 export and eval hand-off",
    "soe-m1-identity-full": "M1.4 export and eval hand-off",
    "soe-m1-identity-lora": "M1.4 export and eval hand-off",
    "soe-m1-resume-control": "M1.5 kill-and-resume test",
    "soe-m1-resume-test": "M1.5 kill-and-resume test",
    "soe-m1-guard": "M1.5 kill-and-resume test",
    "soe-m1.6-smoke": "M1.6 metrics smoke",
    "soe-m1.8-smoke": "M1.8 SFT wiring",
    "soe-m1.8-sft": "M1.8 SFT wiring",
    "soe-m1.8-eval": "M1.8 SFT wiring",
    "soe-m1.9-dev": "M1.9 2B development run",
    "soe-m1.9-eval-base": "M1.9 dev evaluations",
    "soe-m1.9-eval-s12": "M1.9 dev evaluations",
    "soe-m1.9-eval-s24": "M1.9 dev evaluations",
    "soe-m1.9-eval-s37": "M1.9 dev evaluations",
    "soe-train": "chained successors of M1.5 and M1.9",
    "soe-m1.10-smoke": "M1.10 4B memory study",
    "soe-m1.10-smoke-bl": "M1.10 4B memory study",
    "soe-m1.10-probe-bf0": "M1.10 4B memory study",
    "soe-m1.10-probe-bl": "M1.10 4B memory study",
    "soe-m1.10-bf0": "M1.10 4B memory study",
    "soe-m1.10-bl": "M1.10 4B memory study",
    "soe-m1.11-bf0": "M1.11 throughput runs",
    "soe-m1.11-bl": "M1.11 throughput runs",
    "soe-gpu-probe": "checks",
    "soe-m1-tmpclean": "checks",
}


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((REPORTS / name).read_text())
    return data


def table(header: list[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def mean_sd(values: list[float], fmt: str = ",.0f") -> str:
    return f"{statistics.mean(values):{fmt}} ± {statistics.stdev(values):{fmt}}"


def dev_tables(out: dict[str, str]) -> None:
    evals = {"base": load("m1_dev_qwen35-2b_base.json")}
    evals |= {f"step {s}": load(f"m1_dev_qwen35-2b_step{s}.json") for s in DEV_STEPS}
    out["dev_eval"] = table(
        ["Level", "Checkpoint", "Problems", "pass@1", "pass@4", "pass@16", "maj@16", "truncated",
         "mean tokens"],
        [
            [
                lvl, name, int(r["problems"]), r["pass@1"], r["pass@4"], r["pass@16"],
                r["maj@16"], r["truncation_rate"], f"{r['mean_completion_tokens']:,.0f}",
            ]
            for lvl in ("A", "B")
            for name, rep in evals.items()
            for r in [rep["main"]["per_level"][lvl]]
        ],
    )  # fmt: skip
    rows: list[list[str]] = []
    for s in DEV_STEPS:
        cmp = load(f"m1_dev_qwen35-2b_compare_step{s}.json")
        for lvl in ("A", "B"):
            c = cmp[f"level {lvl}"]
            rows.append(
                [f"step {s}", lvl]
                + [f"{100 * c[m]['diff']:+.1f} ± {100 * c[m]['se']:.1f}" for m in
                   ("pass@1", "pass@16", "truncation_rate")]
                + [f"{c['mean_completion_tokens']['diff']:+,.0f}"]
            )  # fmt: skip
    out["dev_compare"] = table(
        ["Checkpoint", "Level", "pass@1 (points)", "pass@16 (points)", "truncated (points)",
         "mean tokens"],
        rows,
    )  # fmt: skip
    tr = load("m1_dev_qwen35-2b_transitions_step37.json")
    out["dev_transitions"] = table(
        ["Level", "k", "solved→solved", "unsolved→solved", "solved→lost", "unsolved→unsolved"],
        [
            [lvl, k, t["solved_to_solved"], t["unsolved_to_solved"], t["solved_to_lost"],
             t["unsolved_to_unsolved"]]
            for lvl in ("A", "B")
            for k in ("1", "16")
            for t in [tr["main"][lvl][k]]
        ],
    )  # fmt: skip
    acc = load("m1_dev_qwen35-2b_acceptance.json")["reward_trend"]
    out["dev_reward"] = table(
        ["Steps", "Window (steps)", "First-window reward", "Last-window reward", "Difference",
         "× its SE"],
        [[acc["steps"], acc["window_steps"], acc["first_mean"], acc["last_mean"],
          acc["difference"], f"{acc['se_multiple']:.1f}"]],
    )  # fmt: skip


def resume_table(out: dict[str, str]) -> None:
    r = load("m1_resume_test.json")
    first = r["first_step_after_resume"]
    step = first.pop("step")
    rows: list[list[object]] = [
        ["two job segments, same node and hardware",
         r["segments"]["two_segments"] and r["segments"]["same_node"]
         and r["segments"]["same_hardware"]],
        ["steps continuous (test, control)",
         r["steps_continuous"]["test"] and r["steps_continuous"]["control"]],
        ["dataloader position restored at every step", all(r["dataloader_restored"].values())],
        ["learning rate equal to the control at every step",
         r["lr_schedule_continues"]["equal_every_step"]],
        ["optimizer steps (test / control)",
         f"{r['optimizer_step_continues']['test']:.0f} / "
         f"{r['optimizer_step_continues']['control']:.0f}"],
        ["rollout tokens increasing; manifest equals a recount",
         r["rollout_tokens_continuous"]["increasing"]
         and r["rollout_tokens_continuous"]["manifest_equals_recount"]],
    ]  # fmt: skip
    for metric, m in first.items():
        rows.append(
            [f"step {step} `{metric}`: test − control vs the control's largest step change",
             f"{m['difference']:.4f} vs {m['control_max_step_change']:.4f} "
             f"({'within' if m['within'] else 'outside'})"]
        )  # fmt: skip
    out["resume"] = table(
        ["Check (2B; killed after step 8, resumed from the step-6 save)", "Result"], rows
    )


def memory_table(out: dict[str, str]) -> None:
    rows: list[list[object]] = []
    for cell in load("m1_memory_study.json")["cells"]:
        (m,) = cell["measured"]
        steps = [s for k, s in m["steps"].items() if int(k) >= 2]
        kv = m["kv_cache"]
        rows.append(
            [
                cell["cell"], m["training_mode"], m["rollout_tensor_parallel"],
                str(m["gpu_memory_utilization"]),
                mean_sd([s["rollout_tokens_per_s"] for s in steps]),
                mean_sd([s["step_s_without_save"] for s in steps]),
                mean_sd([s["update_actor_s"] for s in steps]),
                f"{max(m['nvidia_smi_peak']['by_phase_mib']['generate'].values()):,.0f}",
                f"{kv['kv_cache_tokens_per_replica'][0]:,}", f"{min(kv['headroom']):.2f}",
                f"{kv['peak_kv_cache_usage_pct']:.0f}%", kv["preemptions"],
                f"{m['host_ram_peak_gib']:.0f}",
            ]
        )  # fmt: skip
    out["memory"] = table(
        ["Cell", "Mode", "TP", "util", "rollout tok/s", "step s", "update s",
         "GPU peak, gen (MiB)", "KV tokens/replica", "headroom", "KV peak", "preemptions",
         "host RAM (GiB)"],
        rows,
    )  # fmt: skip


def throughput_tables(out: dict[str, str]) -> None:
    runs = load("m1_throughput.json")["runs"]
    out["throughput"] = table(
        ["Run", "Mode", "Steps", "Wall-clock h", "Rollout tokens", "R (tokens/h)",
         "steps / validation / other (h)", "mean length (run, base)"],
        [
            [
                r["config_name"], r["training_mode"], r["steps"], f"{r['wall_clock_hours']:.2f}",
                f"{r['rollout_tokens'] / 1e6:.1f}M",
                f"{r['rate_tokens_per_hour'] / 1e6:.2f}M ± {100 * r['rate_spread']:.1f}%",
                " / ".join(f"{r['time_split_hours'][k]:.2f}"
                           for k in ("steps", "validation", "other")),
                f"{r['run_response_length']:,.0f}, {r['base_response_length']:,.0f}",
            ]
            for r in runs
        ],
    )  # fmt: skip
    (bf0,) = [r for r in runs if r["training_mode"] == "full_bf16"]
    rows: list[list[object]] = []
    for b in bf0["budgets"]:
        tokens = b["rollout_tokens"]
        by_length = [
            next(x for x in s["budgets"] if x["rollout_tokens"] == tokens)["hours"]
            for s in bf0["length_sensitivity"]
        ]
        rows.append(
            [f"{tokens / 1e6:.0f}M", f"{b['hours']:.1f}",
             f"{b['hours_range'][0]:.1f}–{b['hours_range'][1]:.1f}", b["jobs"],
             f"{b['hours_with_queue_factor']:.1f}",
             f"{min(by_length):.1f}–{max(by_length):.1f}"]
        )  # fmt: skip
    out["budgets"] = table(
        ["Budget", "hours", "range (spread)", "jobs", "× 1.25", "at 0.5–1.5× length"], rows
    )


def schedule_tables(out: dict[str, str]) -> None:
    rates = load("m0_eval_gpu_rates.json")
    a40 = rates["by_gpu"]["A40"]["tokens_per_second"]
    out["eval_rates"] = table(
        ["GPU", "Shards", "Shard-hours", "Completion tokens", "tok/s", "relative to A40"],
        [
            [gpu, r["shards"], f"{r['shard_hours']:.1f}", f"{r['completion_tokens'] / 1e6:.1f}M",
             f"{r['tokens_per_second']:,.0f}", f"{r['tokens_per_second'] / a40:.2f}"]
            for gpu, r in sorted(rates["by_gpu"].items(), key=lambda x: -x[1]["tokens_per_second"])
        ],
    )  # fmt: skip
    sched = load("g0_schedule.json")
    pools = list(sched["inputs"]["lane_b_pools"])

    def span(lo: float, hi: float, fmt: str = ".1f") -> str:
        return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}}–{hi:{fmt}}"

    rows: list[list[object]] = []
    for m in sched["milestones"]:
        lo, hi = m["low"], m["high"]
        every = "lane_b_every_save"
        rows.append(
            [m["name"],
             span(lo["lane_a_hours_with_queue_factor"], hi["lane_a_hours_with_queue_factor"]),
             span(lo[every]["evaluations"], hi[every]["evaluations"], "d")]
            + [span(lo[every]["days"][p], hi[every]["days"][p]) for p in pools]
        )  # fmt: skip
    out["schedule"] = table(
        ["Milestone", "Lane A h (× 1.25)", "evaluations"] + [f"Lane B days, {p}" for p in pools],
        rows,
    )
    tot = sched["totals"]
    eq = sched["lane_b_pool_a40_equivalents"]
    out["schedule_totals"] = table(
        ["Lane B pool", "GPUs", "A40-equivalents", "Lane B days (every save / ≤ 4 per run)",
         "critical path days (every save / ≤ 4 per run)"],
        [
            [p, ", ".join(f"{n} {g}" for g, n in sched["inputs"]["lane_b_pools"][p].items()),
             f"{eq[p]:.2f}",
             " / ".join(span(tot["low"][pol]["lane_b_days"][p], tot["high"][pol]["lane_b_days"][p])
                        for pol in ("lane_b_every_save", "lane_b_capped")),
             " / ".join(span(tot["low"][pol]["critical_path_days"][p],
                             tot["high"][pol]["critical_path_days"][p])
                        for pol in ("lane_b_every_save", "lane_b_capped"))]
            for p in pools
        ]
        + [["Lane A (fc098), M2–M8", "4 A40", "",
            span(tot["low"]["lane_a_days_with_queue_factor"],
                 tot["high"]["lane_a_days_with_queue_factor"]) + " (× 1.25)", ""]],
    )  # fmt: skip


def compute_table(out: dict[str, str]) -> None:
    comp = load("m1_compute.json")
    tasks: dict[str, dict[str, float]] = {}
    for family, v in comp["families"].items():
        if family in M0_FAMILIES:
            continue
        task = TASK_OF_FAMILY[family]  # a new family must be assigned a task explicitly
        t = tasks.setdefault(task, {"jobs": 0.0, "gpu_hours": 0.0})
        t["jobs"] += v["jobs"]
        t["gpu_hours"] += v["gpu_hours"]
        for gpu, h in v["by_gpu"].items():
            t[gpu] = t.get(gpu, 0.0) + h
    order = list(dict.fromkeys(TASK_OF_FAMILY.values()))  # roadmap order
    rows: list[list[object]] = [
        [task, int(t["jobs"]), f"{t['gpu_hours']:.1f}",
         ", ".join(f"{g} {t[g]:.1f}" for g in ("a40", "a30", "a10") if g in t)]
        for task, t in sorted(tasks.items(), key=lambda x: order.index(x[0]))
    ]  # fmt: skip
    total = sum(t["gpu_hours"] for t in tasks.values())
    rows.append(["total", "", f"{total:.1f}", ""])
    out["compute"] = table(["Task", "Jobs", "GPU-hours", "By GPU type"], rows)


def main() -> None:
    out: dict[str, str] = {}
    resume_table(out)
    dev_tables(out)
    memory_table(out)
    throughput_tables(out)
    schedule_tables(out)
    compute_table(out)
    for name, text in out.items():
        print(f"<!-- table: {name} -->\n{text}\n")


if __name__ == "__main__":
    main()
