"""M1.10 memory study report: every probe and measured cell run (docs/reports/m1_memory_study.json).

    python scripts/memory_study_report.py --runs runs/<probe or cell run> --runs ... \\
        --out docs/reports/m1_memory_study.json

Run directories are grouped into cells by config name (a probe is `<cell name>-probe-tpN-gmuNN`).
For each run: its config knobs, the exact verl override list, commit, node and GPUs; whether it
finished, and for a failed run the phase that was running and the out-of-memory message; per step:
rollout tokens/s (response tokens / timing_s/gen), training tokens/s ((prompt + response) tokens /
(old_log_prob + update_actor)), the step time split, and rollout tokens per wall-clock hour; per
phase and GPU: torch peaks (phase_memory marks, soe/workers/rollout/phase_memory.py) and the
nvidia-smi peak inside the phase's window (sidecar, slurm/common.sh); the host RAM peak; vLLM's KV
cache size, maximum concurrency, peak KV-cache usage and preemptions from the run's log.
Measured cells report steps 2-5 as mean and standard deviation (step 1 warms up); a probe reports
its last step, the first with AdamW's state resident (scripts/memory_probe_config.py). For each
cell, the fastest probe that finished is the chosen setting, and grid points below a finished
probe's gpu_memory_utilization at the same TP are "not run, dominated" (user, 2026-10-31).
"""

import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.config.experiment import ExperimentConfig, load_config, rollout_protocol
from soe.trainer.rl.rollout_tokens import tokens_by_step
from soe.trainer.rl.segments import merged_metrics, read_events
from soe.workers.rollout.phase_memory import PHASE_MEMORY_DIR, PhaseMark

CELLS = {
    "m1_memory_bf0_qwen35-4b": "B-F0",
    "m1_memory_bf1_qwen35-4b": "B-F1",
    "m1_memory_bl_qwen35-4b": "B-L",
}
GRID_TP = (1, 2)
GRID_GMU = (0.4, 0.5, 0.6, 0.7)
MEASURED_STEPS = (2, 3, 4, 5)
PROBE = re.compile(r"^(?P<cell>.+)-probe-tp(?P<tp>\d+)-gmu(?P<gmu>\d+)$")
# Marks that close each phase (phase_memory.py): generation ends where the old log-prob pass starts.
PHASES = {
    "update_weights": "weight_sync",
    "before_compute_log_prob": "generate",
    "compute_log_prob": "old_log_prob",
    "update_actor": "update",
}
TIMERS = ("gen", "reward", "old_log_prob", "adv", "update_actor", "update_weights")
KV_SIZE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
KV_MEMORY = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
KV_USAGE = re.compile(r"GPU KV cache usage: ([\d.]+)%")
PREEMPTIONS = re.compile(r"Preemptions: (\d+)")
OOM = re.compile(r"(OutOfMemoryError|CUDA out of memory|Free memory on device|Tried to allocate)")


def mean_sd(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def kv_cache(log_text: str) -> dict[str, Any]:
    """vLLM's start-up KV-cache lines (one per replica) and its periodic stats lines."""
    usage = [float(x) for x in KV_USAGE.findall(log_text)]
    return {
        "kv_cache_tokens_per_replica": [int(x.replace(",", "")) for x in KV_SIZE.findall(log_text)],
        "max_concurrency_per_replica": [float(c) for _, c in CONCURRENCY.findall(log_text)],
        "kv_cache_gib_per_replica": [float(x) for x in KV_MEMORY.findall(log_text)],
        "peak_kv_cache_usage_pct": max(usage, default=None),
        # vLLM 0.24 prints the preemptions of each stats interval, so they add up.
        "preemptions": sum(int(x) for x in PREEMPTIONS.findall(log_text)),
    }


def oom_lines(log_text: str, limit: int = 5) -> list[str]:
    return [line.strip()[:400] for line in log_text.splitlines() if OOM.search(line)][:limit]


def read_marks(run_dir: Path) -> list[PhaseMark]:
    marks = []
    for path in sorted((run_dir / PHASE_MEMORY_DIR).glob("*.jsonl")):
        marks += [PhaseMark.model_validate_json(x) for x in path.read_text().splitlines() if x]
    return sorted(marks, key=lambda m: (m.rank, m.ended_at))


def phase_windows(marks: list[PhaseMark]) -> list[dict[str, Any]]:
    """Per rank, each mark's window (from the previous mark) with the training step it belongs
    to: step k runs from the end of update k-1 to the end of update k."""
    windows: list[dict[str, Any]] = []
    for rank in sorted({m.rank for m in marks}):
        previous, step = None, 1
        for mark in (m for m in marks if m.rank == rank):
            if mark.phase in PHASES and previous is not None:
                windows.append(
                    {
                        "rank": rank,
                        "step": step,
                        "phase": PHASES[mark.phase],
                        "start": previous,
                        "end": mark.ended_at,
                        "mark": mark,
                    }
                )
            if mark.phase == "update_actor":
                step += 1
            previous = mark.ended_at
    return windows


def read_samples(path: Path, columns: int) -> list[list[str]]:
    if not path.exists():
        return []
    # A sidecar killed mid-write leaves a short last line.
    return [row for row in csv.reader(path.read_text().splitlines()) if len(row) == columns]


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def gpu_peaks(run_dir: Path, windows: list[dict[str, Any]], steps: set[int]) -> dict[str, Any]:
    """Max nvidia-smi memory.used (MiB) per phase and GPU index, inside rank 0's windows."""
    samples: list[tuple[datetime, int, float]] = []
    for path in sorted((run_dir / "gpu_memory").glob("*.csv")):
        for row in read_samples(path, 5):
            samples.append((parse_utc(row[0]), int(row[1]), float(row[2])))
    peaks: dict[str, dict[int, float]] = defaultdict(dict)
    for window in (w for w in windows if w["rank"] == 0 and w["step"] in steps):
        for when, index, used in samples:
            if window["start"] <= when <= window["end"]:
                phase = peaks[window["phase"]]
                phase[index] = max(phase.get(index, 0.0), used)
    whole_run: dict[int, float] = {}
    for _when, index, used in samples:
        whole_run[index] = max(whole_run.get(index, 0.0), used)
    return {
        "by_phase_mib": {p: dict(sorted(v.items())) for p, v in peaks.items()},
        "whole_run_mib": dict(sorted(whole_run.items())),
        "samples": len(samples),
    }


def host_ram_peak_gib(run_dir: Path) -> float | None:
    used = []
    for path in sorted((run_dir / "host_memory").glob("*.csv")):
        # /proc/meminfo is in KiB: used = MemTotal - MemAvailable
        used += [(float(row[1]) - float(row[2])) / 1024**2 for row in read_samples(path, 3)]
    return max(used, default=None)


def torch_peaks(windows: list[dict[str, Any]], steps: set[int]) -> dict[str, Any]:
    """Max over the steps of torch's peaks per phase and rank, and what stays resident."""
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for w in (w for w in windows if w["step"] in steps):
        mark: PhaseMark = w["mark"]
        seen = out[w["phase"]].setdefault(
            w["rank"],
            {"max_allocated_gib": 0.0, "max_reserved_gib": 0.0, "allocated_at_end_gib": 0.0},
        )
        seen["max_allocated_gib"] = max(seen["max_allocated_gib"], mark.max_allocated_gib)
        seen["max_reserved_gib"] = max(seen["max_reserved_gib"], mark.max_reserved_gib)
        seen["allocated_at_end_gib"] = max(seen["allocated_at_end_gib"], mark.allocated_gib)
    return {phase: dict(sorted(ranks.items())) for phase, ranks in out.items()}


def step_rates(
    config: ExperimentConfig, metrics: dict[int, dict[str, Any]], tokens: dict[int, list[int]]
) -> dict[int, dict[str, float]]:
    assert config.rollout is not None and config.hardware is not None
    responses = config.rollout.prompts_per_step * config.rollout.group_size
    gpus = config.hardware.n_gpus
    rates = {}
    for step, data in metrics.items():
        if step < 1 or "timing_s/gen" not in data or step not in tokens:
            continue
        response_tokens = sum(tokens[step])
        trained = response_tokens + data["prompt_length/mean"] * responses
        training_s = data["timing_s/old_log_prob"] + data["timing_s/update_actor"]
        extra = data.get("timing_s/testing", 0.0) + data.get("timing_s/save_checkpoint", 0.0)
        step_s = data["timing_s/step"] - extra  # validation and the save are reported apart
        rates[step] = {
            "rollout_tokens": response_tokens,
            "training_tokens": trained,
            "rollout_tokens_per_s": response_tokens / data["timing_s/gen"],
            "rollout_tokens_per_s_per_gpu": response_tokens / data["timing_s/gen"] / gpus,
            "training_tokens_per_s": trained / training_s,
            "training_tokens_per_s_per_gpu": trained / training_s / gpus,
            "step_s_without_validation_and_save": step_s,
            "rollout_tokens_per_hour": response_tokens / step_s * 3600,
            **{f"{t}_s": data.get(f"timing_s/{t}", 0.0) for t in TIMERS},
            "validation_s": data.get("timing_s/testing", 0.0),
            "save_checkpoint_s": data.get("timing_s/save_checkpoint", 0.0),
            "response_length_mean": data["response_length/mean"],
        }
    return rates


def failed_phase(marks: list[PhaseMark]) -> str:
    """The phase that was running when a failed run stopped: the one after rank 0's last mark."""
    last = [m for m in marks if m.rank == 0]
    if not last:
        return "before the first mark (start-up, model load or first weight sync)"
    after = {
        "update_weights": "generate",
        "before_compute_log_prob": "old_log_prob",
        "compute_log_prob": "update (or advantage)",
        "update_actor": "weight_sync or save",
        "before_update_weights": "weight_sync",
        "before_update_actor": "update",
    }
    return after.get(last[-1].phase, f"after {last[-1].phase}")


def run_report(run_dir: Path) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    assert config.hardware is not None and config.rollout is not None
    hardware = config.hardware
    probe = PROBE.match(config.name)
    log_text = "\n".join(p.read_text(errors="replace") for p in sorted(run_dir.glob("logs/*.log")))
    metrics = merged_metrics(run_dir / "metrics")
    tokens = tokens_by_step(run_dir / "rollout_tokens")
    marks = read_marks(run_dir)
    windows = phase_windows(marks)
    done = (run_dir / "DONE").exists()
    logged = sorted(s for s in metrics if s >= 1)
    steps = ({logged[-1]} if logged else set()) if probe else set(MEASURED_STEPS)
    rates = {s: r for s, r in step_rates(config, metrics, tokens).items() if s in steps}
    starts = [e for e in read_events(run_dir) if e.event == "start"]
    replicas = hardware.n_gpus // hardware.rollout_tensor_parallel
    kv = kv_cache(log_text)
    max_response = rollout_protocol(config, load_suite(Path(config.eval_suite))).max_response_tokens
    max_model_len = config.rollout.max_prompt_tokens + max_response
    responses_per_replica = config.rollout.prompts_per_step * config.rollout.group_size // replicas
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "config_name": config.name,
        "kind": "probe" if probe else "measured",
        "training_mode": config.training_mode.value,
        "rollout_tensor_parallel": hardware.rollout_tensor_parallel,
        "gpu_memory_utilization": hardware.gpu_memory_utilization,
        "param_offload": hardware.param_offload,
        "optimizer_offload": hardware.optimizer_offload,
        "fused_log_prob": hardware.fused_log_prob,
        "max_token_len_per_gpu": hardware.max_token_len_per_gpu,
        "finished": done,
        "commit": starts[0].git_commit if starts else None,
        "node": starts[0].node if starts else None,
        "gpus": (run_dir / "hardware.txt").read_text().strip()
        if (run_dir / "hardware.txt").exists()
        else None,
        "verl_overrides": json.loads((run_dir / "verl_overrides.json").read_text())
        if (run_dir / "verl_overrides.json").exists()
        else None,
        "steps": rates,
        "torch_peak_by_phase": torch_peaks(windows, steps),
        "nvidia_smi_peak": gpu_peaks(run_dir, windows, steps),
        "host_ram_peak_gib": host_ram_peak_gib(run_dir),
        "kv_cache": kv
        | {
            "replicas": replicas,
            "responses_per_replica": responses_per_replica,
            "max_model_len": max_model_len,
            # KV tokens / (responses per replica x (prompt + response cap)), per replica
            "headroom": [
                n / (responses_per_replica * max_model_len)
                for n in kv["kv_cache_tokens_per_replica"]
            ],
        },
    }
    if not probe and rates:
        keys = [k for k, v in rates[min(rates)].items() if isinstance(v, float)]
        report["measured_steps"] = {k: mean_sd([rates[s][k] for s in sorted(rates)]) for k in keys}
    if not done:
        report["failure"] = {
            "phase": failed_phase(marks),
            "out_of_memory_lines": oom_lines(log_text),
        }
    return report


def probe_speed(run: dict[str, Any]) -> float:
    (step,) = run["steps"].values()  # a probe reports only its last step
    return float(step["rollout_tokens_per_s"])


def cell_report(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    probes = [r for r in runs if r["kind"] == "probe"]
    fits = [p for p in probes if p["finished"] and p["steps"]]
    chosen = max(fits, key=probe_speed, default=None)
    grid = []
    for tp in GRID_TP:
        best_fit = max(
            (p["gpu_memory_utilization"] for p in fits if p["rollout_tensor_parallel"] == tp),
            default=None,
        )
        for gmu in sorted(GRID_GMU, reverse=True):
            run = next(
                (
                    p
                    for p in probes
                    if p["rollout_tensor_parallel"] == tp and p["gpu_memory_utilization"] == gmu
                ),
                None,
            )
            if run is not None:
                status = "fits" if run["finished"] else "failed"
            elif best_fit is not None and gmu < best_fit:
                status = "not run, dominated"
            else:
                status = "not run"
            grid.append({"tp": tp, "gpu_memory_utilization": gmu, "status": status})
    return {
        "cell": CELLS.get(name, name),
        "config_name": name,
        "grid": grid,
        "chosen": None
        if chosen is None
        else {
            "rollout_tensor_parallel": chosen["rollout_tensor_parallel"],
            "gpu_memory_utilization": chosen["gpu_memory_utilization"],
            "probe_rollout_tokens_per_s": probe_speed(chosen),
        },
        "probes": probes,
        "measured": [r for r in runs if r["kind"] == "measured"],
    }


def study_report(run_dirs: list[Path]) -> dict[str, Any]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_dir in run_dirs:
        run = run_report(run_dir)
        probe = PROBE.match(run["config_name"])
        by_cell[probe["cell"] if probe else run["config_name"]].append(run)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "measured_steps": list(MEASURED_STEPS),
        "cells": [cell_report(name, runs) for name, runs in sorted(by_cell.items())],
    }


class MemoryStudyCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/memory_study_report.py", cli_kebab_case=True
    )

    runs: list[Path]
    out: Path

    def cli_cmd(self) -> None:
        report = study_report(self.runs)
        self.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        for cell in report["cells"]:
            print(cell["cell"], "chosen:", cell["chosen"])
            for point in cell["grid"]:
                print(f"  tp{point['tp']} gmu {point['gpu_memory_utilization']}: {point['status']}")


if __name__ == "__main__":
    CliApp.run(MemoryStudyCli)
