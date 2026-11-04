"""M2.8 4B SFT memory and timing probe report (docs/reports/m2_sft_probe_4b.json).

    python scripts/sft_probe_report.py --run runs/<probe run> \\
        --report docs/reports/m2_sft_probe_4b.json

From one SFT run of a single segment (soe/trainer/sft/trainer.py's logs):
- per optimizer step: tokens and loss tokens (example logs, summed over ranks and micro-batches)
  and seconds (wall-time difference of consecutive log lines, minus a save in between); step 1
  also holds start-up and warm-up, so the rates use steps 2 onward;
- tokens/h and loss tokens/h, against the G0 schedule's assumed 24.06M tokens/h;
- save seconds, validation seconds, export seconds (segment end minus the last log line) and the
  time from the segment start to the end of step 1;
- peak memory: rank 0's torch peaks, nvidia-smi per GPU (the free margin against 4 GiB), and the
  node's host RAM in use (/proc/meminfo: other jobs on the node count too);
- run #1 at the same rate: steps, hours, and `save_every_steps` for a save about every 2, 3 or 4 h.
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.rl.segments import read_events

G0_TOKENS_PER_HOUR = 24.06e6  # decision 025's SFT rate, borrowed from the RL update rate
MIN_FREE_GIB = 4.0  # below this, the response is a decision-028 note before run #1 (roadmap M2.8)
SAVE_EVERY_HOURS = (2, 3, 4)


def step_tokens(run_dir: Path) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = defaultdict(lambda: {"tokens": 0, "loss_tokens": 0})
    for path in sorted((run_dir / "examples").glob("segment-*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            out[row["step"]]["tokens"] += row["tokens"]
            out[row["step"]]["loss_tokens"] += row["loss_tokens"]
    return dict(sorted(out.items()))


def metric_lines(run_dir: Path) -> list[dict[str, Any]]:
    (path,) = (run_dir / "metrics").glob("segment-*.jsonl")  # a probe has one segment
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def step_seconds(lines: list[dict[str, Any]]) -> dict[int, float]:
    """Seconds of each training step after the first: from the previous training log line to its
    own, minus a save logged in between."""
    out: dict[int, float] = {}
    previous: float | None = None
    save = 0.0
    for line in lines:
        data = line["data"]
        if "save_seconds" in data:
            save += data["save_seconds"]
        elif "loss" in data:
            if previous is not None:
                out[line["step"]] = data["wall_time"] - previous - save
            previous, save = data["wall_time"], 0.0
    return out


def gpu_peaks(run_dir: Path) -> dict[str, dict[str, float]]:
    peaks: dict[str, dict[str, float]] = {}
    for path in (run_dir / "gpu_memory").glob("*.csv"):
        for row in csv.reader(path.open()):
            index, used, total = row[1].strip(), float(row[2]), float(row[3])
            peak = peaks.setdefault(index, {"peak_used_gib": 0.0, "total_gib": total / 1024})
            peak["peak_used_gib"] = max(peak["peak_used_gib"], used / 1024)
    for peak in peaks.values():
        peak["free_at_peak_gib"] = peak["total_gib"] - peak["peak_used_gib"]
    return dict(sorted(peaks.items()))


def host_memory(run_dir: Path) -> dict[str, float]:
    used = [
        (float(row[1]) - float(row[2])) / 2**20  # kB -> GiB
        for path in (run_dir / "host_memory").glob("*.csv")
        for row in csv.reader(path.open())
    ]
    return {"node_used_at_start_gib": used[0], "node_peak_used_gib": max(used)}


def run1_estimate(
    rate: float, step_mean: float, overhead: dict[str, float], train: dict[str, int], batch: int
) -> dict[str, Any]:
    steps = math.ceil(train["examples"] / batch)
    train_hours = train["tokens"] / rate
    save_every = {f"{h}h": max(1, round(h * 3600 / step_mean)) for h in SAVE_EVERY_HOURS}
    saves = steps // save_every["3h"] + 1  # and the final save
    fixed = overhead["start_to_step1"] + overhead["validation"] + overhead["export"]
    return {
        "steps": steps,
        "train_tokens": train["tokens"],
        "training_hours": train_hours,
        "save_every_steps_for": save_every,
        "hours_with_3h_saves": train_hours + (saves * overhead["save"] + fixed) / 3600,
        "note": "same rate as the probe's timed steps; start-up, validation and export once",
    }


def report(run_dir: Path) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    assert config.sft is not None
    lines = metric_lines(run_dir)
    tokens, seconds = step_tokens(run_dir), step_seconds(lines)
    timed = sorted(seconds)
    total_s = sum(seconds[s] for s in timed)
    rate = sum(tokens[s]["tokens"] for s in timed) / total_s * 3600
    loss_rate = sum(tokens[s]["loss_tokens"] for s in timed) / total_s * 3600
    events = read_events(run_dir)
    start = next(e.at for e in events if e.event == "start").timestamp()
    end = next(e.at for e in events if e.event == "end").timestamp()
    training = [x["data"] for x in lines if "loss" in x["data"]]
    saves = [x["data"]["save_seconds"] for x in lines if "save_seconds" in x["data"]]
    validation = next(x["data"]["eval_runtime"] for x in lines if "eval_runtime" in x["data"])
    overhead = {
        "start_to_step1": training[0]["wall_time"] - start,
        "save": statistics.mean(saves),
        "validation": validation,
        "export": end - lines[-1]["data"]["wall_time"],
    }
    gpus = gpu_peaks(run_dir)
    batch = (
        config.sft.n_gpus * config.sft.batch_size_per_gpu * config.sft.gradient_accumulation_steps
    )
    train_counts = json.loads((run_dir / "sft_data.json").read_text())["train"]
    return {
        "run": run_dir.name,
        "commit": events[0].git_commit,
        "node": events[0].node,
        "hardware": events[0].hardware,
        "sequences_per_step": batch,
        "steps": {
            s: {**tokens[s], "seconds": seconds.get(s)} for s in sorted(tokens) if s in seconds
        },
        "timed_steps": [timed[0], timed[-1]],
        "step_seconds": {
            "mean": statistics.mean(seconds.values()),
            "sd": statistics.stdev(seconds.values()),
        },
        "tokens_per_hour": rate,
        "loss_tokens_per_hour": loss_rate,
        "vs_g0_rate": rate / G0_TOKENS_PER_HOUR,
        "seconds": {**overhead, "saves": saves},
        "memory": {
            "torch_rank0_peak_allocated_gib": max(x["gpu_peak_allocated_gib"] for x in training),
            "torch_rank0_peak_reserved_gib": max(x["gpu_peak_reserved_gib"] for x in training),
            "nvidia_smi": gpus,
            "min_free_gib": min(g["free_at_peak_gib"] for g in gpus.values()),
            "below_free_margin": any(g["free_at_peak_gib"] < MIN_FREE_GIB for g in gpus.values()),
            "host": host_memory(run_dir),
        },
        "run1_estimate": run1_estimate(
            rate, statistics.mean(seconds.values()), overhead, train_counts, batch
        ),
    }


class SftProbeReportCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/sft_probe_report.py", cli_kebab_case=True
    )

    run: Path
    report: Path

    def cli_cmd(self) -> None:
        result = report(self.run)
        self.report.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=1))


if __name__ == "__main__":
    CliApp.run(SftProbeReportCli)
