"""Fail a training run that is missing any mandatory metric (M1.6).

    python scripts/check_run_metrics.py --run-dir runs/<id>

Checks, for every trained step 1..N (N = the last step verl logged):
- verl's per-step metrics hold every key in VERL_KEYS (training, off-policy, behaviour, timing);
- `rollout_metrics.jsonl` holds every soe key (soe/trainer/rl/metrics.py);
- the step trained exactly prompts_per_step x group_size rollouts: the rollout-token and outcome
  records hold that many, and soe's reward mean and length mean equal verl's
  `critic/score/mean` and `response_length/mean`, so verl trained the same rollouts (decision 016:
  verl drops a failed rollout silently);
- `efficiency.json` covers step N with the recounted rollout tokens and non-zero hours;
- provenance: run.json, and every job segment's commit, hardware and library versions.
Prints each problem and exits 1 if there is any. Capability and transition metrics are checked
per evaluation by main_eval and scripts/transitions.py, not here.
"""

import math
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.rl.metrics import (
    OUTCOMES_DIR,
    SOE_KEYS,
    STEP_METRICS,
    outcomes_by_step,
    read_step_metrics,
)
from soe.trainer.rl.rollout_tokens import cumulative_rollout_tokens, tokens_by_step
from soe.trainer.rl.segments import merged_metrics, read_events
from soe.utils.run import read_efficiency, read_run_info

VERL_KEYS = (
    # Training
    "critic/score/mean",
    "actor/entropy",
    "actor/ppo_kl",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "actor/lr",
    # Rollout vs trainer (importance-ratio diagnostics, correction off)
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "rollout_corr/chi2_token",
    "rollout_corr/chi2_seq",
    "rollout_corr/log_ppl_diff_max",
    "rollout_corr/log_ppl_diff_min",
    "training/rollout_probs_diff_max",
    # Behaviour
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    # Efficiency
    "perf/total_num_tokens",
    "perf/throughput",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/update_weights",
)
# soe and verl compute these from the same rollouts; float32 reductions differ in the last bits.
CROSS_CHECKS = {"soe/reward/mean": "critic/score/mean", "soe/length/mean": "response_length/mean"}
RELATIVE_TOLERANCE = 1e-5


def check_run(run_dir: Path) -> list[str]:
    problems: list[str] = []
    config = load_config(run_dir / "config.yaml")
    if config.rollout is None:
        return [f"{config.name} is not an RL config"]
    per_step = config.rollout.prompts_per_step * config.rollout.group_size
    verl = {s: data for s, data in merged_metrics(run_dir / "metrics").items() if s >= 1}
    if not verl:
        return ["no verl metrics for any training step"]
    last = max(verl)
    soe = read_step_metrics(run_dir / STEP_METRICS)
    lengths = tokens_by_step(run_dir / "rollout_tokens")
    outcomes = outcomes_by_step(run_dir / OUTCOMES_DIR)
    for step in range(1, last + 1):
        logged, ours = verl.get(step, {}), soe.get(step, {})
        problems += [f"step {step}: verl metric {k} missing" for k in VERL_KEYS if k not in logged]
        problems += [f"step {step}: soe metric {k} missing" for k in SOE_KEYS if k not in ours]
        for name, records in (("rollout-token", lengths), ("outcome", outcomes)):
            count = len(records.get(step, []))
            if count != per_step:
                problems.append(f"step {step}: {count} {name} records, expected {per_step}")
        for mine, theirs in CROSS_CHECKS.items():
            if mine not in ours or theirs not in logged:
                continue  # reported as missing above
            if not math.isclose(ours[mine], logged[theirs], rel_tol=RELATIVE_TOLERANCE):
                problems.append(
                    f"step {step}: {mine} {ours[mine]} != verl {theirs} {logged[theirs]}"
                )
    efficiency = read_efficiency(run_dir)
    if efficiency.through_step != last:
        problems.append(f"efficiency.json covers step {efficiency.through_step}, not {last}")
    elif efficiency.rollout_tokens != cumulative_rollout_tokens(
        run_dir / "rollout_tokens", last, per_step
    ):
        problems.append("efficiency.json rollout_tokens differ from the recount")
    if efficiency.gpu_hours <= 0 or efficiency.wall_clock_hours <= 0:
        problems.append("efficiency.json has no GPU-hours or wall-clock hours")
    info = read_run_info(run_dir)
    if not (info.git_commit and info.config_sha256 and info.gpus and info.library_versions):
        problems.append("run.json lacks commit, config sha256, GPUs or library versions")
    starts = [e for e in read_events(run_dir) if e.event == "start"]
    if not starts:
        problems.append("segments.jsonl has no job segment")
    for event in starts:
        if not (event.git_commit and event.hardware and event.library_versions):
            problems.append(f"segment {event.segment} lacks commit, hardware or library versions")
    return problems


class CheckRunCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/check_run_metrics.py", cli_kebab_case=True
    )

    run_dir: Path

    def cli_cmd(self) -> None:
        problems = check_run(self.run_dir)
        for problem in problems:
            print(problem)
        print(f"{self.run_dir}: {'FAIL' if problems else 'ok'} ({len(problems)} problems)")
        if problems:
            sys.exit(1)


if __name__ == "__main__":
    CliApp.run(CheckRunCli)
