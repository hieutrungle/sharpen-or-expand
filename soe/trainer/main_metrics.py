"""soe per-step metrics and run efficiency: `python -m soe.trainer.main_metrics --run-dir runs/<id>
--through-step N`.

Started in the background by the checkpoint callback after each save, and run by main_rl when a job
segment ends. Recomputes `runs/<id>/rollout_metrics.jsonl` for steps 1..N from the rollout records
(soe/trainer/rl/metrics.py) and rewrites `efficiency.json`. A process of its own, because
math-verify's time limits (answer equivalence) need a main thread.

Efficiency covers the steps verl has logged (at a save, verl logs that step's metrics only after
the callback returns): rollout tokens (decision 018), training tokens (verl's
`perf/total_num_tokens`, prompt + response tokens of the trained batch), wall-clock hours of all
job segments, and GPU-hours = wall-clock x the run's GPU count. `newly_solved_heldout` is kept as
scripts/transitions.py set it.
"""

import fcntl
from datetime import UTC, datetime
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.rl.metrics import (
    OUTCOMES_DIR,
    STEP_METRICS,
    outcomes_by_step,
    step_metrics,
    write_step_metrics,
)
from soe.trainer.rl.rollout_tokens import cumulative_rollout_tokens, tokens_by_step
from soe.trainer.rl.segments import merged_metrics, training_hours
from soe.utils.reward_score.math_verify import answers_equivalent
from soe.utils.run import RunEfficiency, read_efficiency, read_run_info, write_efficiency


def efficiency(run_dir: Path, through_step: int, responses_per_step: int) -> RunEfficiency:
    logged = merged_metrics(run_dir / "metrics")
    last = max((s for s in logged if 1 <= s <= through_step), default=0)
    hours = training_hours(run_dir, datetime.now(UTC))
    return RunEfficiency(
        through_step=last,
        rollout_tokens=(
            cumulative_rollout_tokens(run_dir / "rollout_tokens", last, responses_per_step)
            if last
            else 0
        ),
        training_tokens=sum(int(logged[s]["perf/total_num_tokens"]) for s in range(1, last + 1)),
        gpu_hours=hours * len(read_run_info(run_dir).gpus),
        wall_clock_hours=hours,
        newly_solved_heldout=read_efficiency(run_dir).newly_solved_heldout,
    )


class MetricsCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_metrics",
        env_prefix="SOE_METRICS_",
        cli_kebab_case=True,
    )

    run_dir: Path
    through_step: int

    def cli_cmd(self) -> None:
        run_dir = self.run_dir.resolve()
        config = load_config(run_dir / "config.yaml")
        if config.rollout is None:
            raise ValueError(f"{config.name} is not an RL config")
        prompts, group = config.rollout.prompts_per_step, config.rollout.group_size
        # The callback's background run and main_rl's final run may overlap; under the lock the
        # later one also reads later, so the files only move forward.
        with open(run_dir / ".metrics.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            outcomes = outcomes_by_step(run_dir / OUTCOMES_DIR)
            lengths = tokens_by_step(run_dir / "rollout_tokens")
            by_step = {
                step: step_metrics(
                    outcomes.get(step, []),
                    lengths.get(step, []),
                    prompts=prompts,
                    group_size=group,
                    same=answers_equivalent,
                )
                for step in range(1, self.through_step + 1)
            }
            write_step_metrics(run_dir / STEP_METRICS, by_step)
            write_efficiency(run_dir, efficiency(run_dir, self.through_step, prompts * group))
        print(f"soe metrics: steps 1..{self.through_step} written to {run_dir / STEP_METRICS}")


if __name__ == "__main__":
    CliApp.run(MetricsCli)
