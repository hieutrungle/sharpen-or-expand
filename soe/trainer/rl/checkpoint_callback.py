"""verl checkpoint callback, after each save: the manifest entry, the background HF export
(decision 018), pruning of older FSDP shards across segments, and the stop step of the token
budget (decision 020).

Named in the overrides as `trainer.checkpoint_callback_class`. verl calls `on_save` on the driver
right after `global_step_<N>` and `latest_checkpointed_iteration.txt` are written, before that
step's metrics are logged. Setting `trainer.total_training_steps` here is how a run stops at its
budget: verl's loop reads it before every step and treats that step as its last.

Image only (imports verl).
"""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback

from soe.trainer.config.experiment import ExperimentConfig, load_config
from soe.trainer.rl.budget import BudgetState, last_step, write_stop_step
from soe.trainer.rl.checkpoints import (
    RUN_DIR_ENV,
    CheckpointRecord,
    append_record,
    export_dir,
    prune_shards,
)
from soe.trainer.rl.rollout_tokens import cumulative_rollout_tokens, tokens_by_step
from soe.trainer.rl.segments import RESUMED_FROM_ENV, SEGMENT_STARTED_ENV, training_hours
from soe.utils.run import read_run_info

REPO_ROOT = Path(__file__).resolve().parents[3]


class SoeCheckpointCallback(CheckpointCallback):  # type: ignore[misc]
    def __init__(self, config: Any = None) -> None:
        super().__init__(config=config)
        self._previous_save: tuple[datetime, int] | None = None  # (time, step) in this segment

    def on_save(
        self,
        trainer: Any,
        global_step: int,
        checkpoint_dir: str,
        async_save: bool = False,
        **kwargs: Any,
    ) -> None:
        if async_save:
            raise RuntimeError("async checkpoint saves are not supported by the soe callback")
        run_dir = Path(os.environ[RUN_DIR_ENV])
        responses = int(self.config.data.train_batch_size) * int(
            self.config.actor_rollout_ref.rollout.n
        )
        hardware = run_dir / "hardware.txt"
        run_id = read_run_info(run_dir).run_id
        record = CheckpointRecord(
            step=global_step,
            # Raises if any step 1..N lost a rollout (decisions 016, 018); verl then stops.
            rollout_tokens=cumulative_rollout_tokens(
                run_dir / "rollout_tokens", global_step, responses
            ),
            saved_at=datetime.now(UTC),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            hardware=hardware.read_text().strip() if hardware.exists() else "",
            checkpoint_dir=checkpoint_dir,
            export_dir=str(export_dir(Path(os.environ["SOE_DATA"]), run_id, global_step)),
        )
        append_record(run_dir, record)
        config = load_config(run_dir / "config.yaml")
        if config.checkpoint is None:
            raise ValueError(f"{config.name} is not an RL config")
        # verl prunes only this job's saves; shards an earlier job left behind are removed here.
        pruned = prune_shards(
            run_dir / "checkpoints",
            Path(os.environ["SOE_DATA"]),
            run_id,
            config.checkpoint.keep_last,
        )
        if pruned:
            print(f"soe: removed the FSDP shards of steps {pruned}", flush=True)
        self._update_stop_step(trainer, run_dir, config, record)
        log = run_dir / "exports" / f"step_{global_step}.log"
        log.parent.mkdir(exist_ok=True)
        with open(log, "a") as out:
            # A CPU process next to training; main_rl's sweep redoes it if the job ends first.
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "soe.trainer.main_export",
                    "--run-dir",
                    str(run_dir),
                    "--step",
                    str(global_step),
                ],
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                start_new_session=True,
            )

    def _update_stop_step(
        self, trainer: Any, run_dir: Path, config: ExperimentConfig, record: CheckpointRecord
    ) -> None:
        if config.budget is None or config.checkpoint is None:
            raise ValueError(f"{config.name} is not an RL config")
        step = record.step
        since = max(0, step - config.checkpoint.save_every_steps)
        recent = tokens_by_step(run_dir / "rollout_tokens")
        tokens_per_step = sum(sum(recent[s]) for s in range(since + 1, step + 1)) / (step - since)
        if self._previous_save is None:  # first save of this job: time since the job started
            started = datetime.fromisoformat(os.environ[SEGMENT_STARTED_ENV])
            self._previous_save = (started, int(os.environ[RESUMED_FROM_ENV]))
        previous_time, previous_step = self._previous_save
        hours_per_step = (
            (record.saved_at - previous_time).total_seconds() / 3600 / (step - previous_step)
        )
        self._previous_save = (record.saved_at, step)
        state = BudgetState(
            step=step,
            rollout_tokens=record.rollout_tokens,
            tokens_per_step=tokens_per_step,
            hours_used=training_hours(run_dir, record.saved_at),
            hours_per_step=hours_per_step,
        )
        stop = last_step(
            state,
            budget_tokens=config.budget.rollout_tokens,
            max_hours=config.budget.max_hours,
            max_steps=config.budget.max_steps,
        )
        trainer.total_training_steps = stop
        write_stop_step(run_dir, stop, state)
        print(
            f"soe: step {step}, {record.rollout_tokens:,} rollout tokens, "
            f"{state.hours_used:.2f} h: last step set to {stop}",
            flush=True,
        )
