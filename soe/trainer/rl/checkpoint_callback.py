"""verl checkpoint callback: manifest entry and background HF export after each save (decision 018).

Named in the overrides as `trainer.checkpoint_callback_class`. verl calls `on_save` on the driver
right after `global_step_<N>` and `latest_checkpointed_iteration.txt` are written, before that
step's metrics are logged.

Image only (imports verl).
"""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback

from soe.trainer.rl.checkpoints import RUN_DIR_ENV, CheckpointRecord, append_record, export_dir
from soe.trainer.rl.rollout_tokens import cumulative_rollout_tokens
from soe.utils.run import read_run_info

REPO_ROOT = Path(__file__).resolve().parents[3]


class SoeCheckpointCallback(CheckpointCallback):  # type: ignore[misc]
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
