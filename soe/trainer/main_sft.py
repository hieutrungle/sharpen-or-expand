"""SFT entry point called by slurm/train.sbatch: `python -m soe.trainer.main_sft --run-dir ...`.

Starts the TRL worker (soe/trainer/sft/trainer.py) on the config's GPUs with torchrun, records the
job segment in `segments.jsonl` like main_rl, and writes the DONE marker when training and the
checkpoint export finished. SFT has no resume yet (M2): a run directory that already has a job
segment is refused rather than trained again from the start.
"""

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.rl.segments import SegmentEvent, append_event, read_events
from soe.utils.run import check_gpu_count, library_versions


def worker_command(run_dir: Path, n_gpus: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",  # torchrun
        "--standalone",
        f"--nproc_per_node={n_gpus}",
        "-m",
        "soe.trainer.sft.trainer",
        "--run-dir",
        str(run_dir),
    ]


class SftLaunchCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_sft", cli_kebab_case=True
    )

    run_dir: Path

    def cli_cmd(self) -> None:
        run_dir = self.run_dir.resolve()
        config = load_config(run_dir / "config.yaml")
        if config.sft is None:
            raise ValueError(f"{config.name} is not an SFT config")
        if read_events(run_dir):
            raise RuntimeError(f"{run_dir} was started before; SFT resume is not supported yet")
        check_gpu_count(config.sft.n_gpus)
        segment = os.environ.get("SLURM_JOB_ID", "local")
        record = partial(
            SegmentEvent,
            segment=segment,
            node=platform.node(),
            hardware=(run_dir / "hardware.txt").read_text().strip(),
            git_commit=os.environ.get("SOE_GIT_COMMIT", "unknown"),
        )
        started = datetime.now(UTC)
        append_event(
            run_dir,
            record(
                event="start", at=started, resumed_from_step=0, library_versions=library_versions()
            ),
        )
        try:
            subprocess.run(worker_command(run_dir, config.sft.n_gpus), check=True)
        except BaseException:
            append_event(run_dir, record(event="end", at=datetime.now(UTC), exit="failed"))
            raise
        append_event(run_dir, record(event="end", at=datetime.now(UTC), exit="done"))
        (run_dir / "DONE").write_text(f"finished in job {segment}\n")


if __name__ == "__main__":
    CliApp.run(SftLaunchCli)
