"""Training entry point called by slurm/train.sbatch: `python -m soe.trainer.main_rl --run-dir ...`.

It reads the frozen runs/<run_id>/config.yaml, so a chained or resumed job trains exactly what the
first job started, builds verl's Hydra overrides from it (soe/trainer/rl/verl_overrides.py) and runs
`verl.trainer.main_ppo` in a subprocess. Every job segment writes its own metrics file, because
verl's file logger truncates its target on open. A rollout failure (soe/workers/rollout/failures.py)
stops the trainer within a minute, instead of verl training on the rest of the batch.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import file_sha256, load_suite
from soe.trainer.config.experiment import ExperimentConfig, load_config, rollout_protocol
from soe.trainer.rl.budget import read_stop_step
from soe.trainer.rl.checkpoints import RUN_DIR_ENV, missing_exports, prune_shards
from soe.trainer.rl.rollout_tokens import TOKENS_ENV
from soe.trainer.rl.segments import (
    RESUMED_FROM_ENV,
    SEGMENT_STARTED_ENV,
    SegmentEvent,
    append_event,
)
from soe.trainer.rl.verl_overrides import LaunchPaths, verl_overrides
from soe.utils.run import read_run_info
from soe.workers.rollout.failures import FAILURES_ENV, read_failures

REWARD_FUNCTION = Path("soe/workers/reward_manager/math_reward.py")
AGENT_LOOPS = Path("soe/workers/rollout/agent_loops.yaml")


def checked_data_file(data_root: Path, relative: str, sha256: str) -> Path:
    path = data_root / relative
    if not path.is_file() or file_sha256(path) != sha256:
        raise ValueError(f"{path} is missing or does not match its recorded sha256")
    return path


def model_path(config: ExperimentConfig, data_root: Path) -> Path:
    """The frozen SFT checkpoint when the config names one, else the staged Hub snapshot."""
    if config.init_checkpoint is not None:
        return data_root / config.init_checkpoint
    from huggingface_hub import snapshot_download  # image only

    hub_path = snapshot_download(
        config.model.hf_id, revision=config.model.revision, local_files_only=True
    )
    return Path(hub_path)


def frozen_overrides(run_dir: Path, overrides: list[str]) -> list[str]:
    """Write the overrides on the first job; later jobs must build the identical list."""
    path = run_dir / "verl_overrides.json"
    if path.exists():
        recorded: list[str] = json.loads(path.read_text())
        if recorded != overrides:
            changed = sorted(set(recorded) ^ set(overrides))
            raise ValueError(f"verl overrides differ from the run's first job: {changed[:6]}")
        return recorded
    path.write_text(json.dumps(overrides, indent=1) + "\n")
    return overrides


class RolloutFailure(RuntimeError):
    pass


def run_until_failure(
    cmd: list[str], env: dict[str, str], failures: Path, poll_seconds: float = 30.0
) -> None:
    """Run the trainer; stop it and raise at the first recorded rollout failure."""
    with subprocess.Popen(cmd, env=env) as proc:
        while proc.poll() is None:
            time.sleep(poll_seconds)
            if found := read_failures(failures):
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RolloutFailure(f"{len(found)} rollout failure(s) in {failures}: {found[0]}")
    if found := read_failures(failures):  # recorded during the last poll interval
        raise RolloutFailure(f"{len(found)} rollout failure(s) in {failures}: {found[0]}")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def latest_checkpoint_step(checkpoints: Path) -> int:
    """The step verl resumes from (its latest_checkpointed_iteration.txt), 0 for a fresh run."""
    tracker = checkpoints / "latest_checkpointed_iteration.txt"
    return int(tracker.read_text().strip()) if tracker.exists() else 0


def segment_overrides(frozen: list[str], resumed_from: int, stop_step: int | None) -> list[str]:
    """The frozen override list with this job's two differences (decision 020): a resumed job skips
    the validation before training (its checkpoint step was validated when saved, and a second
    record at that step would replace the step's training metrics), and starts with the stop step
    the last save estimated."""
    changes = {}
    if resumed_from > 0:
        changes["trainer.val_before_train"] = "false"
    if stop_step is not None:
        changes["trainer.total_training_steps"] = str(stop_step)
    out = []
    for item in frozen:
        key = item.split("=", 1)[0]
        out.append(f"{key}={changes.pop(key)}" if key in changes else item)
    if changes:
        raise ValueError(f"override keys not in the frozen list: {sorted(changes)}")
    return out


def export_missing(run_dir: Path, data_root: Path, run_id: str) -> None:
    """Export every saved checkpoint that has no HF export yet, one after another."""
    for step in missing_exports(run_dir / "checkpoints", data_root, run_id):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "soe.trainer.main_export",
                "--run-dir",
                str(run_dir),
                "--step",
                str(step),
            ],
            check=True,
        )


class LaunchCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_rl", env_prefix="SOE_LAUNCH_", cli_kebab_case=True
    )

    run_dir: Path

    def cli_cmd(self) -> None:
        run_dir = self.run_dir.resolve()
        info = read_run_info(run_dir)
        config = load_config(run_dir / "config.yaml")
        protocol = rollout_protocol(config, load_suite(Path(config.eval_suite)))
        data_root = Path(os.environ["SOE_DATA"])
        paths = LaunchPaths(
            model=model_path(config, data_root),
            run_dir=run_dir,
            train_file=checked_data_file(
                data_root, config.data.train_file, config.data.train_sha256
            ),
            val_file=checked_data_file(data_root, config.data.val_file, config.data.val_sha256),
            checkpoints=run_dir / "checkpoints",
            reward_function=REWARD_FUNCTION.resolve(),
            agent_loops=AGENT_LOOPS.resolve(),
        )
        overrides = frozen_overrides(
            run_dir,
            verl_overrides(config, protocol, seed=info.seed, run_id=info.run_id, paths=paths),
        )
        segment = os.environ.get("SLURM_JOB_ID", "local")
        metrics = run_dir / "metrics" / f"segment-{segment}.jsonl"
        metrics.parent.mkdir(exist_ok=True)
        failures = run_dir / f"rollout_failures-{segment}.jsonl"
        tokens = run_dir / "rollout_tokens" / f"segment-{segment}.jsonl"
        tokens.parent.mkdir(exist_ok=True)
        env = os.environ | {
            "VERL_FILE_LOGGER_PATH": str(metrics),
            FAILURES_ENV: str(failures),
            TOKENS_ENV: str(tokens),
            RUN_DIR_ENV: str(run_dir),
        }
        resumed_from = latest_checkpoint_step(run_dir / "checkpoints")
        started = datetime.now(UTC)
        env |= {SEGMENT_STARTED_ENV: started.isoformat(), RESUMED_FROM_ENV: str(resumed_from)}
        record = partial(
            SegmentEvent,
            segment=segment,
            node=platform.node(),
            hardware=(run_dir / "hardware.txt").read_text().strip(),
            git_commit=os.environ.get("SOE_GIT_COMMIT", "unknown"),
        )
        append_event(run_dir, record(event="start", at=started, resumed_from_step=resumed_from))
        # A checkpoint whose export the previous job did not finish (decision 018).
        export_missing(run_dir, data_root, info.run_id)
        this_job = segment_overrides(overrides, resumed_from, read_stop_step(run_dir))
        try:
            run_until_failure(
                [sys.executable, "-m", "verl.trainer.main_ppo", *this_job], env, failures
            )
            export_missing(run_dir, data_root, info.run_id)
            # Shards kept at a save because their export was still running.
            assert config.checkpoint is not None
            prune_shards(
                run_dir / "checkpoints", data_root, info.run_id, config.checkpoint.keep_last
            )
        except BaseException:
            append_event(run_dir, record(event="end", at=datetime.now(UTC), exit="failed"))
            raise
        append_event(run_dir, record(event="end", at=datetime.now(UTC), exit="done"))
        # verl exits 0 only when the run reached its last step; a killed job leaves no DONE marker.
        (run_dir / "DONE").write_text(f"finished in job {segment}\n")


if __name__ == "__main__":
    CliApp.run(LaunchCli)
