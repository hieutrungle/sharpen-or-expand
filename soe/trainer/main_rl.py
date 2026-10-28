"""Training entry point called by slurm/train.sbatch: `python -m soe.trainer.main_rl --run-dir ...`.

It reads the frozen runs/<run_id>/config.yaml, so a chained or resumed job trains exactly what the
first job started, builds verl's Hydra overrides from it (soe/trainer/rl/verl_overrides.py) and runs
`verl.trainer.main_ppo` in a subprocess. Every job segment writes its own metrics file, because
verl's file logger truncates its target on open.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import file_sha256, load_suite
from soe.trainer.config.experiment import ExperimentConfig, load_config, rollout_protocol
from soe.trainer.rl.verl_overrides import LaunchPaths, verl_overrides
from soe.utils.run import read_run_info

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
        env = os.environ | {"VERL_FILE_LOGGER_PATH": str(metrics)}
        subprocess.run(
            [sys.executable, "-m", "verl.trainer.main_ppo", *overrides], env=env, check=True
        )
        # verl exits 0 only when the step budget is done; a killed job leaves no DONE marker.
        (run_dir / "DONE").write_text(f"finished in job {segment}\n")


if __name__ == "__main__":
    CliApp.run(LaunchCli)
