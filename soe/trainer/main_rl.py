"""Training entry point called by slurm/train.sbatch: `python -m soe.trainer.main_rl --run-dir ...`.

It reads the frozen runs/<run_id>/config.yaml, so a chained or resumed job trains exactly what
the first job started. The verl/TRL wiring is M1 work (docs/roadmap/M0.md, "Hand-off to M1");
until then this validates the run directory and stops.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.utils.run import read_run_info


class LaunchCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_rl", env_prefix="SOE_LAUNCH_", cli_kebab_case=True
    )

    run_dir: Path

    def cli_cmd(self) -> None:
        info = read_run_info(self.run_dir)
        config = load_config(self.run_dir / "config.yaml")
        raise NotImplementedError(
            f"run {info.run_id} ({config.algorithm}) is valid, but training is wired up in M1"
        )


if __name__ == "__main__":
    CliApp.run(LaunchCli)
