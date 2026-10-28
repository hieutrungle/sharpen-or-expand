"""Export one RL checkpoint to HF: `python -m soe.trainer.main_export --run-dir runs/<id> --step N`.

Started by the checkpoint callback after each save and by main_rl's sweep (decision 018). Writes
`$SOE_DATA/checkpoints/<run_id>/step_<N>/hf`; does nothing if that export exists already.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import ExperimentConfig, load_config
from soe.trainer.main_rl import model_path
from soe.trainer.rl.checkpoints import export_dir, read_records
from soe.utils.run import read_run_info


def base_model_name(config: ExperimentConfig) -> str:
    if config.init_checkpoint is not None:
        return config.init_checkpoint
    return f"{config.model.hf_id}@{config.model.revision}"


class ExportCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_export",
        env_prefix="SOE_EXPORT_",
        cli_kebab_case=True,
    )

    run_dir: Path
    step: int

    def cli_cmd(self) -> None:
        from soe.trainer.rl.export import export_checkpoint  # image only (torch)

        run_dir = self.run_dir.resolve()
        info = read_run_info(run_dir)
        config = load_config(run_dir / "config.yaml")
        data_root = Path(os.environ["SOE_DATA"])
        out = export_dir(data_root, info.run_id, self.step)
        if out.exists():
            print(f"step {self.step}: already exported to {out}")
            return
        records = {record.step: record for record in read_records(run_dir)}
        record = records.get(self.step)
        exported = export_checkpoint(
            run_dir / "checkpoints" / f"global_step_{self.step}" / "actor",
            model_path(config, data_root),
            out,
            run_id=info.run_id,
            step=self.step,
            rollout_tokens=None if record is None else record.rollout_tokens,
            base_model=base_model_name(config),
        )
        print(exported.model_dump_json(indent=1))


if __name__ == "__main__":
    CliApp.run(ExportCli)
