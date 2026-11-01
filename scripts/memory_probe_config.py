"""Write one M1.10 knob probe: a memory-study cell with one rollout TP and gpu_memory_utilization.

    python scripts/memory_probe_config.py \\
        --cell examples/grpo_trainer/m1_memory_bl_qwen35-4b.yaml \\
        --tp 1 --gmu 0.6 --out-dir runs/memory-probes

A probe is two training steps with an 8-problem validation pass, so it costs little; every other
field is the cell's. Two, because AdamW creates its state at the first optimizer step: only from
step 2 on does generation run with the full training state resident (2B smoke, job 3227978: 4.1 GiB
per rank in step 1's generation, 11.2 GiB after its update). Step 2 also repeats every phase (weight
sync, generation, old log-prob, update) and ends with the save and its export. Prints the written
path (slurm/memory_probe.sbatch reads it).
"""

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import ExperimentConfig

PROBE_VALIDATION = {"max_problems": 8, "samples_per_problem": 1}
PROBE_STEPS = 2


def probe_name(cell_name: str, tp: int, gmu: float) -> str:
    return f"{cell_name}-probe-tp{tp}-gmu{round(gmu * 100):02d}"


def probe_config(cell: dict[str, Any], tp: int, gmu: float) -> dict[str, Any]:
    data = copy.deepcopy(cell)
    data["name"] = probe_name(cell["name"], tp, gmu)
    data["hardware"] |= {"rollout_tensor_parallel": tp, "gpu_memory_utilization": gmu}
    data["budget"] |= {"max_steps": PROBE_STEPS}
    data["checkpoint"] |= {"save_every_steps": PROBE_STEPS}
    data["validation"] = dict(PROBE_VALIDATION)
    ExperimentConfig.model_validate(data)  # refuse an invalid probe before writing it
    return data


class ProbeConfigCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/memory_probe_config.py", cli_kebab_case=True
    )

    cell: Path
    tp: int
    gmu: float
    out_dir: Path

    def cli_cmd(self) -> None:
        data = probe_config(yaml.safe_load(self.cell.read_text()), self.tp, self.gmu)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{data['name']}.yaml"
        header = f"# M1.10 probe derived from {self.cell} by scripts/memory_probe_config.py\n"
        path.write_text(header + yaml.safe_dump(data, sort_keys=False))
        print(path)


if __name__ == "__main__":
    CliApp.run(ProbeConfigCli)
