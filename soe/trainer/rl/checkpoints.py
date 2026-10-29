"""RL checkpoint bookkeeping (decisions 013, 018): the manifest, export paths and HF tensor names.

- `runs/<id>/checkpoints.jsonl` gets one CheckpointRecord per save (step, cumulative rollout tokens,
  time, job, hardware), written by the verl checkpoint callback.
- Each checkpoint is exported to `$SOE_DATA/checkpoints/<run_id>/step_<N>/hf`; the export is
  complete when that directory exists (it is renamed into place at the end).
- `prune_shards` keeps the newest `keep_last` FSDP shards across segments: verl 0.9.1 prunes only
  the saves of its own process, so a resumed job never removed the previous job's shards (M1.5).
- verl's FSDP state dict of a PEFT model names tensors `base_model.model.<hf name>` with
  `.base_layer.` inside LoRA-wrapped modules; `hf_name` and `lora_parts` map them back.

No verl or torch import, so this runs in the CPU tests.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST = "checkpoints.jsonl"
RUN_DIR_ENV = "SOE_RUN_DIR"  # set by main_rl for the verl processes
PEFT_PREFIX = "base_model.model."
_LORA = re.compile(r"^(?P<module>.+)\.lora_(?P<part>[AB])\.default\.weight$")
_STEP_DIR = re.compile(r"^global_step_(\d+)$")


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    step: int = Field(gt=0)
    rollout_tokens: int = Field(ge=0)  # generated tokens of training rollouts, steps 1..step
    saved_at: datetime
    slurm_job_id: str | None
    hardware: str  # the run's hardware.txt line
    checkpoint_dir: str  # verl's global_step_<N> directory
    export_dir: str  # where the HF export goes


def append_record(run_dir: Path, record: CheckpointRecord) -> None:
    with open(run_dir / MANIFEST, "a") as f:
        f.write(record.model_dump_json() + "\n")


def read_records(run_dir: Path) -> list[CheckpointRecord]:
    """Manifest entries, the latest per step (a step saved again after a resume replaces the
    earlier entry)."""
    path = run_dir / MANIFEST
    if not path.exists():
        return []
    by_step: dict[int, CheckpointRecord] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = CheckpointRecord.model_validate_json(line)
            by_step[record.step] = record
    return [by_step[step] for step in sorted(by_step)]


def export_dir(data_root: Path, run_id: str, step: int) -> Path:
    return data_root / "checkpoints" / run_id / f"step_{step}" / "hf"


def saved_steps(checkpoints_dir: Path) -> list[int]:
    """Steps whose actor shards are still on disk (verl prunes older ones)."""
    steps = []
    for path in checkpoints_dir.glob("global_step_*"):
        match = _STEP_DIR.match(path.name)
        if match and any((path / "actor").glob("model_world_size_*_rank_0.pt")):
            steps.append(int(match.group(1)))
    return sorted(steps)


def missing_exports(checkpoints_dir: Path, data_root: Path, run_id: str) -> list[int]:
    return [
        step
        for step in saved_steps(checkpoints_dir)
        if not export_dir(data_root, run_id, step).exists()
    ]


def prune_shards(checkpoints_dir: Path, data_root: Path, run_id: str, keep_last: int) -> list[int]:
    """Remove the actor shards of every step older than the newest `keep_last`, whichever job saved
    them; a step without its HF export keeps its shards (the export reads them). Returns the steps
    removed. Like verl, it leaves `global_step_<N>/data.pt` (dataloader state, a few KB)."""
    removed = []
    for step in saved_steps(checkpoints_dir)[:-keep_last]:
        if export_dir(data_root, run_id, step).exists():
            shutil.rmtree(checkpoints_dir / f"global_step_{step}" / "actor")
            removed.append(step)
    return removed


def hf_name(fsdp_key: str) -> str:
    """HF tensor name of a non-LoRA FSDP key (the wrapped layer's own weight or bias)."""
    name = fsdp_key.removeprefix(PEFT_PREFIX)
    return name.replace(".base_layer.", ".")


def lora_parts(fsdp_key: str) -> tuple[str, str] | None:
    """(HF weight name of the adapted module, "A" or "B") for a LoRA matrix, else None."""
    match = _LORA.match(fsdp_key.removeprefix(PEFT_PREFIX))
    if match is None:
        return None
    return f"{match.group('module')}.weight", match.group("part")


def is_language_model(name: str) -> bool:
    return name.startswith("model.language_model.") or name == "lm_head.weight"
