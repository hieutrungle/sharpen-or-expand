"""Run directories: one folder per training run, holding everything needed to audit it.

Layout of runs/<UTC yyyymmdd-HHMMSS>-<config name>-s<seed>[-j<slurm job id>]/:
    config.yaml      frozen copy of the experiment config (resumed jobs read this, not examples/)
    run.json         provenance: commit, seed, host, GPUs, library versions (RunInfo)
    efficiency.json  token and time accounting, updated by the trainer (RunEfficiency)
    checkpoints/     written by the trainer at fixed rollout-token intervals
    card.md          human-readable summary (soe.utils.run_card)
"""

import hashlib
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config

TRACKED_LIBRARIES = ("torch", "vllm", "verl", "transformers", "trl", "flash-attn", "peft")


class RunInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str
    config_name: str
    config_sha256: str
    seed: int
    git_commit: str
    git_dirty: bool
    created_at: datetime
    hostname: str
    slurm_job_id: str | None
    gpus: list[str]
    library_versions: dict[str, str | None]


class RunEfficiency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    through_step: int = Field(default=0, ge=0)  # the token counts cover steps 1..through_step
    rollout_tokens: int = Field(default=0, ge=0)
    training_tokens: int = Field(default=0, ge=0)
    gpu_hours: float = Field(default=0.0, ge=0)
    wall_clock_hours: float = Field(default=0.0, ge=0)
    newly_solved_heldout: int | None = None

    def newly_solved_per_million_rollout_tokens(self) -> float | None:
        if self.newly_solved_heldout is None or self.rollout_tokens == 0:
            return None
        return self.newly_solved_heldout / (self.rollout_tokens / 1e6)


def make_run_id(config_name: str, seed: int, now: datetime, slurm_job_id: str | None = None) -> str:
    # The job id keeps ids unique when two jobs of one config and seed start in the same second
    # (the M1.5 control and test runs did, job 3222907).
    job = f"-j{slurm_job_id}" if slurm_job_id else ""
    return f"{now:%Y%m%d-%H%M%S}-{config_name}-s{seed}{job}"


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def gpu_names() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def create_run(config_path: Path, seed: int, runs_root: Path, repo_root: Path) -> Path:
    """Validate the config, then create runs/<run_id>/ with a frozen config copy and run.json."""
    config = load_config(config_path)  # fail before creating anything
    now = datetime.now(UTC)
    run_id = make_run_id(config.name, seed, now, os.environ.get("SLURM_JOB_ID"))
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    shutil.copyfile(config_path, run_dir / "config.yaml")
    info = RunInfo(
        run_id=run_id,
        config_name=config.name,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        seed=seed,
        git_commit=_git(["rev-parse", "HEAD"], repo_root),
        git_dirty=bool(_git(["status", "--porcelain", "--untracked-files=no"], repo_root)),
        created_at=now,
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpus=gpu_names(),
        library_versions=library_versions(),
    )
    (run_dir / "run.json").write_text(info.model_dump_json(indent=2) + "\n")
    (run_dir / "efficiency.json").write_text(RunEfficiency().model_dump_json(indent=2) + "\n")
    return run_dir


def visible_gpus(nvidia_smi_list: str) -> int:
    """GPUs in `nvidia-smi -L` output, which Slurm limits to the job's allocation."""
    return sum(line.startswith("GPU ") for line in nvidia_smi_list.splitlines())


def check_gpu_count(configured: int) -> None:
    """Refuse a job whose allocation differs from the config's GPU count: with more GPUs some sit
    idle, with fewer verl waits for Ray resources forever or torchrun shares a GPU."""
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    visible = visible_gpus(listed.stdout)
    if visible != configured:
        raise RuntimeError(f"the config asks for {configured} GPUs but this job has {visible}")


def read_run_info(run_dir: Path) -> RunInfo:
    return RunInfo.model_validate_json((run_dir / "run.json").read_text())


def read_efficiency(run_dir: Path) -> RunEfficiency:
    return RunEfficiency.model_validate_json((run_dir / "efficiency.json").read_text())


def write_efficiency(run_dir: Path, efficiency: RunEfficiency) -> None:
    tmp = run_dir / "efficiency.json.tmp"
    tmp.write_text(efficiency.model_dump_json(indent=2) + "\n")
    tmp.replace(run_dir / "efficiency.json")


class NewRunCli(BaseSettings):
    """Create a run directory and print its path (used by slurm/train.sbatch)."""

    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.utils.run", env_prefix="SOE_RUN_", cli_kebab_case=True
    )

    config: Path
    seed: int
    runs_root: Path = Path("runs")

    def cli_cmd(self) -> None:
        print(create_run(self.config, self.seed, self.runs_root, repo_root=Path.cwd()))


if __name__ == "__main__":
    CliApp.run(NewRunCli)
