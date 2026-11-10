"""Frozen starting checkpoints (decision 031): what every non-development RL/DPO run starts from.

`frozen_start.yaml` names one pinned Hub snapshot per model with a digest over its files. A config
must name a frozen model (ExperimentConfig), and the launch recomputes the digest over the files it
will load (main_rl), so a re-downloaded, edited or partly copied snapshot cannot start a run.

The digest is the sha256 of the sorted lines `<relative path> <file sha256>\\n` over every regular
file under the directory, dotfiles included and symlinks followed (Hub snapshots link into blobs).
"""

import hashlib
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import file_sha256

FROZEN_STARTS = Path(__file__).with_name("frozen_start.yaml")
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenStart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_id: str
    revision: str = Field(min_length=40, max_length=40)
    files: int = Field(gt=0)
    digest: str = Field(pattern=SHA256_PATTERN)
    report: str  # the freeze report with every file's sha256, relative to the repo root


class FrozenStarts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    starts: list[FrozenStart]


@cache
def load_frozen_starts(path: Path = FROZEN_STARTS) -> tuple[FrozenStart, ...]:
    return tuple(FrozenStarts.model_validate(yaml.safe_load(path.read_text())).starts)


def frozen_start_for(hf_id: str, revision: str) -> FrozenStart | None:
    matches = [s for s in load_frozen_starts() if (s.hf_id, s.revision) == (hf_id, revision)]
    return matches[0] if matches else None


def file_hashes(directory: Path) -> dict[str, str]:
    """sha256 of every regular file under `directory` (symlinks followed), by relative path."""
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return {p.relative_to(directory).as_posix(): file_sha256(p) for p in files}


def directory_digest(hashes: dict[str, str]) -> str:
    lines = "".join(f"{name} {sha}\n" for name, sha in sorted(hashes.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def check_frozen_start(start: FrozenStart, directory: Path) -> None:
    hashes = file_hashes(directory)
    if len(hashes) != start.files or directory_digest(hashes) != start.digest:
        raise ValueError(
            f"{directory} does not match the frozen start {start.hf_id}@{start.revision} "
            f"({len(hashes)} files vs {start.files}; see {start.report})"
        )
