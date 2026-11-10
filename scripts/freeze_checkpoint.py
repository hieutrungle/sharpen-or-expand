"""M2.12: hash a checkpoint directory, optionally copy it, and make it read-only (decision 031).

The RL/DPO start (the pinned base snapshot, files left in place; their blobs made read-only):

    S=$SOE_DATA/hf/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
    python scripts/freeze_checkpoint.py --source $S --read-only \\
        --report docs/reports/m2_start_frozen.json

The SFT arm (decision 030; copied, the copy made read-only; run metadata recorded):

    R=20261109-185607-sft_m2_run2_qwen35-4b-s1-j3237928
    python scripts/freeze_checkpoint.py --source $SOE_DATA/checkpoints/$R/step_719/hf \\
        --copy-to $SOE_DATA/frozen/sft_qwen35-4b_v1/hf --read-only --run-dir runs/$R \\
        --report docs/reports/m2_sft_frozen.json

The report holds every file's sha256 and size, the digest (soe/trainer/config/frozen_start.py),
the export's `soe_export.json` if present, and with `--run-dir` the run id, every segment's commit
and the data hashes. A copy is re-hashed and must equal the source. No model weights are read.
"""

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.config.frozen_start import directory_digest, file_hashes
from soe.utils.run import read_run_info

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def make_read_only(path: Path) -> None:
    """Clear the write bits of `path`'s target (a symlink's blob, not the link)."""
    target = path.resolve()
    os.chmod(target, target.stat().st_mode & ~WRITE_BITS)


def copy_files(source: Path, destination: Path) -> None:
    if destination.exists():
        raise SystemExit(f"{destination} exists; a frozen copy is written once")
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)  # follows symlinks: the copy holds real files


def run_metadata(run_dir: Path) -> dict[str, Any]:
    info = read_run_info(run_dir)
    segments = [json.loads(line) for line in (run_dir / "segments.jsonl").read_text().splitlines()]
    return {
        "run_id": info.run_id,
        "config_sha256": info.config_sha256,
        "seed": info.seed,
        "segment_commits": sorted({s["git_commit"] for s in segments if "git_commit" in s}),
        "data": load_config(run_dir / "config.yaml").data.model_dump(),
    }


class FreezeCheckpointCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/freeze_checkpoint.py", cli_kebab_case=True
    )

    source: Path
    report: Path
    copy_to: Path | None = None
    read_only: bool = False
    run_dir: Path | None = None

    def cli_cmd(self) -> None:
        hashes = file_hashes(self.source)
        frozen = self.source
        if self.copy_to is not None:
            copy_files(self.source, self.copy_to)
            if file_hashes(self.copy_to) != hashes:
                raise SystemExit(f"{self.copy_to} differs from {self.source} after the copy")
            frozen = self.copy_to
        if self.read_only:
            for path in frozen.rglob("*"):
                if path.is_file():
                    make_read_only(path)
            if self.copy_to is not None:
                for directory in [frozen, *(p for p in frozen.rglob("*") if p.is_dir())]:
                    make_read_only(directory)
        export = self.source / "soe_export.json"
        report: dict[str, Any] = {
            "source": str(self.source),
            "frozen": str(frozen),
            "read_only": self.read_only,
            "files": {
                name: {"sha256": sha, "bytes": (frozen / name).stat().st_size}
                for name, sha in hashes.items()
            },
            "file_count": len(hashes),
            "digest": directory_digest(hashes),
            "soe_export": json.loads(export.read_text()) if export.is_file() else None,
            "run": run_metadata(self.run_dir) if self.run_dir is not None else None,
        }
        self.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in ("frozen", "file_count", "digest")}))


if __name__ == "__main__":
    CliApp.run(FreezeCheckpointCli)
