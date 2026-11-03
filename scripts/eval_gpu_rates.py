"""Sampling rate per GPU type of one sharded evaluation, from its shard claims (M1.12, G0 schedule).

    python scripts/eval_gpu_rates.py --eval-dir $SOE_DATA/eval/qwen35-4b-base_soe-eval-v3 \\
        --out docs/reports/m0_eval_gpu_rates.json

A shard's time runs from its claim (`claims/shard-k/owner` written) to `claims/shard-k/done`, so it
includes the engine start-up of a job's first shard; its GPU type is the one its runinfo records.
Rate per type = completion tokens / shard seconds summed over that type's shards. The schedule turns
one evaluation into A40-hours with these rates (total tokens / the A40 rate), and a Lane B pool into
A40-equivalents (GPUs x rate / the A40 rate).
"""

import json
import statistics
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.utils.shards import claim_dir, shard_path


def gpu_type(runinfo: dict[str, Any]) -> str:
    """'A40' from 'NVIDIA A40, 49140 MiB' (one-GPU workers record one GPU)."""
    (gpu,) = runinfo["gpus"]
    return str(gpu).split(",")[0].removeprefix("NVIDIA ").strip()


def shard_record(eval_dir: Path, shard: int, num_shards: int) -> dict[str, Any]:
    runinfo_path = eval_dir / f"runinfo-{shard}-of-{num_shards}.jsonl"
    runinfo = json.loads(runinfo_path.read_text().splitlines()[0])
    claim = claim_dir(eval_dir, shard)
    seconds = (claim / "done").stat().st_mtime - (claim / "owner").stat().st_mtime
    with shard_path(eval_dir, shard, num_shards).open() as f:
        tokens = sum(int(json.loads(line)["completion_tokens"]) for line in f)
    return {"shard": shard, "gpu": gpu_type(runinfo), "seconds": seconds, "tokens": tokens}


def rates(eval_dir: Path) -> dict[str, Any]:
    num_shards = int(json.loads((eval_dir / "plan.json").read_text())["num_shards"])
    shards = [shard_record(eval_dir, k, num_shards) for k in range(num_shards)]
    by_gpu: dict[str, Any] = {}
    for gpu in sorted({s["gpu"] for s in shards}):
        mine = [s for s in shards if s["gpu"] == gpu]
        seconds = sum(s["seconds"] for s in mine)
        tokens = sum(s["tokens"] for s in mine)
        by_gpu[gpu] = {
            "shards": len(mine),
            "shard_hours": seconds / 3600,
            "completion_tokens": tokens,
            "tokens_per_second": tokens / seconds,
            "median_shard_tokens_per_second": statistics.median(
                s["tokens"] / s["seconds"] for s in mine
            ),
        }
    return {
        "eval_dir": eval_dir.name,
        "shards": num_shards,
        "completion_tokens": sum(s["tokens"] for s in shards),
        "shard_hours": sum(s["seconds"] for s in shards) / 3600,
        "by_gpu": by_gpu,
    }


class RatesCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/eval_gpu_rates.py", cli_kebab_case=True
    )

    eval_dir: Path
    out: Path

    def cli_cmd(self) -> None:
        result = rates(self.eval_dir)
        self.out.write_text(json.dumps(result, indent=2) + "\n")
        for gpu, r in result["by_gpu"].items():
            print(f"{gpu}: {r['shards']} shards, {r['tokens_per_second']:.0f} tok/s")


if __name__ == "__main__":
    CliApp.run(RatesCli)
