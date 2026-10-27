"""Work split for sampling jobs: a fixed list of units cut into shards that concurrent jobs claim.

A job claims a shard with mkdir, which is atomic on the shared (Lustre) file system, so jobs on
different GPUs never sample the same shard and faster GPUs simply take more shards. A job that dies
leaves its claim without `done`; that shard is finished by naming it explicitly (`--shard k`).
"""

from pathlib import Path


def shard_path(work: Path, shard: int, num_shards: int) -> Path:
    return work / f"samples-{shard}-of-{num_shards}.jsonl"


def claim_dir(work: Path, shard: int) -> Path:
    return work / "claims" / f"shard-{shard}"


def claim_shard(work: Path, shard: int, owner: str) -> bool:
    """Atomically claim a shard for this job; False if another job already holds it."""
    claim = claim_dir(work, shard)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        return False
    (claim / "owner").write_text(owner + "\n")
    return True


def mark_done(work: Path, shard: int) -> None:
    claim = claim_dir(work, shard)
    claim.mkdir(parents=True, exist_ok=True)  # an explicit --shard run may not have claimed it
    (claim / "done").touch()


def unfinished_shards(work: Path, num_shards: int) -> list[int]:
    return [k for k in range(num_shards) if not (claim_dir(work, k) / "done").exists()]
