"""SFT sampling runs (M2.4/M2.5): their config, the seeded draw of problems, the finished samples.

Shared by `examples/data_preprocess/m2_sft_sample.py` (which samples) and `m2_sft_data.py` (which
builds the SFT files), so both use one definition of the permutation. No GPU imports.
"""

import random
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import file_sha256
from soe.utils.dataset.problems import Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl
from soe.utils.shards import shard_path, unfinished_shards

ASSETS = Path("scripts/stage_assets.yaml")  # repo path


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    suite: Path
    model: str  # a Hub id from scripts/stage_assets.yaml
    problems: str  # the clean pool, relative to $SOE_DATA
    problems_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    meta: str  # pool_meta.jsonl (difficulty per problem), relative to $SOE_DATA
    draw_seed: int
    draw_start: int = Field(ge=0)
    draw_stop: int = Field(gt=0)
    samples_per_problem: int = Field(ge=1)
    max_kept_per_problem: int = Field(ge=1)
    sampling_seed: int
    num_shards: int = Field(ge=1)
    work_dir: str  # relative to $SOE_DATA


def drawn(config: SampleConfig, data_root: Path) -> list[Problem]:
    """Problems [draw_start, draw_stop) of the pool, shuffled by draw_seed after sorting by id."""
    path = data_root / config.problems
    if file_sha256(path) != config.problems_sha256:
        raise SystemExit(f"{path} does not match its recorded sha256")
    pool = sorted(read_jsonl(path, Problem), key=lambda p: p.problem_id)
    random.Random(config.draw_seed).shuffle(pool)
    if config.draw_stop > len(pool) or config.draw_start >= config.draw_stop:
        raise SystemExit(f"draw [{config.draw_start}, {config.draw_stop}) outside {len(pool)}")
    return pool[config.draw_start : config.draw_stop]


def read_samples(config: SampleConfig, data_root: Path) -> dict[str, list[SampleRecord]]:
    """Every sample of a finished run by problem; refuses unfinished shards, missing or extra
    problems, and problems with the wrong number of samples."""
    work = data_root / config.work_dir
    left = unfinished_shards(work, config.num_shards)
    if left:
        raise SystemExit(f"{config.work_dir}: shards {left} not finished")
    by_problem: dict[str, list[SampleRecord]] = defaultdict(list)
    for k in range(config.num_shards):
        for record in read_jsonl(shard_path(work, k, config.num_shards), SampleRecord):
            by_problem[record.problem_id].append(record)
    if set(by_problem) != {p.problem_id for p in drawn(config, data_root)}:
        raise SystemExit(f"{config.work_dir}: samples do not match the drawn problems")
    if any(len(rs) != config.samples_per_problem for rs in by_problem.values()):
        raise SystemExit(f"{config.work_dir}: a problem has the wrong number of samples")
    return dict(by_problem)


def pinned_revision(hf_id: str) -> str:
    assets = yaml.safe_load(ASSETS.read_text())["assets"]
    revision: str = next(a["revision"] for a in assets if a["hf_id"] == hf_id)
    return revision
