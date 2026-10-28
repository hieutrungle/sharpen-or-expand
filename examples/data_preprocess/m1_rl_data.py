"""RL data v3.1 (decision 012): a 256-problem in-loop validation split carved from RL train v3,
and both splits written as verl parquet files with the frozen suite's prompt.

    SOE_DATA=... python examples/data_preprocess/m1_rl_data.py \
        --config examples/data_preprocess/m1_rl_data.yaml

Writes, under $SOE_DATA/datasets/soe/v3.1/rl/: train.jsonl and val.jsonl (Problem records), and
verl/train.parquet and verl/val.parquet (one verl row per problem). The report
docs/reports/m1_datasets_v3.1.json holds counts, the validation ids and every sha256, but no
problem text. Nothing is dropped from train except the validation problems themselves.
"""

import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import PromptSpec, file_sha256, load_suite
from soe.utils.dataset.decontaminate import Ngram, common_ngrams, word_ngrams
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl, write_jsonl


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationSplit(Frozen):
    size: int = Field(gt=0)
    seed: int


class NearDuplicate(Frozen):
    ngram: int = Field(gt=0)
    min_fraction: float = Field(gt=0, le=1)
    boilerplate_max_count: int = Field(gt=0)


class DataConfig(Frozen):
    version: str
    source: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite: str
    output_dir: str
    report: str
    validation: ValidationSplit
    near_duplicate: NearDuplicate


def shared_fraction(
    problems: Sequence[Problem], *, n: int, ignore: frozenset[Ngram]
) -> dict[str, float | None]:
    """For each problem, the fraction of its n-grams that occur in at least one other problem.

    This is decision 003's union rule (`covered_by_union`) against "everything else", computed in
    one pass. None marks a problem with no distinctive n-gram left, which cannot be checked.
    """
    grams = {p.problem_id: word_ngrams(p.question, n) - ignore for p in problems}
    counts: dict[Ngram, int] = defaultdict(int)
    for own in grams.values():
        for gram in own:
            counts[gram] += 1
    return {
        pid: (sum(counts[g] > 1 for g in own) / len(own) if own else None)
        for pid, own in grams.items()
    }


def carve_validation(
    problems: Sequence[Problem],
    fractions: dict[str, float | None],
    *,
    size: int,
    seed: int,
    min_fraction: float,
) -> tuple[list[Problem], list[Problem], int]:
    """(train, val, eligible count). Val is a seeded draw from the problems the rest of the pool
    does not cover; train keeps every other problem, in the original order."""
    eligible = sorted(pid for pid, f in fractions.items() if f is not None and f < min_fraction)
    if len(eligible) < size:
        raise ValueError(f"only {len(eligible)} eligible problems, need {size}")
    chosen = set(random.Random(seed).sample(eligible, size))
    train = [p for p in problems if p.problem_id not in chosen]
    val = [p for p in problems if p.problem_id in chosen]
    return train, val, len(eligible)


def verl_row(problem: Problem, prompt: PromptSpec, *, split: str, index: int) -> dict[str, Any]:
    """One row of verl's RLHFDataset: the chat turn (verl applies the chat template, with
    enable_thinking from the experiment config) and the gold answer for the reward."""
    return {
        "data_source": f"soe/{problem.source}",
        "prompt": [{"role": "user", "content": prompt.render(problem.question)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": problem.answer},
        "extra_info": {"index": index, "problem_id": problem.problem_id, "split": split},
    }


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Content hash of the rows, independent of the parquet writer."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def write_parquet(rows: Sequence[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa  # data extra / image only
    import pyarrow.parquet as pq

    message = pa.struct([("role", pa.string()), ("content", pa.string())])
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message)),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [("index", pa.int64()), ("problem_id", pa.string()), ("split", pa.string())]
                ),
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path, compression="snappy")


def build(config: DataConfig, data_root: Path) -> dict[str, Any]:
    source = data_root / config.source
    if file_sha256(source) != config.source_sha256:
        raise ValueError(f"{source} does not match its recorded sha256")
    suite = load_suite(Path(config.suite))
    problems = read_jsonl(source, Problem)
    nd = config.near_duplicate
    ignore = common_ngrams([p.question for p in problems], nd.ngram, nd.boilerplate_max_count)
    fractions = shared_fraction(problems, n=nd.ngram, ignore=ignore)
    train, val, eligible = carve_validation(
        problems,
        fractions,
        size=config.validation.size,
        seed=config.validation.seed,
        min_fraction=nd.min_fraction,
    )
    out = data_root / config.output_dir
    files: dict[str, dict[str, Any]] = {}
    for split, split_problems in (("train", train), ("val", val)):
        write_jsonl(out / f"{split}.jsonl", split_problems)
        rows = [
            verl_row(p, suite.prompt, split=split, index=i) for i, p in enumerate(split_problems)
        ]
        write_parquet(rows, out / "verl" / f"{split}.parquet")
        files[split] = {
            "problems": len(split_problems),
            "jsonl": f"{config.output_dir}/{split}.jsonl",
            "jsonl_sha256": file_sha256(out / f"{split}.jsonl"),
            "parquet": f"{config.output_dir}/verl/{split}.parquet",
            "parquet_sha256": file_sha256(out / "verl" / f"{split}.parquet"),
            "rows_sha256": rows_sha256(rows),
        }
    val_fractions = [fractions[p.problem_id] or 0.0 for p in val]
    return {
        "version": config.version,
        "derived_from": {"path": config.source, "sha256": config.source_sha256},
        "suite": {
            "path": config.suite,
            "name": suite.name,
            "sha256": file_sha256(Path(config.suite)),
        },
        "validation": config.validation.model_dump(),
        "near_duplicate": nd.model_dump(),
        "ignored_boilerplate_ngrams": len(ignore),
        "eligible_for_validation": eligible,
        "unchecked_no_distinctive_ngrams": sum(f is None for f in fractions.values()),
        "max_shared_fraction_in_val": max(val_fractions),
        "files": files,
        "validation_ids": sorted(p.problem_id for p in val),
        "pyarrow": metadata.version("pyarrow"),
    }


class BuildCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python examples/data_preprocess/m1_rl_data.py", cli_kebab_case=True
    )

    config: Path

    def cli_cmd(self) -> None:
        import yaml

        config = DataConfig.model_validate(yaml.safe_load(self.config.read_text()))
        report = build(config, Path(os.environ["SOE_DATA"]))
        Path(config.report).write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "validation_ids"}, indent=1))


if __name__ == "__main__":
    CliApp.run(BuildCli)
