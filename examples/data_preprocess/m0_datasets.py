"""Build the M0 problem files: Level A/C evaluation sets, the RL training pool and Level B.

Reads the staged Hub snapshots offline (scripts/stage_offline.py), applies decisions 002 and 003,
and writes Problem JSONL files under $SOE_DATA/<output_dir>/:

    eval/<name>.jsonl      one file per evaluation set (Level A, C1, C2)
    rl/train.jsonl         RL training pool
    rl/level_b.jsonl       Level B held-out split
    build_report.json      counts, dropped ids per reason, sha256 per file (no problem text)

The report is also written to --report so it can be committed (problem text never is).
"""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict

from soe.utils.dataset.decontaminate import covered_by_union
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.splits import build_rl_splits
from soe.utils.jsonl import write_jsonl

DAPO_WRAPPER_START = "Solve the following math problem step by step."


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    hf_id: str
    file: str
    question: str
    answer: str
    id: str
    level: Level = Level.TRAIN
    exclude: dict[str, str] = {}  # source id -> reason


class NearDuplicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ngram: int
    min_fraction: float
    boilerplate_max_count: int


class LevelB(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int
    seed: int


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    output_dir: str
    near_duplicate: NearDuplicate
    level_b: LevelB
    eval_sets: list[Source]
    rl_pool: Source


def pinned_revisions(assets_path: Path) -> dict[str, str]:
    assets = yaml.safe_load(assets_path.read_text())["assets"]
    return {a["hf_id"]: a["revision"] for a in assets}


def read_frame(source: Source, revision: str) -> pd.DataFrame:
    path = hf_hub_download(
        source.hf_id, source.file, repo_type="dataset", revision=revision, local_files_only=True
    )
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True)


def field_value(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for key in dotted.split("."):
        value = value[key]
    return value


def clean_answer(value: Any) -> str:
    """Gold answers arrive as int, str or a one-element array of str; strip $...$ delimiters."""
    if not isinstance(value, (str, int)):
        values = list(value)
        if len(values) != 1:
            raise ValueError(f"expected one answer, got {values!r}")
        value = values[0]
    text = str(value).strip()
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    if not text:
        raise ValueError("empty answer")
    return text


def clean_question(text: str) -> str:
    if text.startswith(DAPO_WRAPPER_START):
        raise ValueError("DAPO instruction wrapper found; expected the bare problem")
    return text.strip()


def load_problems(source: Source, revision: str) -> tuple[list[Problem], list[str]]:
    """Problems of one source, and the ids excluded by the config."""
    problems: list[Problem] = []
    excluded: list[str] = []
    for row in read_frame(source, revision).to_dict("records"):
        source_id = str(field_value(row, source.id))
        problem_id = f"{source.name}/{source_id}"
        if source_id in source.exclude:
            excluded.append(problem_id)
            continue
        problems.append(
            Problem(
                problem_id=problem_id,
                source=source.name,
                level=source.level,
                question=clean_question(str(field_value(row, source.question))),
                answer=clean_answer(field_value(row, source.answer)),
            )
        )
    missing = set(source.exclude) - {e.split("/", 1)[1] for e in excluded}
    if missing:
        raise ValueError(f"{source.name}: excluded ids not found: {sorted(missing)}")
    ids = [p.problem_id for p in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source.name}: duplicate problem ids")
    return problems, excluded


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("scripts/stage_assets.yaml"))
    parser.add_argument("--report", type=Path, default=Path("docs/reports/m0_datasets_v1.json"))
    args = parser.parse_args()
    config = BuildConfig.model_validate(yaml.safe_load(args.config.read_text()))
    revisions = pinned_revisions(args.assets)
    out = Path(os.environ["SOE_DATA"]) / config.output_dir
    files: dict[str, Path] = {}
    counts: dict[str, int] = {}
    excluded: dict[str, list[str]] = {}

    eval_problems: list[Problem] = []
    for source in config.eval_sets:
        problems, dropped = load_problems(source, revisions[source.hf_id])
        files[f"eval/{source.name}"] = out / "eval" / f"{source.name}.jsonl"
        write_jsonl(files[f"eval/{source.name}"], problems)
        counts[f"eval/{source.name}"] = len(problems)
        excluded.update({pid: source.exclude[pid.split("/", 1)[1]] for pid in dropped})
        eval_problems += problems

    pool, _ = load_problems(config.rl_pool, revisions[config.rl_pool.hf_id])
    splits = build_rl_splits(
        pool,
        eval_problems,
        level_b_size=config.level_b.size,
        seed=config.level_b.seed,
        ngram=config.near_duplicate.ngram,
        min_fraction=config.near_duplicate.min_fraction,
        boilerplate_max_count=config.near_duplicate.boilerplate_max_count,
    )
    files["rl/train"] = out / "rl" / "train.jsonl"
    files["rl/level_b"] = out / "rl" / "level_b.jsonl"
    write_jsonl(files["rl/train"], splits.train)
    write_jsonl(files["rl/level_b"], splits.level_b)
    counts |= {"rl/pool_raw": len(pool), "rl/train": len(splits.train)}
    counts["rl/level_b"] = len(splits.level_b)

    report = {
        "version": config.version,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": json.loads(config.model_dump_json()),
        "revisions": {s.hf_id: revisions[s.hf_id] for s in [*config.eval_sets, config.rl_pool]},
        "counts": counts,
        "excluded_by_config": excluded,
        "dropped_from_rl_pool": {k: len(v) for k, v in splits.dropped.items()},
        "dropped_ids": splits.dropped,
        "ignored_boilerplate_ngrams": len(splits.ignored_ngrams),
        # Final audit: evaluation and Level B problems the train pool covers as a whole (want 0).
        "residual_covered_by_train": covered_by_union(
            eval_problems + splits.level_b,
            splits.train,
            n=config.near_duplicate.ngram,
            min_fraction=config.near_duplicate.min_fraction,
            ignore=splits.ignored_ngrams,
        ),
        "sha256": {name: sha256(path) for name, path in files.items()},
    }
    text = json.dumps(report, indent=2) + "\n"
    (out / "build_report.json").write_text(text)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    for name, n in counts.items():
        print(f"{name:24s} {n:6d}")
    print("dropped from RL pool:", report["dropped_from_rl_pool"])
    print("ignored boilerplate n-grams:", report["ignored_boilerplate_ngrams"])
    print("residual covered by train:", report["residual_covered_by_train"])
    print(f"wrote {out} and {args.report}")


if __name__ == "__main__":
    main()
