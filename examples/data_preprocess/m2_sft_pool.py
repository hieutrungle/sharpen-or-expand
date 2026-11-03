"""M2.2: the SFT problem pool (roadmap D1 (c), D2): DeepMath-103K problems as `Problem` records.

    SOE_DATA=... python examples/data_preprocess/m2_sft_pool.py \
        --config examples/data_preprocess/m2_sft_pool.yaml --report docs/reports/m2_sft_pool.json

Reads only the question, final answer, difficulty and topic columns of the staged snapshot (the
R1 solutions are never used, D1). A problem is kept when math-verify parses its gold answer to a
mathematical object (not just a string), the answer is not a yes/no or true/false guess and holds
no English word (a correct sample could be a coin flip, and words compare as products of symbols;
roadmap risk 8), and the question has no CJK characters. Exact
duplicate questions keep their first copy. Ids are content hashes, `deepmath/<16 hex>`, so they do
not depend on row order. Writes `pool.jsonl` (Problem lines) and `pool_meta.jsonl` (difficulty and
topic per id) under `output_dir`; the report holds counts, drops by reason and sha256s, no text.
The decontamination of this pool is M2.3.
"""

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from math_verify import parse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import file_sha256
from soe.utils.dataset.problems import Level, Problem
from soe.utils.jsonl import write_jsonl

COLUMNS = ["question", "final_answer", "difficulty", "topic"]
_BINARY = re.compile(r"^(?:\\text\{)?\s*(yes|no|true|false)\s*\}?\.?$", re.IGNORECASE)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_COMMAND = re.compile(r"\\[A-Za-z]+")
_LETTERS = re.compile(r"[A-Za-z]{3,}")
FUNCTIONS = frozenset({"sin", "cos", "tan", "cot", "sec", "csc", "log", "exp", "max", "min", "gcd"})


class PoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    hf_id: str
    revision: str = Field(min_length=40, max_length=40)
    source_released: str  # dates for the Level C check (M2.3), from the dataset card / Hub API
    source_last_modified: str
    id_prefix: str
    output_dir: str  # relative to $SOE_DATA


class PoolRow(BaseModel):
    """One line of pool_meta.jsonl."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_id: str
    difficulty: float
    topic: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Pool:
    problems: list[Problem]
    meta: list[PoolRow]
    dropped: Counter[str]
    rows: int


def drop_reason(question: str, answer: str) -> str | None:
    if not question.strip() or not answer.strip():
        return "empty"
    if _CJK.search(question):
        return "non_english"
    if _BINARY.match(answer.strip()):
        return "binary_answer"
    if any(w.lower() not in FUNCTIONS for w in _LETTERS.findall(_COMMAND.sub(" ", answer))):
        return "word_answer"
    parsed = parse(f"\\boxed{{{answer}}}")
    if not parsed or isinstance(parsed[0], str):
        return "gold_unreadable"
    return None


def problem_id(prefix: str, question: str) -> str:
    return f"{prefix}/{hashlib.sha256(question.encode()).hexdigest()[:16]}"


def build_pool(rows: list[dict[str, object]], prefix: str) -> Pool:
    problems: list[Problem] = []
    meta: list[PoolRow] = []
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        question, answer = str(row["question"]).strip(), str(row["final_answer"]).strip()
        reason = drop_reason(question, answer)
        pid = problem_id(prefix, question)
        if reason is None and pid in seen:
            reason = "duplicate_question"
        if reason is not None:
            dropped[reason] += 1
            continue
        seen.add(pid)
        problems.append(
            Problem(
                problem_id=pid, source=prefix, level=Level.TRAIN, question=question, answer=answer
            )
        )
        meta.append(
            PoolRow(
                problem_id=pid, difficulty=float(str(row["difficulty"])), topic=str(row["topic"])
            )
        )
    return Pool(problems=problems, meta=meta, dropped=dropped, rows=len(rows))


def read_rows(config: PoolConfig) -> list[dict[str, object]]:
    import pyarrow.parquet as pq  # data extra / image only; CI installs CPU deps without it
    from huggingface_hub import snapshot_download  # offline, from the staged snapshot

    snapshot = Path(
        snapshot_download(
            config.hf_id, repo_type="dataset", revision=config.revision, local_files_only=True
        )
    )
    files = sorted((snapshot / "data").glob("train-*.parquet"))
    rows: list[dict[str, object]] = []
    for path in files:
        rows += pq.read_table(path, columns=COLUMNS).to_pylist()
    return rows


class PoolCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="examples/data_preprocess/m2_sft_pool.py", cli_kebab_case=True
    )

    config: Path
    report: Path

    def cli_cmd(self) -> None:
        config = PoolConfig.model_validate(yaml.safe_load(self.config.read_text()))
        pool = build_pool(read_rows(config), config.id_prefix)
        out = Path(os.environ["SOE_DATA"]) / config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "pool.jsonl", pool.problems)
        write_jsonl(out / "pool_meta.jsonl", pool.meta)
        difficulty = Counter(f"{m.difficulty:g}" for m in pool.meta)
        report = {
            "version": config.version,
            "source": {
                "hf_id": config.hf_id,
                "revision": config.revision,
                "released": config.source_released,
                "last_modified": config.source_last_modified,
            },
            "rows": pool.rows,
            "kept": len(pool.problems),
            "dropped": dict(sorted(pool.dropped.items())),
            "kept_by_difficulty": dict(sorted(difficulty.items(), key=lambda kv: float(kv[0]))),
            "kept_by_topic_top20": dict(Counter(m.topic for m in pool.meta).most_common(20)),
            "files": {
                name: {
                    "path": f"{config.output_dir}/{name}",
                    "sha256": file_sha256(out / name),
                }
                for name in ("pool.jsonl", "pool_meta.jsonl")
            },
        }
        self.report.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "kept_by_topic_top20"}, indent=1))


if __name__ == "__main__":
    CliApp.run(PoolCli)
