"""RL train v3 (decision 011): v2 train minus the problems the hand review marked as copies of an
evaluation or Level B problem. Nothing else changes: Level B and the evaluation files stay v2.

    SOE_DATA=... python examples/data_preprocess/m0_train_v3.py
reads datasets/soe/v2/rl/train.jsonl and docs/reports/m0_train_v3_review.json, writes
datasets/soe/v3/rl/train.jsonl and docs/reports/m0_datasets_v3.json (counts, dropped ids, hashes).
"""

import json
import os
from pathlib import Path

from soe.trainer.config.eval_suite import file_sha256
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl, write_jsonl

REVIEW = Path("docs/reports/m0_train_v3_review.json")
REPORT = Path("docs/reports/m0_datasets_v3.json")
SOURCE = "datasets/soe/v2/rl/train.jsonl"
TARGET = "datasets/soe/v3/rl/train.jsonl"


def reviewed_drops(review: dict[str, list[dict[str, str]]]) -> set[str]:
    """Train ids with at least one pair judged the same problem as its reference."""
    verdicts = {p["verdict"] for p in review["pairs"]}
    if not verdicts <= {"drop", "keep"}:
        raise ValueError(f"unknown verdicts: {verdicts - {'drop', 'keep'}}")
    return {p["train"] for p in review["pairs"] if p["verdict"] == "drop"}


def split(problems: list[Problem], drops: set[str]) -> tuple[list[Problem], list[Problem]]:
    missing = drops - {p.problem_id for p in problems}
    if missing:
        raise ValueError(f"reviewed ids not in the train pool: {sorted(missing)[:5]}")
    kept = [p for p in problems if p.problem_id not in drops]
    dropped = [p for p in problems if p.problem_id in drops]
    return kept, dropped


def main() -> None:
    root = Path(os.environ["SOE_DATA"])
    review = json.loads(REVIEW.read_text())
    kept, dropped = split(read_jsonl(root / SOURCE, Problem), reviewed_drops(review))
    (root / TARGET).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / TARGET, kept)
    report = {
        "version": "v3",
        "derived_from": {"path": SOURCE, "sha256": file_sha256(root / SOURCE)},
        "review": {"path": str(REVIEW), "sha256": file_sha256(REVIEW)},
        "counts": {"rl/train": len(kept), "dropped": len(dropped)},
        "dropped_ids": sorted(p.problem_id for p in dropped),
        "sha256": {"rl/train": file_sha256(root / TARGET)},
        "unchanged": "eval/* and rl/level_b stay in datasets/soe/v2 (decision 011)",
    }
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_ids"}, indent=1))


if __name__ == "__main__":
    main()
