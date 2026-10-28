"""Build data v2 from v1 by repairing encoding debris in problem text, keeping every split frozen.

v1 (m0_datasets.py, decision 003) stays as built. A full rebuild with clean text would reshuffle
Level B, because its seeded draw runs over the de-duplicated pool, which the repair changes; Level B
and the hard subset are frozen, so v2 repairs the text in place instead (decision 007):

1. Check every v1 file against its committed report.
2. Repair the question text: "\\x0crac" -> "\\\\frac", then remove every other control character
   (C0 except tab and newline, and C1). Evaluation files must come out unchanged.
3. Re-run decision 003's leak checks on the repaired text. Only train problems may be dropped;
   a leak into Level B or the evaluation sets stops the build.
4. Write v2: evaluation files and the hard subset copied byte for byte, Level B repaired (same
   ids, same order), train repaired minus any new duplicates; plus a report with sha256 values.

Run: python examples/data_preprocess/m0_repair_v2.py
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from soe.utils.dataset.decontaminate import (
    NearDuplicateIndex,
    common_ngrams,
    covered_by_union,
    normalize,
)
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl, write_jsonl

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # all but tab and newline


def repair_text(text: str) -> str:
    """Upstream decoded "\\\\frac" as a form feed + "rac"; other control characters are debris
    with no meaning (stray form feeds inside words, bytes around "\\\\le")."""
    return CONTROL_CHARACTERS.sub("", text.replace("\x0crac", "\\frac"))


def repair(problems: list[Problem]) -> tuple[list[Problem], list[str]]:
    out, changed = [], []
    for p in problems:
        text = repair_text(p.question)
        if text != p.question:
            changed.append(p.problem_id)
        out.append(p.model_copy(update={"question": text}))
    return out, changed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-config", type=Path, default=Path("examples/data_preprocess/m0_datasets.yaml")
    )
    parser.add_argument("--v1-report", type=Path, default=Path("docs/reports/m0_datasets_v1.json"))
    parser.add_argument(
        "--hard-subset-report", type=Path, default=Path("docs/reports/m0_hard_subset_v1.json")
    )
    parser.add_argument("--report", type=Path, default=Path("docs/reports/m0_datasets_v2.json"))
    args = parser.parse_args()
    data = Path(os.environ["SOE_DATA"])
    v1_dir, v2_dir = data / "datasets/soe/v1", data / "datasets/soe/v2"
    near = yaml.safe_load(args.build_config.read_text())["near_duplicate"]
    n, frac, max_count = near["ngram"], near["min_fraction"], near["boilerplate_max_count"]

    v1 = json.loads(args.v1_report.read_text())["sha256"]
    bad = [name for name, digest in v1.items() if sha256(v1_dir / f"{name}.jsonl") != digest]
    if bad:
        raise SystemExit(f"v1 files differ from {args.v1_report}: {bad}")
    hard = json.loads(args.hard_subset_report.read_text())
    hard_v1 = data / hard["output"]
    if sha256(hard_v1) != hard["output_sha256"]:
        raise SystemExit("the v1 hard subset differs from its selection report")

    eval_names = sorted(k for k in v1 if k.startswith("eval/"))
    eval_problems: list[Problem] = []
    for name in eval_names:
        problems = read_jsonl(v1_dir / f"{name}.jsonl", Problem)
        if repair(problems)[1]:
            raise SystemExit(f"{name}: evaluation text would change; this repair is for DAPO data")
        eval_problems += problems
    level_b, repaired_b = repair(read_jsonl(v1_dir / "rl/level_b.jsonl", Problem))
    train, repaired_train = repair(read_jsonl(v1_dir / "rl/train.jsonl", Problem))

    # Decision 003's checks on the repaired text. Boilerplate n-grams are recounted over train +
    # Level B (v1 counted the whole raw pool): slightly fewer ignored, so slightly stricter checks.
    ignore = common_ngrams([p.question for p in train + level_b], n, max_count)
    protected = {normalize(p.question) for p in eval_problems + level_b}
    eval_index = NearDuplicateIndex(eval_problems, n=n, min_fraction=frac, ignore=ignore)
    b_index = NearDuplicateIndex(level_b, n=n, min_fraction=frac, ignore=ignore)
    if any(eval_index.matches(p.question) for p in level_b):
        raise SystemExit("a repaired Level B problem near-duplicates an evaluation problem")
    dropped: dict[str, list[str]] = {
        "exact_duplicate": [],
        "near_duplicate_of_eval": [],
        "near_duplicate_of_level_b": [],
    }
    kept: list[Problem] = []
    seen: set[str] = set()
    for p in train:
        key = normalize(p.question)
        if key in protected or key in seen:
            dropped["exact_duplicate"].append(p.problem_id)
        elif eval_index.matches(p.question):
            dropped["near_duplicate_of_eval"].append(p.problem_id)
        elif b_index.matches(p.question):
            dropped["near_duplicate_of_level_b"].append(p.problem_id)
        else:
            kept.append(p)
        seen.add(key)
    residual = covered_by_union(
        eval_problems + level_b, kept, n=n, min_fraction=frac, ignore=ignore
    )
    if residual:
        raise SystemExit(f"train still covers evaluation or Level B problems: {residual}")

    files: dict[str, Path] = {}
    for name in eval_names:
        files[name] = v2_dir / f"{name}.jsonl"
        files[name].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v1_dir / f"{name}.jsonl", files[name])
    files["rl/level_b"], files["rl/train"] = v2_dir / "rl/level_b.jsonl", v2_dir / "rl/train.jsonl"
    write_jsonl(files["rl/level_b"], level_b)
    write_jsonl(files["rl/train"], kept)
    hard_v2 = v2_dir / "eval/hard_subset.jsonl"
    shutil.copyfile(hard_v1, hard_v2)
    report: dict[str, Any] = {
        "version": "v2",
        "derived_from": {"report": str(args.v1_report), "sha256": sha256(args.v1_report)},
        "repair": "\\x0crac -> \\frac; then remove C0 (except tab, newline) and C1 controls",
        "counts": {name: len(read_jsonl(path, Problem)) for name, path in files.items()},
        "repaired_ids": {"rl/level_b": repaired_b, "rl/train": repaired_train},
        "dropped_from_train": {k: len(v) for k, v in dropped.items()},
        "dropped_ids": dropped,
        "ignored_boilerplate_ngrams": len(ignore),
        "residual_covered_by_train": residual,
        "sha256": {name: sha256(path) for name, path in files.items()},
        "hard_subset": {"path": str(hard_v2.relative_to(data)), "sha256": sha256(hard_v2)},
    }
    text = json.dumps(report, indent=2) + "\n"
    (v2_dir / "build_report.json").write_text(text)
    args.report.write_text(text)
    print(
        json.dumps(
            {k: report[k] for k in ("counts", "dropped_from_train", "ignored_boilerplate_ngrams")},
            indent=1,
        )
    )
    print("repaired:", {k: len(v) for k, v in report["repaired_ids"].items()})
    print("sha256 changed vs v1:", sorted(k for k in v1 if report["sha256"][k] != v1[k]))


if __name__ == "__main__":
    main()
