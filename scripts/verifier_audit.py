"""Draw the M0.8 verifier audit: ~200 base-model answers where a math verifier is most likely wrong.

Streams the SampleRecord shards of the 4B-Base evaluation (seed 0, all levels) and the M0.6
selection run (seed 1, Levels A and B), keeps one candidate per distinct (problem, extracted
answer), tags each with risk categories and draws a fixed number per category (seeded, one item
per problem).
Only completions that stopped count for equivalence: truncated ones are incorrect by rule.

Writes items.jsonl (problem text included: keep it out of git, MathArena sets are CC-BY-NC-SA) and
pool.json (candidate counts per category). Run on FASTER, where the samples are:
    sbatch slurm/cpu.sbatch python scripts/verifier_audit.py --out $SOE_DATA/audit/verifier_v1
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl

# Items per category; categories are drawn in this order and a problem is used once.
QUOTAS = {
    "multi_answer": 30,  # gold is a list, tuple, set or interval
    "units_text": 20,  # \text, \mathrm, degrees, percent in gold or answer
    "false_positive_risk": 45,  # judged correct, strings differ
    "false_negative_risk": 55,  # judged incorrect, close to gold
    "extraction": 20,  # several different \boxed{} in one completion
    "control_correct": 10,
    "control_incorrect": 10,
}
LEVEL_C_ITEMS = 10  # drawn from Level C candidates, any category, on top of the quotas
SEED = 20261028
SOURCES = {
    "eval_seed0": "eval/qwen35-4b-base_soe-eval-v1",
    "selection_seed1": "selection/hard_subset_v1",
}
_SPACING = re.compile(r"\\[,;:! ]|\\left|\\right|\\displaystyle|\$|\s")
_TEXT = re.compile(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}")
Candidate = dict[str, Any]
_UNITS = re.compile(r"\\text|\\mathrm|\\circ|\\%|%|degree", re.IGNORECASE)


def norm(answer: str) -> str:
    """Formatting-insensitive form, only to decide whether two answers *look* the same."""
    text = _SPACING.sub("", answer).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return text.rstrip(".")


def strip_text(answer: str) -> str:
    return norm(_TEXT.sub("", answer))


def digits(answer: str) -> str:
    return "".join(ch for ch in answer if ch.isdigit())


def parts(answer: str) -> list[str]:
    """Comma-separated parts with outer brackets removed, sorted: order-insensitive comparison."""
    text = norm(answer).strip("()[]{}").replace("\\{", "").replace("\\}", "")
    return sorted(p for p in text.split(",") if p)


def all_boxed(text: str) -> list[str]:
    """Contents of every \\boxed{...} / \\fbox{...}, in order (nested braces handled)."""
    found = []
    for match in re.finditer(r"\\(?:boxed|fbox)\{", text):
        depth = 0
        for i in range(match.end() - 1, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                found.append(text[match.end() : i].strip())
                break
    return found


def categories(gold: str, extracted: str, correct: bool, boxes: list[str]) -> list[str]:
    tags = []
    if "," in gold or "\\cup" in gold or re.search(r"^[\[(].*[\])]$", norm(gold)):
        tags.append("multi_answer")
    if _UNITS.search(gold) or _UNITS.search(extracted):
        tags.append("units_text")
    same_look = norm(gold) == norm(extracted)
    if correct and not same_look:
        tags.append("false_positive_risk")
    if not correct and extracted:
        g, e = norm(gold), norm(extracted)
        close = (
            (digits(g) != "" and digits(g) == digits(e))
            or (len(g) > 1 and (g in e or e in g))
            or strip_text(gold) == strip_text(extracted)
            or (len(parts(gold)) > 1 and parts(gold) == parts(extracted))
        )
        if close:
            tags.append("false_negative_risk")
    if len({norm(b) for b in boxes}) > 1:
        tags.append("extraction")
    tags.append("control_correct" if correct else "control_incorrect")
    return tags


def context(text: str) -> str:
    """The end of the completion up to the last box: enough to judge extraction and intent."""
    end = text.rfind("\\boxed")
    end = len(text) if end == -1 else min(len(text), end + 300)
    return text[max(0, end - 900) : end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    problems: dict[str, tuple[Problem, str, str]] = {}
    for problem_set in suite.problem_sets:
        for problem in read_jsonl(data_root / problem_set.path, Problem):
            problems[problem.problem_id] = (problem, problem_set.name, problem_set.level.value)

    # (problem_id, extracted) -> candidate; counts per problem for the "how common" column
    candidates: dict[tuple[str, str], Candidate] = {}
    samples_per_problem: dict[str, int] = defaultdict(int)
    for source, rel in SOURCES.items():
        for shard in sorted((data_root / rel).glob("samples-*-of-*.jsonl")):
            with shard.open() as f:
                for line in f:
                    r = json.loads(line)
                    samples_per_problem[r["problem_id"]] += 1
                    if r["finish_reason"] != "stop" or r["extracted_answer"] is None:
                        continue
                    key = (r["problem_id"], r["extracted_answer"])
                    if key in candidates:
                        candidates[key]["answer_count"] += 1
                        continue
                    problem, set_name, level = problems[r["problem_id"]]
                    boxes = all_boxed(r["completion"])
                    candidates[key] = {
                        "problem_id": r["problem_id"],
                        "problem_set": set_name,
                        "level": level,
                        "source": source,
                        "question": problem.question,
                        "gold": problem.answer,
                        "extracted": r["extracted_answer"],
                        "verifier_correct": r["correct"],
                        "answer_count": 1,
                        "distinct_boxes": list(dict.fromkeys(boxes)),
                        "context": context(r["completion"]),
                        "categories": categories(
                            problem.answer, r["extracted_answer"], r["correct"], boxes
                        ),
                    }

    pool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for key in sorted(candidates):
        cand = candidates[key]
        for tag in cand["categories"]:
            pool[tag][cand["level"]] += 1
            by_category[tag].append(cand)

    rng = random.Random(SEED)
    used: set[str] = set()
    items: list[Candidate] = []

    def draw(pool_items: list[Candidate], k: int, category: str) -> None:
        fresh = [c for c in pool_items if c["problem_id"] not in used]
        for cand in rng.sample(fresh, min(k, len(fresh))):
            used.add(cand["problem_id"])
            items.append(cand | {"category": category})

    level_c = [
        c
        for c in sorted(candidates.values(), key=lambda c: (c["problem_id"], c["extracted"]))
        if c["level"] in ("C1", "C2") and c["categories"][0] != "control_correct"
    ]
    draw(level_c, LEVEL_C_ITEMS, "level_c")
    for category, quota in QUOTAS.items():
        draw([c for c in by_category[category] if c["level"] not in ("C1", "C2")], quota, category)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "items.jsonl").open("w") as f:
        for i, item in enumerate(items, start=1):
            f.write(json.dumps({"item_id": f"V{i:03d}"} | item) + "\n")
    summary = {
        "seed": SEED,
        "candidates": len(candidates),
        "items": len(items),
        "items_per_category": {
            c: sum(i["category"] == c for i in items) for c in ["level_c", *QUOTAS]
        },
        "pool_per_category_and_level": {k: dict(v) for k, v in sorted(pool.items())},
    }
    (args.out / "pool.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
