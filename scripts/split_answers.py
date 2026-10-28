"""How often an answer is split over several \\boxed{} and the last box alone is judged wrong.

Decision 004 scores the last box only. Some completions box the parts of a multi-part answer
separately ("\\boxed{864}, \\boxed{1350}, \\boxed{2400}"). For every stopped sample whose last box
is wrong and that has two or more distinct boxes, this checks whether the last j boxes joined with
commas (j = 2, 3, 4) match the gold, and, for comparison, whether any earlier single box does (a
model that changed its mind; not a reason to accept). Counts only; examples list sample ids.

    python scripts/split_answers.py --samples $SOE_DATA/eval/<run> --out <file>.json
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl
from soe.utils.reward_score.math_verify import all_boxed, answers_equivalent

JOINS = (2, 3, 4)
EXAMPLES = 40


def joined_match(boxes: list[str], gold: str) -> int | None:
    """Smallest j for which the last j boxes, joined with commas, match the gold."""
    for j in JOINS:
        if len(boxes) >= j and answers_equivalent(", ".join(boxes[-j:]), gold):
            return j
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    problems = {
        p.problem_id: (p, s.name, s.level.value)
        for s in suite.problem_sets
        for p in read_jsonl(root / s.path, Problem)
        if p.problem_id not in suite.excluded_problems
    }
    counts: Counter[str] = Counter()
    by_set: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    samples_per_level: Counter[str] = Counter()
    problems_hit: set[str] = set()
    multi_part_gold = 0
    examples: list[list[str | int]] = []
    for shard in sorted(args.samples.glob("samples-*-of-*.jsonl")):
        with shard.open() as f:
            for line in f:
                r = json.loads(line)
                if r["problem_id"] not in problems:
                    continue
                problem, set_name, level = problems[r["problem_id"]]
                samples_per_level[level] += 1
                counts["samples"] += 1
                if r["finish_reason"] != "stop":
                    continue
                counts["stopped"] += 1
                boxes = all_boxed(r["completion"])
                distinct = list(dict.fromkeys(b for b in boxes if b))
                if len(boxes) >= 2:
                    counts["stopped_with_2+_boxes"] += 1
                if len(distinct) < 2:
                    continue
                counts["stopped_with_2+_distinct_boxes"] += 1
                if r["correct"]:
                    continue
                counts["last_box_wrong_with_2+_distinct_boxes"] += 1
                j = joined_match(boxes, problem.answer)
                if j is not None:
                    counts[f"recovered_by_joining_last_{j}"] += 1
                    by_set[set_name] += 1
                    by_level[level] += 1
                    problems_hit.add(r["problem_id"])
                    multi_part_gold += "," in problem.answer
                    if len(examples) < EXAMPLES:
                        examples.append([r["problem_id"], r["sample_index"], j])
                elif any(answers_equivalent(b, problem.answer) for b in distinct[:-1]):
                    counts["an_earlier_single_box_matches"] += 1
    recovered = sum(by_level.values())
    report = {
        "samples_dir": str(args.samples),
        "counts": dict(counts),
        "recovered_samples": recovered,
        "recovered_with_multi_part_gold": multi_part_gold,
        "recovered_per_set": dict(sorted(by_set.items())),
        "pass@1_gain_per_level": {
            level: round(by_level[level] / samples_per_level[level], 5)
            for level in sorted(samples_per_level)
        },
        "problems_with_a_recovered_sample": len(problems_hit),
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
