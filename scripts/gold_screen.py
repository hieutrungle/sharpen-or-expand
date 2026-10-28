"""Screen the suite for doubtful gold answers: problems where the base model agrees with itself
on one answer that the verifier marks wrong.

A confident, consistent answer that disagrees with gold is either a model blind spot, a verifier
false negative (an equivalent form it misses) or a wrong or ambiguous gold answer. The M0.8 audit
found the last two by sampling; this lists every problem that shows the pattern so each can be
reviewed. Answers are grouped after removing spacing only (not by verifier equivalence), so the
list errs toward missing a few cases rather than inventing agreement.

    python scripts/gold_screen.py --samples $SOE_DATA/eval/<run> [--samples <dir> ...] --out <dir>
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl

MIN_SAMPLES = 16
MIN_AGREEMENT = 0.75  # share of all samples that give the modal answer
_SPACING = re.compile(r"\\[,;:! ]|\\left|\\right|\\displaystyle|\\quad|\\qquad|\$|\s")


def norm(answer: str) -> str:
    return (
        _SPACING.sub("", answer)
        .replace("\\dfrac", "\\frac")
        .replace("\\tfrac", "\\frac")
        .rstrip(".")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--samples", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    hard = set()
    if suite.hard_subset is not None:
        hard = {p.problem_id for p in read_jsonl(data_root / suite.hard_subset.path, Problem)}
        hard -= set(suite.removed_from_hard_subset)
    problems = {
        p.problem_id: (p, s.name, s.level.value)
        for s in suite.problem_sets
        for p in read_jsonl(data_root / s.path, Problem)
        if p.problem_id not in suite.excluded_problems
    }
    answers: dict[str, Counter[str]] = defaultdict(Counter)
    raw: dict[tuple[str, str], str] = {}
    wrong: dict[tuple[str, str], bool] = {}
    total: Counter[str] = Counter()
    truncated: Counter[str] = Counter()
    for directory in args.samples:
        for shard in sorted(directory.glob("samples-*-of-*.jsonl")):
            with shard.open() as f:
                for line in f:
                    r = json.loads(line)
                    pid = r["problem_id"]
                    if pid not in problems:
                        continue
                    total[pid] += 1
                    if r["finish_reason"] != "stop" or r["extracted_answer"] is None:
                        truncated[pid] += r["finish_reason"] != "stop"
                        continue
                    key = norm(r["extracted_answer"])
                    answers[pid][key] += 1
                    raw.setdefault((pid, key), r["extracted_answer"])
                    wrong[(pid, key)] = not r["correct"]
    flagged = []
    for pid, counts in answers.items():
        if total[pid] < MIN_SAMPLES:
            continue
        key, count = counts.most_common(1)[0]
        if count / total[pid] >= MIN_AGREEMENT and wrong[(pid, key)]:
            problem, set_name, level = problems[pid]
            flagged.append(
                {
                    "problem_id": pid,
                    "problem_set": set_name,
                    "level": level,
                    "hard": pid in hard,
                    "gold": problem.answer,
                    "modal_answer": raw[(pid, key)],
                    "modal_count": count,
                    "samples": total[pid],
                    "truncated": truncated[pid],
                    "other_answers": [[raw[(pid, k)], c] for k, c in counts.most_common(4)[1:]],
                    "question": problem.question,
                }
            )
    flagged.sort(key=lambda x: (x["level"], x["problem_set"], x["problem_id"]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "flagged.jsonl").open("w") as f:
        for item in flagged:
            f.write(json.dumps(item) + "\n")
    by_set = Counter(x["problem_set"] for x in flagged)
    print(f"{len(flagged)} of {len(problems)} problems flagged")
    print("per set:", dict(sorted(by_set.items())))
    print("in the hard subset:", sum(1 for x in flagged if x["hard"]))


if __name__ == "__main__":
    main()
