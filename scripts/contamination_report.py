"""M0.9: how close the RL training pool comes to every evaluation problem (and Level B).

The build (decisions 002/003) already drops any training problem that near-duplicates an
evaluation or Level B problem (word 13-grams, >= 50% of the reference's n-grams) and audits the
residual. This report re-checks the final files and adds a sensitivity analysis: for each
reference problem, the largest share of its n-grams found in any single training problem, at
13-grams and 8-grams, and whether that closest training problem has the same answer. Ids and
numbers only: MathArena problem text is CC-BY-NC-SA and never goes into git.

    python scripts/contamination_report.py --out docs/reports/m0_contamination_rl.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.decontaminate import Ngram, common_ngrams, covered_by_union, word_ngrams
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl
from soe.utils.reward_score.math_verify import answers_equivalent

BOILERPLATE_MAX_COUNT = 20  # as in the build config (near_duplicate.boilerplate_max_count)
FROZEN = (13, 0.5)  # the build's rule
THRESHOLDS = (0.2, 0.3, 0.5)
TOP = 10  # closest pairs listed per n
# A suspect: the closest training problem has the same answer and a large share of the
# reference's n-grams (a reworded copy the frozen rule lets through). Reviewed by hand.
SUSPECT = {13: 0.3, 8: 0.5}


def closest(
    references: list[Problem], corpus: list[Problem], n: int, ignore: frozenset[Ngram]
) -> dict[str, tuple[float, str]]:
    """For each reference: (largest share of its n-grams in one corpus problem, that problem)."""
    owners: dict[Ngram, list[int]] = defaultdict(list)
    for i, problem in enumerate(corpus):
        for gram in word_ngrams(problem.question, n) - ignore:
            owners[gram].append(i)
    out = {}
    for ref in references:
        grams = word_ngrams(ref.question, n) - ignore
        hits: dict[int, int] = defaultdict(int)
        for gram in grams:
            for i in owners.get(gram, ()):
                hits[i] += 1
        if not grams or not hits:
            out[ref.problem_id] = (0.0, "")
            continue
        best = max(hits, key=lambda i: (hits[i], -i))
        out[ref.problem_id] = (hits[best] / len(grams), corpus[best].problem_id)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    sets = {s.name: read_jsonl(root / s.path, Problem) for s in suite.problem_sets}
    level = {s.name: s.level.value for s in suite.problem_sets}
    train = read_jsonl(root / "datasets/soe/v2/rl/train.jsonl", Problem)
    by_id = {p.problem_id: p for p in train} | {p.problem_id: p for ps in sets.values() for p in ps}
    report: dict[str, object] = {
        "train": {"path": "datasets/soe/v2/rl/train.jsonl", "problems": len(train)},
        "frozen_rule": {"ngram": FROZEN[0], "min_fraction": FROZEN[1]},
    }
    references = [p for ps in sets.values() for p in ps]
    suspects: dict[str, str] = {}  # reference -> closest training problem
    for n in (13, 8):
        ignore = common_ngrams([p.question for p in train], n, BOILERPLATE_MAX_COUNT)
        best = closest(references, train, n, ignore)
        per_set = {}
        for name, problems in sets.items():
            shares = [best[p.problem_id][0] for p in problems]
            per_set[name] = {
                "level": level[name],
                "problems": len(problems),
                **{f"max_share>={t}": sum(s >= t for s in shares) for t in THRESHOLDS},
            }
        for ref in references:
            share, tid = best[ref.problem_id]
            if share >= SUSPECT[n] and answers_equivalent(by_id[tid].answer, ref.answer):
                suspects.setdefault(ref.problem_id, tid)
        ranked = sorted(references, key=lambda p: -best[p.problem_id][0])[:TOP]
        pairs = []
        for ref in ranked:
            share, tid = best[ref.problem_id]
            same = bool(tid) and answers_equivalent(by_id[tid].answer, ref.answer)
            pairs.append(
                {
                    "reference": ref.problem_id,
                    "train": tid,
                    "share": round(share, 3),
                    "same_answer": same,
                }
            )
        report[f"{n}-gram"] = {
            "ignored_boilerplate": len(ignore),
            "per_set": per_set,
            "closest": pairs,
        }
        if n == FROZEN[0]:
            report["residual_at_frozen_rule"] = {
                "single_problem": sorted(
                    p.problem_id for p in references if best[p.problem_id][0] >= FROZEN[1]
                ),
                "covered_by_union": covered_by_union(
                    references, train, n=n, min_fraction=FROZEN[1], ignore=ignore
                ),
            }
    set_of = {p.problem_id: name for name, ps in sets.items() for p in ps}
    suspect_counts = {name: sum(set_of[r] == name for r in suspects) for name in sets}
    report["suspects"] = {
        "rule": {f"{n}-gram share": t for n, t in SUSPECT.items()} | {"same_answer": True},
        "per_set": {name: count for name, count in suspect_counts.items() if count},
        "pairs": [{"reference": r, "train": t} for r, t in sorted(suspects.items())],
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print("suspects:", {name: count for name, count in suspect_counts.items() if count})
    print(json.dumps(report["residual_at_frozen_rule"]))
    for n in (13, 8):
        block = report[f"{n}-gram"]
        assert isinstance(block, dict)
        print(f"{n}-gram:", {k: v for k, v in block["per_set"].items() if v["max_share>=0.2"]})
        for pair in block["closest"]:
            print("  ", pair)


if __name__ == "__main__":
    main()
