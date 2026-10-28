"""M0.9: how close the RL training pool comes to every evaluation problem (and Level B).

The build (decisions 002/003) drops any training problem that near-duplicates an evaluation or
Level B problem (word 13-grams, >= 50% of the reference's n-grams) and audits the residual. This
report re-checks the final files and then lists every training problem that shares at least 20%
of a reference's 13-grams or 8-grams, classified so that deletion options can be weighed:

- E: identical after formatting (math-aware text similarity >= 0.9 and the same numbers);
- R: the same numbers but reworded, or DAPO's answer re-encoding ("give m + n");
- V: different numbers (a variant, or an unrelated match on shared phrasing).

Ids and numbers only: MathArena problem text is CC-BY-NC-SA and never goes into git.

    python scripts/contamination_report.py --out docs/reports/m0_contamination_rl.json
"""

import argparse
import difflib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.decontaminate import Ngram, common_ngrams, covered_by_union, word_ngrams
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl
from soe.utils.reward_score.math_verify import answers_equivalent

BOILERPLATE_MAX_COUNT = 20  # as in the build config (near_duplicate.boilerplate_max_count)
FROZEN = (13, 0.5)  # the build's rule
MIN_SHARE = 0.2  # candidate pairs listed from here
# DAPO appends answer-format instructions; they are not part of the problem.
_SUFFIX = re.compile(
    r"(the (original )?answer (is|should be) (in|of) the form|please provide|please give|"
    r"provide the (value|sum|number)|express your answer|the final answer).*$",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True, kw_only=True)
class Pair:
    reference: str
    level: str
    train: str
    share_13: float
    share_8: float
    same_answer: bool
    category: str  # E, R or V (module docstring)

    @property
    def share(self) -> float:
        return max(self.share_13, self.share_8)


def math_text(question: str) -> str:
    """Question text with formatting differences removed (dollar signs, spacing commands,
    \\frac{a}{b} vs a/b, dots, comparison commands, text wrappers, whitespace, braces)."""
    q = _SUFFIX.sub("", question)
    q = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", q)
    q = re.sub(r"\\(left|right|displaystyle|,|;|!|quad|qquad)|\$|\\\(|\\\)|\\\[|\\\]", " ", q)
    q = re.sub(r"\\(cdots|ldots|dots)", "...", q)
    q = re.sub(r"\\leq?\b", "<=", q)
    q = re.sub(r"\\geq?\b", ">=", q)
    q = re.sub(r"\\(overline|mathrm|text|textbf|operatorname)\{([^{}]*)\}", r"\2", q)
    return re.sub(r"[\s{}]+", "", q).lower()


def same_numbers(a: str, b: str) -> bool:
    na, nb = (
        Counter(_NUMBER.findall(_SUFFIX.sub("", a))),
        Counter(_NUMBER.findall(_SUFFIX.sub("", b))),
    )
    return sum((na & nb).values()) >= 0.9 * max(sum(na.values()), sum(nb.values()), 1)


def category(reference: Problem, train: Problem) -> str:
    if not same_numbers(reference.question, train.question):
        return "V"
    similarity = difflib.SequenceMatcher(
        None, math_text(reference.question), math_text(train.question), autojunk=False
    ).ratio()
    return "E" if similarity >= 0.9 else "R"


def shares(
    references: list[Problem], corpus: list[Problem], n: int, ignore: frozenset[Ngram]
) -> dict[tuple[str, str], float]:
    """(reference, corpus problem) -> share of the reference's n-grams in it (>= MIN_SHARE)."""
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
        for i, count in hits.items():
            if count / len(grams) >= MIN_SHARE:
                out[(ref.problem_id, corpus[i].problem_id)] = count / len(grams)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    sets = {s.name: read_jsonl(root / s.path, Problem) for s in suite.problem_sets}
    level_of = {p.problem_id: s.level.value for s in suite.problem_sets for p in sets[s.name]}
    references = [p for ps in sets.values() for p in ps]
    train = read_jsonl(root / "datasets/soe/v2/rl/train.jsonl", Problem)
    by_id = {p.problem_id: p for p in train} | {p.problem_id: p for p in references}

    found: dict[int, dict[tuple[str, str], float]] = {}
    ignores: dict[int, frozenset[Ngram]] = {}
    for n in (13, 8):
        ignores[n] = common_ngrams([p.question for p in train], n, BOILERPLATE_MAX_COUNT)
        found[n] = shares(references, train, n, ignores[n])
    pairs = [
        Pair(
            reference=ref_id,
            level=level_of[ref_id],
            train=train_id,
            share_13=round(found[13].get((ref_id, train_id), 0.0), 3),
            share_8=round(found[8].get((ref_id, train_id), 0.0), 3),
            same_answer=answers_equivalent(by_id[train_id].answer, by_id[ref_id].answer),
            category=category(by_id[ref_id], by_id[train_id]),
        )
        for ref_id, train_id in sorted(found[13].keys() | found[8].keys())
    ]
    options: dict[str, Callable[[Pair], bool]] = {
        "1_identical_text (E)": lambda p: p.category == "E",
        "2_same_problem (E, R, or same answer)": lambda p: p.category in "ER" or p.same_answer,
        "3_plus_close_variants (option 2, or share >= 0.3)": lambda p: (
            p.category in "ER" or p.same_answer or p.share >= 0.3
        ),
        "4_everything_listed (share >= 0.2)": lambda p: True,
    }
    deletion = {}
    for name, rule in options.items():
        chosen = [p for p in pairs if rule(p)]
        levels = Counter(p.level for p in {p.reference: p for p in chosen}.values())
        deletion[name] = {
            "train_dropped": len({p.train for p in chosen}),
            "references_by_level": dict(sorted(levels.items())),
        }
    frozen_hits = sorted({r for (r, _), share in found[13].items() if share >= FROZEN[1]})
    report = {
        "train": {"path": "datasets/soe/v2/rl/train.jsonl", "problems": len(train)},
        "frozen_rule": {"ngram": FROZEN[0], "min_fraction": FROZEN[1]},
        "residual_at_frozen_rule": {
            "single_problem": frozen_hits,
            "covered_by_union": covered_by_union(
                references, train, n=FROZEN[0], min_fraction=FROZEN[1], ignore=ignores[13]
            ),
        },
        "candidate_pairs": len(pairs),
        "per_category": {
            c: {
                "pairs": sum(p.category == c for p in pairs),
                "same_answer": sum(p.category == c and p.same_answer for p in pairs),
            }
            for c in "ERV"
        },
        "deletion_options": deletion,
        "pairs": [asdict(p) for p in pairs],
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
