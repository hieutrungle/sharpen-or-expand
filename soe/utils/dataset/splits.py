"""RL pool cleaning and the Level B held-out split (decision 003)."""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from soe.utils.dataset.decontaminate import (
    NearDuplicateIndex,
    Ngram,
    common_ngrams,
    covered_by_union,
    drop_exact_duplicates,
)
from soe.utils.dataset.problems import Level, Problem


@dataclass(frozen=True, slots=True, kw_only=True)
class RlSplits:
    train: list[Problem]
    level_b: list[Problem]
    ignored_ngrams: frozenset[Ngram]
    dropped: dict[str, list[str]] = field(default_factory=dict)  # reason -> problem ids


def build_rl_splits(
    pool: Sequence[Problem],
    eval_problems: Sequence[Problem],
    *,
    level_b_size: int,
    seed: int,
    ngram: int,
    min_fraction: float,
    boilerplate_max_count: int,
) -> RlSplits:
    """Clean the pool, carve Level B at random, and keep train free of Level B near-duplicates.

    Order: (1) exact duplicates; (2) near-duplicates of any evaluation problem; (3) a seeded random
    Level B sample; (4) train problems that near-duplicate a Level B problem; (5) Level B problems
    that the remaining train pool still covers as a whole (reworded copies), which leave Level B.
    N-grams shared by more than `boilerplate_max_count` pool problems are ignored throughout.
    """
    unique, exact = drop_exact_duplicates(pool)
    ignore = common_ngrams([p.question for p in unique], ngram, boilerplate_max_count)
    clean, eval_dups = _split_off(unique, eval_problems, ngram, min_fraction, ignore)
    if len(clean) <= level_b_size:
        raise ValueError(f"pool has {len(clean)} clean problems, need more than {level_b_size}")

    chosen = set(random.Random(seed).sample(range(len(clean)), level_b_size))
    sampled_b = [p for i, p in enumerate(clean) if i in chosen]
    rest = [p for i, p in enumerate(clean) if i not in chosen]
    train, b_dups = _split_off(rest, sampled_b, ngram, min_fraction, ignore)
    leaked = set(
        covered_by_union(sampled_b, train, n=ngram, min_fraction=min_fraction, ignore=ignore)
    )
    return RlSplits(
        train=[p.model_copy(update={"level": Level.TRAIN}) for p in train],
        level_b=[
            p.model_copy(update={"level": Level.B}) for p in sampled_b if p.problem_id not in leaked
        ],
        ignored_ngrams=ignore,
        dropped={
            "exact_duplicate": [p.problem_id for p in exact],
            "near_duplicate_of_eval": eval_dups,
            "near_duplicate_of_level_b": b_dups,
            "level_b_covered_by_train": sorted(leaked),
        },
    )


def _split_off(
    candidates: Sequence[Problem],
    references: Sequence[Problem],
    ngram: int,
    min_fraction: float,
    ignore: frozenset[Ngram],
) -> tuple[list[Problem], list[str]]:
    """(candidates that near-duplicate no reference, ids of those that do)."""
    index = NearDuplicateIndex(references, n=ngram, min_fraction=min_fraction, ignore=ignore)
    kept: list[Problem] = []
    dropped: list[str] = []
    for problem in candidates:
        if index.matches(problem.question):
            dropped.append(problem.problem_id)
        else:
            kept.append(problem)
    return kept, dropped
