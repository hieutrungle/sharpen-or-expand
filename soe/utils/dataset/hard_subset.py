"""The M0.6 hard-subset rule (decision 005), as pure functions of per-problem correct counts.

The counts come from a selection run with its own sampling seed, so a model's evaluated pass rate on
the subset is not biased low by the same draws that selected it (no regression to the mean).
"""

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class HardSubsetRule:
    max_correct: int  # eligible: at most this many of the selection samples correct
    fallback_max_correct: int  # used once if fewer than min_size are eligible
    min_size: int
    target_size: int
    draw_seed: int

    def __post_init__(self) -> None:
        if not 0 <= self.max_correct < self.fallback_max_correct:
            raise ValueError("need 0 <= max_correct < fallback_max_correct")
        if not 0 < self.min_size <= self.target_size:
            raise ValueError("need 0 < min_size <= target_size")


@dataclass(frozen=True, slots=True, kw_only=True)
class HardSubset:
    problem_ids: list[str]  # sorted
    max_correct_used: int
    eligible: int


def proportional_quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` across strata in proportion to their sizes (largest-remainder method).

    Ties in the remainder go to the stratum whose name sorts first, so the split is deterministic.
    """
    population = sum(sizes.values())
    if not 0 <= total <= population:
        raise ValueError(f"cannot draw {total} from {population}")
    exact = {name: total * size / population for name, size in sizes.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    by_remainder = sorted(sizes, key=lambda name: (quotas[name] - exact[name], name))
    for name in by_remainder[: total - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def stratified_draw(ids_by_stratum: dict[str, list[str]], size: int, seed: int) -> list[str]:
    quotas = proportional_quotas({k: len(v) for k, v in ids_by_stratum.items()}, size)
    rng = random.Random(seed)
    drawn: list[str] = []
    for name in sorted(ids_by_stratum):  # fixed order, so the draw depends only on the seed
        drawn += rng.sample(sorted(ids_by_stratum[name]), quotas[name])
    return sorted(drawn)


def select_hard_subset(
    correct: dict[str, int], stratum: dict[str, str], rule: HardSubsetRule
) -> HardSubset:
    """Problems with few correct selection samples; a stratified draw if more than target_size.

    `correct` maps problem_id to its number of correct selection samples, `stratum` to its
    problem set (the strata of the draw).
    """
    for max_correct in (rule.max_correct, rule.fallback_max_correct):
        eligible = sorted(pid for pid, c in correct.items() if c <= max_correct)
        if len(eligible) >= rule.min_size:
            break
    else:
        raise ValueError(
            f"only {len(eligible)} problems with <= {rule.fallback_max_correct} correct; "
            f"the rule needs {rule.min_size} (decision 005: stop and decide anew)"
        )
    if len(eligible) <= rule.target_size:
        chosen = eligible
    else:
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for pid in eligible:
            by_stratum[stratum[pid]].append(pid)
        chosen = stratified_draw(dict(by_stratum), rule.target_size, rule.draw_seed)
    return HardSubset(problem_ids=chosen, max_correct_used=max_correct, eligible=len(eligible))
