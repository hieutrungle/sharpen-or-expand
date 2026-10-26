"""Capability transitions between two checkpoints: what RL gained, kept and lost per problem."""

from collections import Counter
from collections.abc import Mapping
from enum import StrEnum

from soe.utils.metric.pass_at_k import pass_at_k


class Transition(StrEnum):
    GAINED = "unsolved_to_solved"
    LOST = "solved_to_lost"
    KEPT = "solved_to_solved"
    NEVER = "unsolved_to_unsolved"


def is_solved(n: int, c: int, k: int, threshold: float) -> bool:
    """A problem counts as solved at k when its pass@k estimate reaches the frozen threshold."""
    return pass_at_k(n, c, k) >= threshold


def classify(before: Mapping[str, bool], after: Mapping[str, bool]) -> dict[str, Transition]:
    if before.keys() != after.keys():
        raise ValueError("both checkpoints must be evaluated on the same problems")
    table = {
        (False, True): Transition.GAINED,
        (True, False): Transition.LOST,
        (True, True): Transition.KEPT,
        (False, False): Transition.NEVER,
    }
    return {pid: table[(before[pid], after[pid])] for pid in before}


def count_transitions(before: Mapping[str, bool], after: Mapping[str, bool]) -> dict[str, int]:
    counts = Counter(classify(before, after).values())
    return {t.value: counts[t] for t in Transition}
