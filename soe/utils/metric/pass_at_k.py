"""Capability metrics. pass@k uses the unbiased estimator of Chen et al. (2021)."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from math import prod

from soe.utils.dataset.records import SampleRecord


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples, drawn without replacement from n, is correct.

    Equals 1 - C(n-c, k) / C(n, k), computed as a product to avoid huge binomials.
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def mean_pass_at_k(counts: Iterable[tuple[int, int]], k: int) -> float:
    """Mean pass@k over problems, given (n, c) per problem."""
    values = [pass_at_k(n, c, k) for n, c in counts]
    if not values:
        raise ValueError("no problems")
    return sum(values) / len(values)


def same_string(a: str, b: str) -> bool:
    """Exact match ignoring whitespace (the "exact_string" maj@k grouping)."""
    return "".join(a.split()) == "".join(b.split())


def maj_at_k(
    answers: Sequence[str | None],
    correct: Sequence[bool],
    k: int,
    same: Callable[[str, str], bool],
) -> bool:
    """Is the most common answer among the first k samples correct?

    Answers form groups: each answer joins the first group whose representative (the group's
    first answer) it matches under `same`, e.g. verifier equivalence so "0.5" and "\\frac12" are one
    vote. Ties go to the group seen first. Samples with no extracted answer do not vote; if none
    vote, the result is False.
    """
    if len(answers) != len(correct):
        raise ValueError("answers and correct must have the same length")
    if not 1 <= k <= len(answers):
        raise ValueError(f"need 1 <= k <= {len(answers)}, got {k}")
    groups: list[list[int]] = []  # sample indices; groups[g][0] is the representative
    for i in range(k):
        answer = answers[i]
        if answer is None:
            continue
        for group in groups:
            representative = answers[group[0]]
            if representative is not None and same(answer, representative):
                group.append(i)
                break
        else:
            groups.append([i])
    if not groups:
        return False
    winner = max(groups, key=len)  # max keeps the first of equally large groups
    return correct[winner[0]]


def correct_counts(records: Iterable[SampleRecord]) -> dict[str, tuple[int, int]]:
    """(n, c) per problem_id."""
    n: Counter[str] = Counter()
    c: Counter[str] = Counter()
    for record in records:
        n[record.problem_id] += 1
        c[record.problem_id] += record.correct
    return {problem_id: (n[problem_id], c[problem_id]) for problem_id in n}
