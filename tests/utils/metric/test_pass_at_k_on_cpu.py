from itertools import combinations
from math import comb

import pytest

from soe.utils.dataset.records import SampleRecord
from soe.utils.metric.pass_at_k import (
    correct_counts,
    maj_at_k,
    mean_pass_at_k,
    pass_at_k,
    same_string,
)
from soe.utils.reward_score.math_verify import answers_equivalent


def brute_force_pass_at_k(n: int, c: int, k: int) -> float:
    """Fraction of all k-subsets of n samples (c correct) that contain a correct sample."""
    samples = [True] * c + [False] * (n - c)
    hits = sum(any(subset) for subset in combinations(samples, k))
    return hits / comb(n, k)


@pytest.mark.parametrize("n", [1, 2, 5, 8, 10])
def test_pass_at_k_matches_enumeration(n: int) -> None:
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_at_k(n, c, k) == pytest.approx(brute_force_pass_at_k(n, c, k), abs=1e-12)


def test_pass_at_1_is_fraction_correct() -> None:
    assert pass_at_k(64, 16, 1) == pytest.approx(0.25)


def test_pass_at_k_large_n_is_stable() -> None:
    value = pass_at_k(1024, 1, 64)
    assert value == pytest.approx(64 / 1024)


@pytest.mark.parametrize(("n", "c", "k"), [(4, 5, 1), (4, -1, 1), (4, 1, 0), (4, 1, 5)])
def test_pass_at_k_rejects_invalid(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_mean_pass_at_k() -> None:
    assert mean_pass_at_k([(4, 4), (4, 0)], k=2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mean_pass_at_k([], k=1)


def test_maj_at_k_exact_string_grouping_ignores_whitespace() -> None:
    answers = ["1/2", "3", "1 / 2", None]
    correct = [True, False, True, False]
    assert maj_at_k(answers, correct, k=4, same=same_string) is True


def test_maj_at_k_verifier_grouping_merges_equivalent_forms() -> None:
    # Under the verifier "0.5", "\frac{1}{2}" and "1/2" are one group of 3, beating "3" (2 votes).
    answers = ["3", "0.5", "\\frac{1}{2}", "3", "1/2"]
    correct = [False, True, True, False, True]
    assert maj_at_k(answers, correct, k=5, same=same_string) is False  # "3" wins 2-1-1-1
    assert maj_at_k(answers, correct, k=5, same=answers_equivalent) is True  # 1/2 wins 3-2


def test_maj_at_k_tie_goes_to_first_seen() -> None:
    assert maj_at_k(["3", "5"], [False, True], k=2, same=same_string) is False
    assert maj_at_k(["5", "3"], [True, False], k=2, same=same_string) is True


def test_maj_at_k_uses_only_first_k_and_no_votes_is_wrong() -> None:
    assert maj_at_k(["3", "5", "5"], [False, True, True], k=1, same=same_string) is False
    assert maj_at_k([None, None], [False, False], k=2, same=same_string) is False


def test_correct_counts() -> None:
    def rec(pid: str, i: int, ok: bool) -> SampleRecord:
        return SampleRecord(
            problem_id=pid,
            sample_index=i,
            checkpoint="base",
            completion="",
            extracted_answer=None,
            correct=ok,
            completion_tokens=0,
            finish_reason="stop",
        )

    records = [rec("a", 0, True), rec("a", 1, False), rec("b", 0, False)]
    assert correct_counts(records) == {"a": (2, 1), "b": (1, 0)}
