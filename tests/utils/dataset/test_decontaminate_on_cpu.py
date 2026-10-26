import pytest

from soe.utils.dataset.decontaminate import (
    NearDuplicateIndex,
    common_ngrams,
    covered_by_union,
    drop_exact_duplicates,
    normalize,
    word_ngrams,
)
from soe.utils.dataset.problems import Level, Problem


def problem(pid: str, question: str) -> Problem:
    return Problem(problem_id=pid, source="t", level=Level.A, question=question, answer="1")


WORDS = " ".join(f"w{i}" for i in range(20))  # 20 distinct words -> 8 13-grams


def test_normalize_ignores_case_punctuation_and_latex_markup() -> None:
    assert normalize(r"Find $\frac{1}{2}$,  NOW!") == "find frac 1 2 now"


def test_word_ngrams_short_text_is_one_gram() -> None:
    assert word_ngrams("a b c", 13) == frozenset({("a", "b", "c")})
    assert word_ngrams("", 13) == frozenset()
    assert len(word_ngrams(WORDS, 13)) == 8


def test_drop_exact_duplicates_keeps_first_after_normalisation() -> None:
    kept, dropped = drop_exact_duplicates(
        [problem("a", "Find x."), problem("b", "find  X"), problem("c", "Find y.")]
    )
    assert [p.problem_id for p in kept] == ["a", "c"]
    assert [p.problem_id for p in dropped] == ["b"]


def test_near_duplicate_threshold_is_over_the_reference() -> None:
    index = NearDuplicateIndex([problem("ref", WORDS)], n=13, min_fraction=0.5)
    words = WORDS.split()
    # first 16 words keep 4 of the reference's 8 13-grams: exactly 50% -> match
    assert index.matches(" ".join(words[:16])) == ["ref"]
    # first 15 words keep 3 of 8 -> no match
    assert index.matches(" ".join(words[:15])) == []
    # a long candidate that quotes the whole reference is still a match
    assert index.matches("preamble " * 50 + WORDS) == ["ref"]


def test_min_fraction_must_be_a_fraction() -> None:
    with pytest.raises(ValueError):
        NearDuplicateIndex([], n=13, min_fraction=0)


def test_common_ngrams_counts_each_text_once() -> None:
    texts = [WORDS, WORDS + " " + WORDS, "other words entirely"]
    assert common_ngrams(texts, 13, 1) == word_ngrams(WORDS, 13)
    assert common_ngrams(texts, 13, 2) == frozenset()


def test_ignored_ngrams_cannot_create_a_match() -> None:
    reference = problem("ref", WORDS)
    index = NearDuplicateIndex([reference], n=13, min_fraction=0.5, ignore=word_ngrams(WORDS, 13))
    assert index.matches(WORDS) == []  # nothing distinctive left in the reference


def test_covered_by_union_sees_copies_split_across_problems() -> None:
    words = WORDS.split()
    halves = [problem("h1", " ".join(words[:14])), problem("h2", " ".join(words[6:]))]
    single = NearDuplicateIndex([problem("ref", WORDS)], n=13, min_fraction=0.5)
    assert all(single.matches(h.question) == [] for h in halves)  # 2/8 and 2/8 n-grams alone
    assert covered_by_union([problem("ref", WORDS)], halves, n=13, min_fraction=0.5) == ["ref"]
