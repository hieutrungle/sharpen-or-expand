"""Exact de-duplication and word n-gram near-duplicate detection (decisions 002 and 003).

A candidate text is a near-duplicate of a reference problem when at least `min_fraction` of the
reference's word n-grams occur in the candidate. The fraction is taken over the reference (the
evaluation problem) so a long training problem that quotes a short evaluation problem is caught.

Boilerplate n-grams (answer-format sentences such as DAPO's "The answer is in the form ...
Please provide the value of ...") are shared by thousands of unrelated problems; `common_ngrams`
finds them so the index can ignore them.
"""

import re
from collections import defaultdict
from collections.abc import Sequence

from soe.utils.dataset.problems import Problem

Ngram = tuple[str, ...]


def normalize(text: str) -> str:
    """Lowercase alphanumeric tokens separated by single spaces (LaTeX markup becomes tokens)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def word_ngrams(text: str, n: int) -> frozenset[Ngram]:
    """All word n-grams; a text shorter than n words counts as one n-gram (itself)."""
    tokens = normalize(text).split()
    if len(tokens) < n:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def common_ngrams(texts: Sequence[str], n: int, max_count: int) -> frozenset[Ngram]:
    """N-grams that occur in more than `max_count` of the texts (counted once per text)."""
    counts: dict[Ngram, int] = defaultdict(int)
    for text in texts:
        for gram in word_ngrams(text, n):
            counts[gram] += 1
    return frozenset(gram for gram, count in counts.items() if count > max_count)


def covered_by_union(
    references: Sequence[Problem],
    corpus: Sequence[Problem],
    *,
    n: int,
    min_fraction: float,
    ignore: frozenset[Ngram] = frozenset(),
) -> list[str]:
    """Ids of references whose n-grams the corpus covers as a whole (catches reworded copies
    spread over several corpus problems, which no single-problem match sees)."""
    corpus_grams: set[Ngram] = set()
    for problem in corpus:
        corpus_grams |= word_ngrams(problem.question, n) - ignore
    covered = []
    for reference in references:
        grams = word_ngrams(reference.question, n) - ignore
        if grams and len(grams & corpus_grams) / len(grams) >= min_fraction:
            covered.append(reference.problem_id)
    return covered


def drop_exact_duplicates(problems: Sequence[Problem]) -> tuple[list[Problem], list[Problem]]:
    """Keep the first problem of each normalised question; return (kept, dropped)."""
    seen: set[str] = set()
    kept: list[Problem] = []
    dropped: list[Problem] = []
    for problem in problems:
        key = normalize(problem.question)
        (dropped if key in seen else kept).append(problem)
        seen.add(key)
    return kept, dropped


class NearDuplicateIndex:
    """Finds the reference problems that a candidate text near-duplicates."""

    def __init__(
        self,
        references: Sequence[Problem],
        *,
        n: int,
        min_fraction: float,
        ignore: frozenset[Ngram] = frozenset(),
    ) -> None:
        if not 0 < min_fraction <= 1:
            raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")
        self._n = n
        self._min_fraction = min_fraction
        self._ignore = ignore
        self._size: dict[str, int] = {}
        self._owners: dict[Ngram, list[str]] = defaultdict(list)
        for reference in references:
            grams = word_ngrams(reference.question, n) - ignore
            if not grams:  # nothing distinctive left: cannot be matched
                continue
            self._size[reference.problem_id] = len(grams)
            for gram in grams:
                self._owners[gram].append(reference.problem_id)

    def matches(self, text: str) -> list[str]:
        """Ids of references with at least min_fraction of their n-grams present in `text`."""
        hits: dict[str, int] = defaultdict(int)
        for gram in word_ngrams(text, self._n) - self._ignore:
            for owner in self._owners.get(gram, ()):
                hits[owner] += 1
        return sorted(
            owner
            for owner, count in hits.items()
            if count / self._size[owner] >= self._min_fraction
        )
