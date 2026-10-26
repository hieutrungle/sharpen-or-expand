import pytest

from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.splits import build_rl_splits


def problem(pid: str, question: str, level: Level = Level.TRAIN) -> Problem:
    return Problem(problem_id=pid, source="t", level=level, question=question, answer="1")


def unique_question(i: int) -> str:
    return " ".join(f"q{i}w{j}" for j in range(20))


POOL = [problem(f"p{i}", unique_question(i)) for i in range(50)]
EVAL = [problem("e0", unique_question(7), Level.A)]  # p7 duplicates an eval problem


def test_sizes_levels_and_disjointness() -> None:
    result = build_rl_splits(
        POOL, EVAL, level_b_size=10, seed=1, ngram=13, min_fraction=0.5, boilerplate_max_count=1000
    )
    assert len(result.level_b) == 10 and len(result.train) == 39
    assert {p.level for p in result.level_b} == {Level.B}
    assert {p.level for p in result.train} == {Level.TRAIN}
    assert not {p.problem_id for p in result.train} & {p.problem_id for p in result.level_b}
    assert result.dropped["near_duplicate_of_eval"] == ["p7"]


def test_same_seed_same_split_different_seed_different_split() -> None:
    def ids(seed: int) -> list[str]:
        result = build_rl_splits(
            POOL,
            EVAL,
            level_b_size=10,
            seed=seed,
            ngram=13,
            min_fraction=0.5,
            boilerplate_max_count=1000,
        )
        return [p.problem_id for p in result.level_b]

    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


def test_exact_duplicates_are_dropped_after_normalisation() -> None:
    pool = [*POOL, problem("dup", unique_question(3).upper())]
    result = build_rl_splits(
        pool, EVAL, level_b_size=10, seed=1, ngram=13, min_fraction=0.5, boilerplate_max_count=1000
    )
    assert result.dropped["exact_duplicate"] == ["dup"]


def test_train_never_near_duplicates_level_b() -> None:
    copies = [problem(f"c{i}", unique_question(i) + " extra words") for i in range(50)]
    result = build_rl_splits(
        POOL + copies,
        EVAL,
        level_b_size=10,
        seed=3,
        ngram=13,
        min_fraction=0.5,
        boilerplate_max_count=1000,
    )
    b_ids = {p.problem_id for p in result.level_b}
    leaked = [
        p for p in result.train if p.problem_id.startswith("c") and f"p{p.problem_id[1:]}" in b_ids
    ]
    assert leaked == []
    assert result.dropped["near_duplicate_of_level_b"]


def test_too_small_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="clean problems"):
        build_rl_splits(
            POOL[:5],
            EVAL,
            level_b_size=10,
            seed=1,
            ngram=13,
            min_fraction=0.5,
            boilerplate_max_count=1000,
        )


BOILERPLATE = " The answer is in the form m over n where gcd m n 1 please provide the value of m n"


def test_shared_answer_format_boilerplate_does_not_make_problems_duplicates() -> None:
    # Short distinct problems that all end with the same long answer-format sentence.
    pool = [problem(f"s{i}", f"compute item {i} of the list." + BOILERPLATE) for i in range(60)]
    kept = build_rl_splits(
        pool, [], level_b_size=10, seed=1, ngram=13, min_fraction=0.5, boilerplate_max_count=20
    )
    assert kept.dropped["near_duplicate_of_level_b"] == []
    assert len(kept.train) == 50 and kept.ignored_ngrams
    naive = build_rl_splits(
        pool, [], level_b_size=10, seed=1, ngram=13, min_fraction=0.5, boilerplate_max_count=1000
    )
    assert (
        len(naive.dropped["near_duplicate_of_level_b"]) == 50
    )  # every train problem looks like a dup


def test_level_b_problem_spread_over_reworded_train_copies_leaves_level_b() -> None:
    # target has 18 13-grams; each reworded copy keeps 8 of them (44%, below the threshold),
    # together they cover 16 of 18 (89%).
    words = [f"x{j}" for j in range(30)]
    first_half = " ".join(words[:20])
    second_half = " ".join(words[10:])
    target = problem("target", " ".join(words))
    pool = [target, problem("h1", first_half), problem("h2", second_half), *POOL]
    for seed in range(200):  # a seed whose Level B holds "target" but neither half
        result = build_rl_splits(
            pool,
            [],
            level_b_size=10,
            seed=seed,
            ngram=13,
            min_fraction=0.5,
            boilerplate_max_count=1000,
        )
        if "target" in result.dropped["level_b_covered_by_train"]:
            break
    assert result.dropped["level_b_covered_by_train"] == ["target"]
    assert "target" not in {p.problem_id for p in result.level_b}
    assert len(result.level_b) == 9
