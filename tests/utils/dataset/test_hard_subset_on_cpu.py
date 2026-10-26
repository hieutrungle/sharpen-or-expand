import pytest

from soe.utils.dataset.hard_subset import (
    HardSubsetRule,
    proportional_quotas,
    select_hard_subset,
    stratified_draw,
)

RULE = HardSubsetRule(max_correct=4, fallback_max_correct=8, min_size=4, target_size=6, draw_seed=7)


def problems(counts: dict[str, list[int]]) -> tuple[dict[str, int], dict[str, str]]:
    """{set: [correct count per problem]} -> (correct, stratum) keyed by "set/i"."""
    correct, stratum = {}, {}
    for name, values in counts.items():
        for i, c in enumerate(values):
            correct[f"{name}/{i}"] = c
            stratum[f"{name}/{i}"] = name
    return correct, stratum


def test_quotas_are_proportional_and_sum_to_total() -> None:
    assert proportional_quotas({"a": 50, "b": 30, "c": 20}, 10) == {"a": 5, "b": 3, "c": 2}
    quotas = proportional_quotas({"a": 1, "b": 1, "c": 1}, 2)
    assert sum(quotas.values()) == 2 and quotas == {"a": 1, "b": 1, "c": 0}  # tie -> name order


def test_quotas_reject_drawing_more_than_exists() -> None:
    with pytest.raises(ValueError):
        proportional_quotas({"a": 2}, 3)


def test_draw_is_deterministic_and_respects_strata() -> None:
    strata = {"a": [f"a{i}" for i in range(20)], "b": [f"b{i}" for i in range(10)]}
    first = stratified_draw(strata, 6, seed=1)
    assert first == stratified_draw(strata, 6, seed=1) and first == sorted(first)
    assert sum(pid.startswith("a") for pid in first) == 4  # 6 * 20/30
    assert stratified_draw(strata, 6, seed=2) != first


def test_all_eligible_kept_when_at_most_target() -> None:
    correct, stratum = problems({"x": [0, 1, 4, 5, 16], "y": [2, 3, 9]})
    subset = select_hard_subset(correct, stratum, RULE)
    assert subset.problem_ids == ["x/0", "x/1", "x/2", "y/0", "y/1"]  # 0/16 counts as hard
    assert (subset.max_correct_used, subset.eligible) == (4, 5)


def test_stratified_draw_when_more_than_target() -> None:
    correct, stratum = problems({"x": [0] * 8, "y": [1] * 4})
    subset = select_hard_subset(correct, stratum, RULE)
    assert len(subset.problem_ids) == 6 and subset.eligible == 12
    assert sum(pid.startswith("x/") for pid in subset.problem_ids) == 4


def test_fallback_threshold_applies_when_too_few_are_hard() -> None:
    correct, stratum = problems({"x": [3, 5, 6, 8, 12]})
    subset = select_hard_subset(correct, stratum, RULE)
    assert subset.max_correct_used == 8 and subset.problem_ids == ["x/0", "x/1", "x/2", "x/3"]


def test_rule_stops_when_even_the_fallback_is_too_small() -> None:
    correct, stratum = problems({"x": [0, 9, 10, 16]})
    with pytest.raises(ValueError, match="decide anew"):
        select_hard_subset(correct, stratum, RULE)


def test_rule_rejects_inconsistent_parameters() -> None:
    with pytest.raises(ValueError):
        HardSubsetRule(
            max_correct=8, fallback_max_correct=4, min_size=1, target_size=2, draw_seed=0
        )
