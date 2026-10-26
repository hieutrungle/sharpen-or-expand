import pytest

from soe.utils.metric.transitions import Transition, classify, count_transitions, is_solved


def test_classify_all_four_cases() -> None:
    before = {"a": False, "b": True, "c": True, "d": False}
    after = {"a": True, "b": False, "c": True, "d": False}
    assert classify(before, after) == {
        "a": Transition.GAINED,
        "b": Transition.LOST,
        "c": Transition.KEPT,
        "d": Transition.NEVER,
    }
    assert count_transitions(before, after) == {
        "unsolved_to_solved": 1,
        "solved_to_lost": 1,
        "solved_to_solved": 1,
        "unsolved_to_unsolved": 1,
    }


def test_classify_requires_same_problems() -> None:
    with pytest.raises(ValueError):
        classify({"a": True}, {"b": True})


def test_is_solved_uses_pass_at_k_threshold() -> None:
    assert is_solved(n=16, c=1, k=16, threshold=0.5)
    assert not is_solved(n=16, c=1, k=1, threshold=0.5)
