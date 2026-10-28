import pytest

from soe.workers.reward_manager.math_reward import compute_score

STOPPED = {"truncated": False}


def test_verl_reward_hook_uses_same_verifier() -> None:
    assert compute_score("soe/dapo", r"\boxed{\frac{2}{4}}", "1/2", STOPPED)["score"] == 1.0
    assert compute_score("soe/dapo", r"\boxed{3}", "1/2", STOPPED)["score"] == 0.0


def test_truncated_rollout_gets_zero_even_with_the_right_box() -> None:
    # Decision 014: evaluation scores a truncated response incorrect, so RL must not reward it.
    result = compute_score("soe/dapo", r"\boxed{7}", "7", {"truncated": True})
    assert result == {"score": 0.0, "acc": 0.0, "truncated": 1.0, "has_box": 1.0}


def test_missing_truncation_flag_fails_loudly() -> None:
    with pytest.raises(KeyError, match="truncated"):
        compute_score("soe/dapo", r"\boxed{7}", "7", {})
    with pytest.raises(KeyError, match="truncated"):
        compute_score("soe/dapo", r"\boxed{7}", "7", None)


def test_answer_is_read_after_the_last_think_close_and_split_boxes_join() -> None:
    thinking = r"<think>maybe \boxed{5}</think> The answer is \boxed{7}."
    assert compute_score("soe/dapo", thinking, "7", STOPPED)["score"] == 1.0
    no_box = compute_score("soe/dapo", "<think>no idea</think> I give up.", "7", STOPPED)
    assert no_box["has_box"] == 0.0 and no_box["score"] == 0.0
