from soe.workers.reward_manager.math_reward import compute_score


def test_verl_reward_hook_uses_same_verifier() -> None:
    assert compute_score("math", r"\boxed{\frac{2}{4}}", "1/2") == 1.0
    assert compute_score("math", r"\boxed{3}", "1/2") == 0.0
