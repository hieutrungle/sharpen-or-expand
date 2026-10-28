"""verl custom reward hook, so RL rewards use exactly the evaluation verifier.

Wire it with `reward.custom_reward_function.path=soe/workers/reward_manager/math_reward.py` and
`reward.custom_reward_function.name=compute_score` (verl 0.9.1). The signature matches verl's naive
reward manager; `extra_info` carries the agent loop's extra fields, among them `truncated`
(soe/workers/rollout/agent_loop.py).
"""

from typing import Any

from soe.utils.reward_score.math_verify import score_response


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None
) -> dict[str, float]:
    """Reward 1 for a verified answer, 0 otherwise. A rollout that hit the length cap gets 0
    whatever its last box says (decision 014), as evaluation scores it incorrect.

    verl logs every returned key next to the reward (`score` is the reward itself).
    """
    if not extra_info or "truncated" not in extra_info:
        # Without the flag a truncated answer would be rewarded; that must never pass silently.
        raise KeyError("extra_info['truncated'] is missing: is the soe agent loop configured?")
    truncated = bool(extra_info["truncated"])
    verdict = score_response(solution_str, ground_truth)
    correct = verdict.correct and not truncated
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "truncated": 1.0 if truncated else 0.0,
        "has_box": 0.0 if verdict.extracted is None else 1.0,
    }
