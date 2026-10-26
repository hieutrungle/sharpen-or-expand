"""verl custom reward hook, so RL rewards use exactly the evaluation verifier.

Wire it with `reward.custom_reward_function.path=soe/workers/reward_manager/math_reward.py` and
`reward.custom_reward_function.name=compute_score` (verl 0.9.1; the top-level
`custom_reward_function` key is legacy). The signature matches verl's naive reward manager.
"""

from typing import Any

from soe.utils.reward_score.math_verify import score_response


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None
) -> float:
    return 1.0 if score_response(solution_str, ground_truth).correct else 0.0
