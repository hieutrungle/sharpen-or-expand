"""verl custom reward hook, so RL rewards use exactly the evaluation verifier.

Wire it with `reward.custom_reward_function.path=soe/workers/reward_manager/math_reward.py` and
`reward.custom_reward_function.name=compute_score` (verl 0.9.1). The signature matches verl's naive
reward manager; `extra_info` carries the agent loop's extra fields, among them `truncated`
(soe/workers/rollout/agent_loop.py).

verl calls the hook from executor threads, but math-verify's 5 s time limits use signal.alarm,
which only works on a process's main thread. Each call therefore runs in a small process pool
whose workers score on their main thread, so RL keeps evaluation's time limits exactly. Any error
is recorded so the run stops (soe/workers/rollout/failures.py).
"""

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from soe.utils.reward_score.math_verify import score_response
from soe.workers.rollout.failures import failures_recorded

SCORING_PROCESSES = 2  # per verl reward worker (8 by default); one score takes milliseconds

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _scoring_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            # spawn, not fork: forking a threaded Ray worker can deadlock the child.
            context = multiprocessing.get_context("spawn")
            _pool = ProcessPoolExecutor(SCORING_PROCESSES, mp_context=context)
        return _pool


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None
) -> dict[str, float]:
    """Reward 1 for a verified answer, 0 otherwise. A rollout that hit the length cap gets 0
    whatever its last box says (decision 014), as evaluation scores it incorrect.

    verl logs every returned key next to the reward (`score` is the reward itself).
    """
    with failures_recorded("reward"):
        if not extra_info or "truncated" not in extra_info:
            # Without the flag a truncated answer would be rewarded; that must never pass silently.
            raise KeyError("extra_info['truncated'] is missing: is the soe agent loop configured?")
        truncated = bool(extra_info["truncated"])
        verdict = _scoring_pool().submit(score_response, solution_str, ground_truth).result()
    correct = verdict.correct and not truncated
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "truncated": 1.0 if truncated else 0.0,
        "has_box": 0.0 if verdict.extracted is None else 1.0,
        # Diagnostic only (M1.3: >= 99% of non-truncated rollouts end on <|im_end|>).
        "turn_end": 1.0 if extra_info.get("turn_end") else 0.0,
    }
