"""RL rewards equal evaluation verdicts on 200 scored rollouts (M1.7).

The fixtures (scripts/make_reward_fixtures.py) are validation-split rollouts of the M1.5 runs:
RL-train-pool problems, never evaluation text. For each one, three things must agree:
- the verl reward hook (`compute_score`);
- main_eval's verdict: the verifier's `correct` for a response that stopped, incorrect when it hit
  the length cap (soe/workers/rollout/vllm_sampler.py);
- the reward verl actually gave during the run.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

from soe.utils.reward_score.math_verify import score_response
from soe.workers.reward_manager.math_reward import compute_score

FIXTURES = Path(__file__).parent / "fixtures" / "reward_fixtures.jsonl.gz"


def fixtures() -> list[dict[str, object]]:
    lines = gzip.decompress(FIXTURES.read_bytes()).decode().splitlines()
    return [json.loads(line) for line in lines]


def eval_verdict(response: str, gold: str, truncated: bool) -> float:
    return 1.0 if score_response(response, gold).correct and not truncated else 0.0


def test_reward_equals_the_evaluation_verdict_and_the_training_reward() -> None:
    rows = fixtures()
    assert Counter(str(r["category"]) for r in rows) == {
        "correct": 80,
        "wrong_boxed": 60,
        "truncated_boxed": 30,
        "truncated_no_box": 30,
    }
    for row in rows:
        response, gold, truncated = str(row["response"]), str(row["gold"]), bool(row["truncated"])
        reward = compute_score("soe/fixture", response, gold, {"truncated": truncated})["score"]
        assert reward == eval_verdict(response, gold, truncated) == row["training_reward"], row[
            "source"
        ]


def test_fixtures_would_catch_a_rewarded_truncation() -> None:
    """At least one truncated rollout ends on a correct box: rewarding truncation would show."""
    truncated = [r for r in fixtures() if r["truncated"]]
    assert any(score_response(str(r["response"]), str(r["gold"])).correct for r in truncated)
