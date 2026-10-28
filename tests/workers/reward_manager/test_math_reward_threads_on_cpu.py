import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from soe.workers.reward_manager.math_reward import compute_score
from soe.workers.rollout.failures import FAILURES_ENV

STOPPED = {"truncated": False}


def test_scores_from_executor_threads_as_verl_calls_it() -> None:
    # M1.3 smoke attempt 2: math-verify's signal.alarm raised on every boxed answer off the main
    # thread, and verl silently dropped those rollouts.
    cases = [(r"\boxed{\frac{2}{4}}", "1/2", 1.0), (r"\boxed{3}", "1/2", 0.0)] * 8
    with ThreadPoolExecutor(8) as threads:
        scores = list(
            threads.map(lambda c: compute_score("soe/dapo", c[0], c[1], STOPPED)["score"], cases)
        )
    assert scores == [expected for _, _, expected in cases]


def test_reward_error_is_recorded_for_the_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failures = tmp_path / "rollout_failures.jsonl"
    monkeypatch.setenv(FAILURES_ENV, str(failures))
    with pytest.raises(KeyError):
        compute_score("soe/dapo", r"\boxed{7}", "7", {})
    [entry] = [json.loads(line) for line in failures.read_text().splitlines()]
    assert entry["where"] == "reward" and "truncated" in str(entry["error"])
