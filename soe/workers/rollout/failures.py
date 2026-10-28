"""A failed rollout must stop an RL run.

verl 0.9.1's sync trainer only logs an exception raised while a rollout is generated or scored,
then trains on the group's remaining trajectories: the failed ones vanish from the batch without a
metric (M1.3 smoke attempt 2 lost every response with a \\boxed answer this way). The soe agent loop
and reward hook append each failure to the file named by SOE_ROLLOUT_FAILURES; soe.trainer.main_rl
sets it, watches the file and stops the trainer at the first entry.

No verl import, so this runs in the CPU tests.
"""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FAILURES_ENV = "SOE_ROLLOUT_FAILURES"


def record_failure(where: str, error: Exception) -> None:
    path = os.environ.get(FAILURES_ENV)
    if path is None:
        return  # not inside an RL launch (unit tests, evaluation)
    message = f"{type(error).__name__}: {error}"[:2000]
    entry = {"where": where, "pid": os.getpid(), "error": message}
    # One short append per failure: concurrent writers from several Ray workers do not interleave.
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


@contextmanager
def failures_recorded(where: str) -> Iterator[None]:
    """Record any exception from the block, then re-raise it. Cancellation is not a failure."""
    try:
        yield
    except Exception as error:
        record_failure(where, error)
        raise


def read_failures(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
