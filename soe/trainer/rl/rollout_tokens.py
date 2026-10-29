"""Rollout-token accounting (decision 018): generated tokens of training rollouts.

The soe agent loop appends one line per training rollout, `{"step": N, "tokens": T}`, to the file
named by SOE_ROLLOUT_TOKENS (runs/<id>/rollout_tokens/segment-<job>.jsonl, set by main_rl). A step
replayed after a resume appears in two segments; only the latest segment's records count, so every
trained step is counted once. The checkpoint callback sums them at each save.

No verl or torch import, so this runs in the CPU tests.
"""

import json
import os
from pathlib import Path

from soe.trainer.rl.segments import records_by_step

TOKENS_ENV = "SOE_ROLLOUT_TOKENS"


def record_rollout(step: int, tokens: int) -> None:
    path = os.environ.get(TOKENS_ENV)
    if path is None:
        return  # not inside an RL launch (unit tests, evaluation)
    # One short append per rollout: concurrent writers from several Ray workers do not interleave.
    with open(path, "a") as f:
        f.write(json.dumps({"step": step, "tokens": tokens}) + "\n")


def tokens_by_step(token_dir: Path) -> dict[int, list[int]]:
    """Per step, the rollout lengths recorded by the latest segment that ran it."""
    return {
        step: [int(entry["tokens"]) for entry in entries]
        for step, entries in records_by_step(token_dir).items()
    }


def cumulative_rollout_tokens(token_dir: Path, through_step: int, responses_per_step: int) -> int:
    """Generated tokens of steps 1..through_step. Raises if a step is missing or does not hold
    exactly responses_per_step rollouts: a lost rollout must not pass silently (decision 016)."""
    steps = tokens_by_step(token_dir)
    total = 0
    for step in range(1, through_step + 1):
        lengths = steps.get(step)
        if lengths is None:
            raise ValueError(f"no rollout-token records for step {step} in {token_dir}")
        if len(lengths) != responses_per_step:
            raise ValueError(
                f"step {step} has {len(lengths)} rollouts recorded, expected {responses_per_step}"
            )
        total += sum(lengths)
    return total
