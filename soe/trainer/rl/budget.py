"""When an RL run stops (decision 020): at its rollout-token budget, its wall-clock cap
(development) or its step limit (development), whichever comes first.

verl stops only at `trainer.total_training_steps`, and soe sees the token count only at a save. At
each save the checkpoint callback therefore re-estimates the step at which the budget will be
reached, from the tokens and time per step since the previous save, and sets verl's total to it.
verl then saves and validates that step as its last. The estimate is written to the run directory,
so a resumed job starts with it.

No verl or torch import, so this runs in the CPU tests.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

STOP_FILE = "stop_step.json"


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetState:
    step: int  # the step just saved
    rollout_tokens: int  # cumulative through `step`
    tokens_per_step: float  # recent mean
    hours_used: float  # training wall-clock over all job segments, through `step`
    hours_per_step: float  # recent mean


def last_step(
    state: BudgetState,
    *,
    budget_tokens: int,
    max_hours: float | None = None,
    max_steps: int | None = None,
) -> int:
    """The step to stop after: the first whose cumulative tokens reach the budget, or the last that
    fits in max_hours, or max_steps; never before the step just saved."""
    remaining = budget_tokens - state.rollout_tokens
    candidates = [state.step + max(0, math.ceil(remaining / state.tokens_per_step))]
    if max_hours is not None:
        spare = max_hours - state.hours_used
        candidates.append(state.step + max(0, math.floor(spare / state.hours_per_step)))
    if max_steps is not None:
        candidates.append(max_steps)
    return max(state.step, min(candidates))


def write_stop_step(run_dir: Path, stop_step: int, state: BudgetState) -> None:
    entry = {
        "stop_step": stop_step,
        "estimated_at_step": state.step,
        "rollout_tokens": state.rollout_tokens,
        "tokens_per_step": state.tokens_per_step,
        "hours_used": state.hours_used,
        "hours_per_step": state.hours_per_step,
    }
    tmp = run_dir / f".{STOP_FILE}.tmp"
    tmp.write_text(json.dumps(entry) + "\n")
    tmp.replace(run_dir / STOP_FILE)


def read_stop_step(run_dir: Path) -> int | None:
    path = run_dir / STOP_FILE
    return int(json.loads(path.read_text())["stop_step"]) if path.exists() else None
