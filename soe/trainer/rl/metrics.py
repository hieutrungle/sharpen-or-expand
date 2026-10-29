"""soe per-step training metrics (M1.6): what verl does not log, from one record per rollout.

The reward hook (soe/workers/reward_manager/math_reward.py) appends one RolloutOutcome per
training rollout to the file named by SOE_ROLLOUT_OUTCOMES, set by main_rl to
`runs/<id>/rollouts/segment-<job>.jsonl`. Response lengths come from the rollout-token records
(decision 018). `step_metrics` turns one step's records into `soe/*` metrics: group reward
variance, zero-variance groups (all correct and all wrong separately), truncation, boxed and
stop-reason rates, length percentiles, and answer diversity (distinct verifier-equivalent final
answers per group, grouped like maj@k).

`python -m soe.trainer.main_metrics` recomputes every step after each save and writes
`runs/<id>/rollout_metrics.jsonl` (same `{"step", "data"}` lines as verl's file logger).

No verl or torch import, so this runs in the CPU tests.
"""

import json
import os
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from soe.trainer.rl.segments import records_by_step

OUTCOMES_ENV = "SOE_ROLLOUT_OUTCOMES"
OUTCOMES_DIR = "rollouts"
STEP_METRICS = "rollout_metrics.jsonl"

# Every key step_metrics returns; the run checker requires all of them for every step.
SOE_KEYS = (
    "soe/reward/mean",
    "soe/group/reward_var_mean",
    "soe/group/zero_var_frac",
    "soe/group/all_correct_frac",
    "soe/group/all_wrong_frac",
    "soe/rollout/truncated_frac",
    "soe/rollout/boxed_frac",
    "soe/rollout/stop_turn_end_frac",
    "soe/rollout/stop_other_frac",
    "soe/length/mean",
    "soe/length/p50",
    "soe/length/p95",
    "soe/length/max",
    "soe/answers/distinct_per_group_mean",
    "soe/answers/none_frac",
)


class RolloutOutcome(BaseModel):
    """One training rollout as the reward hook scored it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    problem_id: str  # the group: one prompt, G rollouts
    score: float
    truncated: bool
    has_box: bool
    turn_end: bool  # finished on <|im_end|> (otherwise on the base model's EOS, or truncated)
    answer: str | None  # the verifier's extracted final answer


def record_outcome(outcome: RolloutOutcome) -> None:
    path = os.environ.get(OUTCOMES_ENV)
    if path is None:
        return  # not inside an RL launch (unit tests, evaluation)
    # One short append per rollout: concurrent writers from several Ray workers do not interleave.
    with open(path, "a") as f:
        f.write(outcome.model_dump_json() + "\n")


def outcomes_by_step(outcome_dir: Path) -> dict[int, list[RolloutOutcome]]:
    return {
        step: [RolloutOutcome.model_validate(entry) for entry in entries]
        for step, entries in records_by_step(outcome_dir).items()
    }


def distinct_answers(answers: Sequence[str], same: Callable[[str, str], bool]) -> int:
    """Number of answer groups, each answer joining the first group whose representative it
    matches (maj@k's grouping, soe/utils/metric/pass_at_k.py)."""
    representatives: list[str] = []
    for answer in answers:
        if not any(same(answer, r) for r in representatives):
            representatives.append(answer)
    return len(representatives)


def step_metrics(
    outcomes: Sequence[RolloutOutcome],
    lengths: Sequence[int],
    *,
    prompts: int,
    group_size: int,
    same: Callable[[str, str], bool],
) -> dict[str, float]:
    """soe metrics of one step. Raises unless the step holds exactly `prompts` groups of
    `group_size` rollouts and as many lengths: a lost rollout must not pass silently."""
    groups: dict[str, list[RolloutOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault(outcome.problem_id, []).append(outcome)
    sizes = sorted({len(g) for g in groups.values()})
    if len(groups) != prompts or sizes != [group_size]:
        raise ValueError(
            f"expected {prompts} groups of {group_size}, got {len(groups)} with sizes {sizes}"
        )
    if len(lengths) != len(outcomes):
        raise ValueError(f"{len(outcomes)} outcomes but {len(lengths)} length records")
    scores = [[o.score for o in g] for g in groups.values()]
    means = [statistics.fmean(g) for g in scores]
    n = len(outcomes)
    finished = [o for o in outcomes if not o.truncated]
    # Diversity of the policy's final answers: a truncated rollout has no final answer.
    distinct = [
        distinct_answers([o.answer for o in g if not o.truncated and o.answer is not None], same)
        for g in groups.values()
    ]
    # "inclusive" is numpy's default linear percentile.
    p50, p95 = (statistics.quantiles(lengths, n=100, method="inclusive")[q - 1] for q in (50, 95))
    return {
        "soe/reward/mean": statistics.fmean(o.score for o in outcomes),
        "soe/group/reward_var_mean": statistics.fmean(statistics.pvariance(g) for g in scores),
        "soe/group/zero_var_frac": statistics.fmean(m in (0.0, 1.0) for m in means),
        "soe/group/all_correct_frac": statistics.fmean(m == 1.0 for m in means),
        "soe/group/all_wrong_frac": statistics.fmean(m == 0.0 for m in means),
        "soe/rollout/truncated_frac": (n - len(finished)) / n,
        "soe/rollout/boxed_frac": sum(o.has_box for o in outcomes) / n,
        "soe/rollout/stop_turn_end_frac": sum(o.turn_end for o in finished) / n,
        "soe/rollout/stop_other_frac": sum(not o.turn_end for o in finished) / n,
        "soe/length/mean": statistics.fmean(lengths),
        "soe/length/p50": p50,
        "soe/length/p95": p95,
        "soe/length/max": float(max(lengths)),
        "soe/answers/distinct_per_group_mean": statistics.fmean(distinct),
        # Finished rollouts without a final answer.
        "soe/answers/none_frac": sum(o.answer is None for o in finished) / max(1, len(finished)),
    }


def write_step_metrics(path: Path, by_step: dict[int, dict[str, float]]) -> None:
    """Replace the whole file at once, so a reader never sees half of it."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for step, data in sorted(by_step.items()):
            f.write(json.dumps({"step": step, "data": data}) + "\n")
    tmp.replace(path)


def read_step_metrics(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {int(x["step"]): x["data"] for x in lines}
