"""Per-problem results and per-group tables for one evaluated checkpoint (decision 004).

Every problem has its main-tier samples (indices 0..15). Hard-subset problems have more (0..63);
their first 16 also count as their main-tier samples, so both tiers come from one sampling pass.
"""

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from soe.utils.dataset.problems import Level
from soe.utils.dataset.records import SampleRecord
from soe.utils.metric.pass_at_k import maj_at_k, pass_at_k

Same = Callable[[str, str], bool]


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int
    correct: int
    truncated: int
    boxed: int
    stopped: int
    boxed_when_stopped: int
    completion_tokens: int


class ProblemResult(BaseModel):
    """One line of per_problem.jsonl: what transitions and figures need, no completions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    problem_id: str
    problem_set: str
    level: Level
    hard: bool
    main: Counts  # the first main-tier samples
    all: Counts  # every sample (equal to main outside the hard subset)
    maj: dict[int, bool]  # maj@k over the first k samples, for every k of the problem's tiers


def counts(records: Sequence[SampleRecord]) -> Counts:
    stopped = [r for r in records if r.finish_reason == "stop"]
    return Counts(
        n=len(records),
        correct=sum(r.correct for r in records),
        truncated=sum(r.finish_reason == "length" for r in records),
        boxed=sum(r.extracted_answer is not None for r in records),
        stopped=len(stopped),
        boxed_when_stopped=sum(r.extracted_answer is not None for r in stopped),
        completion_tokens=sum(r.completion_tokens for r in records),
    )


def problem_result(
    records: Sequence[SampleRecord],
    *,
    problem_set: str,
    level: Level,
    hard: bool,
    n_main: int,
    maj_ks: Sequence[int],
    same: Same,
) -> ProblemResult:
    ordered = sorted(records, key=lambda r: r.sample_index)
    if len({r.problem_id for r in ordered}) != 1:
        raise ValueError("records of more than one problem")
    if [r.sample_index for r in ordered] != list(range(len(ordered))) or len(ordered) < n_main:
        raise ValueError(f"{ordered[0].problem_id}: need sample indices 0..n-1 with n >= {n_main}")
    answers = [r.extracted_answer for r in ordered]
    correct = [r.correct for r in ordered]
    return ProblemResult(
        problem_id=ordered[0].problem_id,
        problem_set=problem_set,
        level=level,
        hard=hard,
        main=counts(ordered[:n_main]),
        all=counts(ordered),
        maj={k: maj_at_k(answers, correct, k, same) for k in maj_ks if k <= len(ordered)},
    )


def tier_table(
    results: Sequence[ProblemResult], ks: Sequence[int], *, hard: bool
) -> dict[str, float]:
    """Problem-weighted pass@k and maj@k per k, plus sample-pooled behaviour rates, for one tier.

    The main tier uses each problem's first main-tier samples; the hard tier uses all of them.
    """
    if not results:
        raise ValueError("empty group")
    tier = [r.all if hard else r.main for r in results]
    table: dict[str, float] = {"problems": len(results)}
    for k in ks:
        table[f"pass@{k}"] = sum(pass_at_k(c.n, c.correct, k) for c in tier) / len(results)
        table[f"maj@{k}"] = sum(r.maj[k] for r in results) / len(results)
    samples = sum(c.n for c in tier)
    table["truncation_rate"] = sum(c.truncated for c in tier) / samples
    table["boxed_rate"] = sum(c.boxed for c in tier) / samples
    table["boxed_rate_when_stopped"] = sum(c.boxed_when_stopped for c in tier) / max(
        1, sum(c.stopped for c in tier)
    )
    table["mean_completion_tokens"] = sum(c.completion_tokens for c in tier) / samples
    return {key: round(value, 4) for key, value in table.items()}
