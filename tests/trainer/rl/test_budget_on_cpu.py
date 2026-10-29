from pathlib import Path

from soe.trainer.rl.budget import BudgetState, last_step, read_stop_step, write_stop_step


def state(step: int, tokens: int, per_step: float = 100.0, hours: float = 1.0) -> BudgetState:
    return BudgetState(
        step=step,
        rollout_tokens=tokens,
        tokens_per_step=per_step,
        hours_used=hours,
        hours_per_step=0.5,
    )


def test_stops_at_the_first_step_that_reaches_the_token_budget() -> None:
    assert last_step(state(10, 1_000), budget_tokens=1_250) == 13  # 1,100 / 1,200 / 1,300
    assert last_step(state(10, 1_000), budget_tokens=1_300) == 13  # reached exactly
    assert last_step(state(10, 1_000), budget_tokens=900) == 10  # already past: stop now


def test_the_earliest_of_tokens_hours_and_steps_wins() -> None:
    s = state(10, 1_000, hours=4.0)
    assert last_step(s, budget_tokens=10_000, max_hours=5.0) == 12  # 2 more steps fit in 1 h
    assert last_step(s, budget_tokens=10_000, max_hours=4.2) == 10  # not even one more step
    assert last_step(s, budget_tokens=10_000, max_steps=11) == 11
    assert last_step(s, budget_tokens=1_150, max_hours=100.0, max_steps=50) == 12


def test_the_estimate_survives_a_resume(tmp_path: Path) -> None:
    assert read_stop_step(tmp_path) is None
    write_stop_step(tmp_path, 42, state(10, 1_000))
    assert read_stop_step(tmp_path) == 42
