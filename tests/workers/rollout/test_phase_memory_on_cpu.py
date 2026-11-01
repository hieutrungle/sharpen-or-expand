"""The phase-memory wrapper marks before and after a method, keeps verl's metadata, and is inert
without $SOE_PHASE_MEMORY (M1.10)."""

import asyncio
from typing import Any

import pytest

from soe.workers.rollout import phase_memory


def test_close_interval_does_nothing_without_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(phase_memory.PHASE_MEMORY_ENV, raising=False)
    phase_memory.close_interval("generate")  # would need torch if it did anything


def test_wrap_marks_both_sides_and_keeps_register_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(phase_memory, "close_interval", seen.append)

    def compute_log_prob(self: Any, data: int) -> int:
        seen.append("run")
        return data + 1

    compute_log_prob.attrs_3141562937 = {"dispatch_mode": "x"}  # type: ignore[attr-defined]
    wrapped = phase_memory.wrap(compute_log_prob, "compute_log_prob")
    assert wrapped(None, 1) == 2
    assert seen == ["before_compute_log_prob", "run", "compute_log_prob"]
    assert wrapped.attrs_3141562937 == {"dispatch_mode": "x"}  # type: ignore[attr-defined]
    assert wrapped.__name__ == "compute_log_prob"
    assert wrapped.soe_phase_memory  # type: ignore[attr-defined]


def test_wrap_keeps_a_coroutine_a_coroutine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(phase_memory, "close_interval", seen.append)

    async def update_weights(self: Any) -> str:
        seen.append("run")
        return "ok"

    wrapped = phase_memory.wrap(update_weights, "update_weights")
    assert asyncio.iscoroutinefunction(wrapped)
    assert asyncio.run(wrapped(None)) == "ok"
    assert seen == ["before_update_weights", "run", "update_weights"]
