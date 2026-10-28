import asyncio
from pathlib import Path

import pytest

from soe.workers.rollout.failures import FAILURES_ENV, failures_recorded, read_failures


def test_failure_is_recorded_and_reraised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "f.jsonl"
    monkeypatch.setenv(FAILURES_ENV, str(path))
    for _ in range(2):
        with pytest.raises(ValueError, match="boom"), failures_recorded("agent_loop"):
            raise ValueError("boom")
    entries = read_failures(path)
    assert [e["where"] for e in entries] == ["agent_loop", "agent_loop"]
    assert entries[0]["error"] == "ValueError: boom"


def test_cancellation_and_unset_env_record_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "f.jsonl"
    monkeypatch.setenv(FAILURES_ENV, str(path))

    async def cancelled() -> None:
        with failures_recorded("agent_loop"):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled())
    assert read_failures(path) == []

    monkeypatch.delenv(FAILURES_ENV)
    with pytest.raises(ValueError), failures_recorded("reward"):
        raise ValueError("outside a launch")
    assert not path.exists()
