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


def test_fatal_engine_lines_reads_only_new_whole_lines(tmp_path: Path) -> None:
    from soe.workers.rollout.failures import fatal_engine_lines

    log = tmp_path / "job.log"
    log.write_text("INFO ok\nERROR Worker failed with error 'x'\nERROR EngineDead")
    lines, offset = fatal_engine_lines(log, 0)
    assert lines == ["ERROR Worker failed with error 'x'"]  # the unfinished line waits
    with log.open("a") as f:
        f.write("Error: boom\n")
    lines, offset = fatal_engine_lines(log, offset)
    assert lines == ["ERROR EngineDeadError: boom"]
    assert fatal_engine_lines(log, offset) == ([], offset)
    assert fatal_engine_lines(tmp_path / "missing.log", 5) == ([], 5)
