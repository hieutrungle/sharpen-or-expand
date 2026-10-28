import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from soe.trainer.main_rl import RolloutFailure, run_until_failure

FAILURE = '{"where": "reward", "pid": 1, "error": "ValueError: x"}\n'


def child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_first_failure_stops_a_running_trainer(tmp_path: Path) -> None:
    failures = tmp_path / "f.jsonl"
    # The fake trainer records a failure, then would run for a minute.
    code = f"open({str(failures)!r}, 'a').write({FAILURE!r}); import time; time.sleep(60)"
    start = time.monotonic()
    with pytest.raises(RolloutFailure, match="1 rollout failure"):
        run_until_failure(child(code), dict(os.environ), failures, poll_seconds=0.2)
    assert time.monotonic() - start < 30


def test_failure_in_the_last_interval_still_fails(tmp_path: Path) -> None:
    failures = tmp_path / "f.jsonl"
    code = f"open({str(failures)!r}, 'a').write({FAILURE!r})"
    with pytest.raises(RolloutFailure):
        run_until_failure(child(code), dict(os.environ), failures, poll_seconds=0.2)


def test_exit_code_is_checked_and_clean_runs_pass(tmp_path: Path) -> None:
    failures = tmp_path / "f.jsonl"
    with pytest.raises(subprocess.CalledProcessError):
        run_until_failure(child("raise SystemExit(3)"), dict(os.environ), failures, 0.2)
    run_until_failure(child("pass"), dict(os.environ), failures, poll_seconds=0.2)
