"""transitions counts unsolved->solved and solved->lost at the suite's frozen k and threshold."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from soe.trainer.rl.checkpoints import CheckpointRecord, append_record
from soe.utils.dataset.problems import Level
from soe.utils.jsonl import write_jsonl
from soe.utils.metric.eval_tables import Counts, ProblemResult


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("transitions", Path("scripts/transitions.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(
    pid: str, level: Level, correct: int, hard: bool = False, n_all: int = 16
) -> ProblemResult:
    def counts(n: int, c: int) -> Counts:
        return Counts(
            n=n,
            correct=c,
            truncated=0,
            boxed=n,
            stopped=n,
            boxed_when_stopped=n,
            completion_tokens=100 * n,
        )

    return ProblemResult(
        problem_id=pid,
        problem_set="s",
        level=level,
        hard=hard,
        main=counts(16, min(correct, 16)),
        all=counts(n_all, correct),
        maj={},
    )


def test_transitions_per_level_and_hard_tier(tmp_path: Path) -> None:
    m = script()
    before = [
        result("b1", Level.B, 0),  # never solved -> pass@1 0.625: gained at k=1 and k=16
        result("b2", Level.B, 1),  # pass@16 = 1 (solved at 16), pass@1 0.06 -> after 0: lost at 16
        result("a1", Level.A, 16),
        result("h1", Level.A, 0, hard=True, n_all=64),
    ]
    after = [
        result("b1", Level.B, 10),
        result("b2", Level.B, 0),
        result("a1", Level.A, 16),
        result("h1", Level.A, 1, hard=True, n_all=64),  # pass@64 = 1: gained on the hard tier
    ]
    out = m.transitions(before, after, m.SUITE)
    assert out["threshold"] == 0.5
    b = out["main"]["B"]
    assert b["problems"] == 2
    assert b["1"] == {
        "unsolved_to_solved": 1,
        "solved_to_lost": 0,
        "solved_to_solved": 0,
        "unsolved_to_unsolved": 1,
    }
    assert b["16"]["unsolved_to_solved"] == 1 and b["16"]["solved_to_lost"] == 1
    assert out["main"]["A"]["16"]["solved_to_solved"] == 1
    assert out["hard"]["problems"] == 1
    assert out["hard"]["64"]["unsolved_to_solved"] == 1
    assert out["hard"]["1"]["unsolved_to_unsolved"] == 1

    # CLI with a run: newly solved held-out problems per million rollout tokens at that step.
    for name, rows in (("before", before), ("after", after)):
        (tmp_path / name).mkdir()
        write_jsonl(tmp_path / name / "per_problem.jsonl", rows)
    run = tmp_path / "run"
    run.mkdir()
    append_record(
        run,
        CheckpointRecord(
            step=4,
            rollout_tokens=2_000_000,
            saved_at=datetime(2026, 10, 29, tzinfo=UTC),
            slurm_job_id=None,
            hardware="",
            checkpoint_dir="",
            export_dir="",
        ),
    )
    report = tmp_path / "t.json"
    m.TransitionsCli(
        before=tmp_path / "before", after=tmp_path / "after", run_dir=run, step=4, report=report
    ).cli_cmd()
    efficiency = json.loads(report.read_text())["efficiency"]
    assert efficiency["newly_solved_heldout"] == {"1": 1, "16": 1}
    assert efficiency["newly_solved_heldout_per_million_rollout_tokens"] == {"1": 0.5, "16": 0.5}
