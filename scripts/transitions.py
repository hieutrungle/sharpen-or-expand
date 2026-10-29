"""Capability transitions of one evaluated checkpoint against a reference (M1.6).

    python scripts/transitions.py --before <eval dir of the reference> --after <eval dir> \
        [--run-dir runs/<id> --step N] --report <out.json>

Both evaluation directories hold main_eval's per_problem.jsonl over the same problems. A problem is
solved at k when its pass@k estimate reaches the suite's frozen threshold; transitions are counted
at the suite's transition_ks: per level on the main tier (16 samples), and on the hard subset with
all its samples. With --run-dir and --step, the newly solved held-out (Level B) problems are also
given per million rollout tokens of that checkpoint (checkpoints.jsonl, decision 018).
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.rl.checkpoints import read_records
from soe.utils.dataset.problems import Level
from soe.utils.jsonl import read_jsonl
from soe.utils.metric.eval_tables import Counts, ProblemResult
from soe.utils.metric.transitions import Transition, count_transitions, is_solved

SUITE = Path(__file__).resolve().parents[1] / "soe/trainer/config/eval_suite.yaml"


def solved(results: dict[str, Counts], k: int, threshold: float) -> dict[str, bool]:
    return {pid: is_solved(c.n, c.correct, k, threshold) for pid, c in results.items()}


def tier_transitions(
    before: dict[str, Counts], after: dict[str, Counts], ks: Sequence[int], threshold: float
) -> dict[str, dict[str, int]]:
    return {
        str(k): count_transitions(solved(before, k, threshold), solved(after, k, threshold))
        for k in ks
    }


def transitions(
    before: list[ProblemResult], after: list[ProblemResult], suite_path: Path
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    threshold = suite.scoring.transition_solved_threshold
    if {r.problem_id for r in before} != {r.problem_id for r in after}:
        raise ValueError("the two evaluations cover different problems")
    main: dict[str, Any] = {}
    for level in sorted({r.level for r in before}):
        b = {r.problem_id: r.main for r in before if r.level == level}
        a = {r.problem_id: r.main for r in after if r.level == level}
        main[level] = {
            "problems": len(b),
            **tier_transitions(b, a, suite.main.transition_ks, threshold),
        }
    hard_b = {r.problem_id: r.all for r in before if r.hard}
    hard_a = {r.problem_id: r.all for r in after if r.hard}
    return {
        "threshold": threshold,
        "main": main,
        "hard": {
            "problems": len(hard_b),
            **tier_transitions(hard_b, hard_a, suite.hard.transition_ks, threshold),
        },
    }


class TransitionsCli(BaseSettings):
    model_config = SettingsConfigDict(cli_prog_name="scripts/transitions.py", cli_kebab_case=True)

    before: Path
    after: Path
    suite: Path = SUITE
    run_dir: Path | None = None
    step: int | None = None
    report: Path

    def cli_cmd(self) -> None:
        before = read_jsonl(self.before / "per_problem.jsonl", ProblemResult)
        after = read_jsonl(self.after / "per_problem.jsonl", ProblemResult)
        result: dict[str, Any] = {
            "before": str(self.before),
            "after": str(self.after),
            **transitions(before, after, self.suite),
        }
        if (self.run_dir is None) != (self.step is None):
            raise ValueError("give --run-dir and --step together")
        if self.run_dir is not None:
            records = {r.step: r for r in read_records(self.run_dir)}
            tokens = records[self.step].rollout_tokens  # KeyError: no checkpoint at that step
            heldout = result["main"].get(Level.B, {})
            gained = {k: v[Transition.GAINED] for k, v in heldout.items() if k != "problems"}
            result["efficiency"] = {
                "step": self.step,
                "rollout_tokens": tokens,
                "newly_solved_heldout": gained,
                "newly_solved_heldout_per_million_rollout_tokens": {
                    k: n / (tokens / 1e6) for k, n in gained.items()
                },
            }
        self.report.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    CliApp.run(TransitionsCli)
