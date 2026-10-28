"""Carry a checkpoint's evaluation to a new suite version when only some problems changed.

When a suite version repairs some problem files (decision 007), problems whose record (text,
answer, level) is identical in both versions keep their earlier results, and the changed ones
are re-evaluated with `main_eval --only-problems`. This merges the two into per_problem.jsonl
and report.json for the new suite. It refuses unless both runs used the same checkpoint and the
two suites the same sampling, prompt, scoring and tiers, and every changed problem was re-run.

    python scripts/merge_eval.py --old-suite soe/trainer/config/eval_suite_v1.yaml \\
        --old-eval $SOE_DATA/eval/<v1 run> --patch-eval $SOE_DATA/eval/<patch run> \\
        --output $SOE_DATA/eval/<v2 run>
"""

import argparse
import json
import os
from pathlib import Path

from soe.trainer.config.eval_suite import EvalSuite, load_suite
from soe.trainer.main_eval import EvalPlan, eval_units, print_main_table, report_tables
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.metric.eval_tables import ProblemResult


def problems_by_id(suite: EvalSuite, data_root: Path) -> dict[str, Problem]:
    return {
        p.problem_id: p
        for problem_set in suite.problem_sets
        for p in read_jsonl(data_root / problem_set.path, Problem)
    }


def merge(
    old: EvalSuite,
    new: EvalSuite,
    old_results: dict[str, ProblemResult],
    patch_results: dict[str, ProblemResult],
    data_root: Path,
) -> tuple[list[ProblemResult], dict[str, int]]:
    same = ("sampling", "prompt", "scoring", "main", "hard")
    if any(getattr(old, field) != getattr(new, field) for field in same):
        raise SystemExit("the suites differ in sampling, prompt, scoring or tiers")
    if (old.hard_subset and old.hard_subset.sha256) != (new.hard_subset and new.hard_subset.sha256):
        raise SystemExit("the hard subset changed; re-evaluate it in full")
    old_problems, new_problems = problems_by_id(old, data_root), problems_by_id(new, data_root)
    merged, reused, patched = [], 0, 0
    for unit in eval_units(new, data_root, None):
        pid = unit.problem.problem_id
        if pid in patch_results:
            merged.append(patch_results[pid])
            patched += 1
        elif old_problems.get(pid) == new_problems[pid] and pid in old_results:
            merged.append(old_results[pid])
            reused += 1
        else:
            raise SystemExit(f"{pid} changed between the suites but was not re-evaluated")
    unused = set(patch_results) - {r.problem_id for r in merged}
    if unused:
        raise SystemExit(f"patch results for problems outside the new suite: {sorted(unused)[:5]}")
    return merged, {"reused": reused, "patched": patched}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-suite", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--old-eval", type=Path, required=True)
    parser.add_argument("--patch-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = Path(os.environ["SOE_DATA"])
    old, new = load_suite(args.old_suite), load_suite(args.suite)
    plans = {
        name: EvalPlan.model_validate_json((path / "plan.json").read_text())
        for name, path in (("old", args.old_eval), ("patch", args.patch_eval))
    }
    if plans["old"].suite_name != old.name or plans["patch"].suite_name != new.name:
        raise SystemExit("the evaluation directories do not match the given suites")
    if (plans["old"].checkpoint, plans["old"].revision) != (
        plans["patch"].checkpoint,
        plans["patch"].revision,
    ):
        raise SystemExit("the two evaluations are of different checkpoints")
    if plans["old"].limit_per_set is not None or plans["old"].only_problems_sha256 is not None:
        raise SystemExit("the old evaluation must be a full one")

    def results(path: Path) -> dict[str, ProblemResult]:
        return {r.problem_id: r for r in read_jsonl(path / "per_problem.jsonl", ProblemResult)}

    merged, counts = merge(old, new, results(args.old_eval), results(args.patch_eval), data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "per_problem.jsonl", merged)
    provenance = {
        "suite_name": new.name,
        "merged_from": {name: plan.model_dump() for name, plan in plans.items()},
        **counts,
    }
    report = {"plan": provenance, **report_tables(merged, new)}
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(counts))
    print_main_table(report["main"], new.main.ks)


if __name__ == "__main__":
    main()
