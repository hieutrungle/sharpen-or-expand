"""List every stored verdict the current verifier would change, to review a verifier fix before
it reaches any report.

Sample records keep the verdict of the verifier at sampling time. This re-judges every record of
the given sample directories against the suite's gold answers and writes the changed ones
(flips.jsonl) and per-problem correct counts before and after (counts.jsonl). A fix that only
adds accepted notations should change incorrect to correct only; review each distinct flipped
answer for false positives.

    python scripts/verifier_diff.py --samples $SOE_DATA/eval/<run> [--samples ...] --out <dir>
"""

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl
from soe.utils.reward_score.math_verify import score_response

_gold: dict[str, str] = {}


def _init(gold: dict[str, str]) -> None:
    _gold.update(gold)


def judge_shard(shard: Path) -> tuple[list[dict[str, object]], dict[str, tuple[int, int]]]:
    """Flipped records and per-problem (old, new) correct counts of one sample file, judged from
    the whole completion as the report stage does."""
    flips: list[dict[str, object]] = []
    counts: dict[str, tuple[int, int]] = {}
    with shard.open() as f:
        for line in f:
            r = json.loads(line)
            gold = _gold.get(r["problem_id"])
            if gold is None:
                continue
            verdict = score_response(r["completion"], gold)
            new = r["finish_reason"] == "stop" and verdict.correct
            old_n, new_n = counts.get(r["problem_id"], (0, 0))
            counts[r["problem_id"]] = (old_n + r["correct"], new_n + new)
            if new != r["correct"]:
                flips.append(
                    {
                        "samples": shard.parent.name,
                        "problem_id": r["problem_id"],
                        "sample_index": r["sample_index"],
                        "answer": verdict.extracted,
                        "gold": gold,
                        "old": r["correct"],
                        "new": new,
                    }
                )
    return flips, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--samples", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()
    data_root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    gold = {
        p.problem_id: p.answer
        for s in suite.problem_sets
        for p in read_jsonl(data_root / s.path, Problem)
    }
    shards = [s for d in args.samples for s in sorted(d.glob("samples-*-of-*.jsonl"))]
    with Pool(args.workers, initializer=_init, initargs=(gold,)) as pool:
        results = pool.map(judge_shard, shards)
    args.out.mkdir(parents=True, exist_ok=True)
    summary: Counter[str] = Counter()
    per_problem: Counter[tuple[str, str, str]] = Counter()
    with (args.out / "flips.jsonl").open("w") as f:
        for shard, (flips, counts) in zip(shards, results, strict=True):
            for row in flips:
                summary[f"{row['old']}->{row['new']}"] += 1
                f.write(json.dumps(row) + "\n")
            for pid, (old, new) in counts.items():
                per_problem[(shard.parent.name, pid, "old")] += old
                per_problem[(shard.parent.name, pid, "new")] += new
    with (args.out / "counts.jsonl").open("w") as f:
        for directory, pid in sorted({(d, p) for d, p, _ in per_problem}):
            row = {"samples": directory, "problem_id": pid}
            row |= {
                "old": per_problem[(directory, pid, "old")],
                "new": per_problem[(directory, pid, "new")],
            }
            f.write(json.dumps(row) + "\n")
    print(f"{len(shards)} sample files; flips:", dict(summary))


if __name__ == "__main__":
    main()
