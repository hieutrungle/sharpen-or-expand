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
from soe.utils.reward_score.math_verify import answers_equivalent

Pair = tuple[str, str]  # (extracted answer, gold)


def judge(pair: Pair) -> bool:
    return answers_equivalent(*pair)


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
    records = []  # (directory, problem_id, sample_index, answer or None if never correct, old)
    for directory in args.samples:
        for shard in sorted(directory.glob("samples-*-of-*.jsonl")):
            with shard.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r["problem_id"] not in gold:
                        continue
                    judged = r["finish_reason"] == "stop" and r["extracted_answer"] is not None
                    answer = r["extracted_answer"] if judged else None
                    records.append(
                        (directory.name, r["problem_id"], r["sample_index"], answer, r["correct"])
                    )
    pairs = sorted({(a, gold[pid]) for _, pid, _, a, _ in records if a is not None})
    with Pool(args.workers) as pool:
        verdict = dict(zip(pairs, pool.map(judge, pairs, chunksize=64), strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    flips: Counter[str] = Counter()
    before: Counter[tuple[str, str]] = Counter()
    after: Counter[tuple[str, str]] = Counter()
    with (args.out / "flips.jsonl").open("w") as f:
        for directory, pid, index, answer, old in records:
            new = answer is not None and verdict[(answer, gold[pid])]
            before[(directory, pid)] += old
            after[(directory, pid)] += new
            if new != old:
                flips[f"{old}->{new}"] += 1
                row = {"samples": directory, "problem_id": pid, "sample_index": index}
                row |= {"answer": answer, "gold": gold[pid], "old": old, "new": new}
                f.write(json.dumps(row) + "\n")
    with (args.out / "counts.jsonl").open("w") as f:
        for directory, pid in sorted(before.keys() | after.keys()):
            row = {"samples": directory, "problem_id": pid}
            f.write(
                json.dumps(row | {"old": before[(directory, pid)], "new": after[(directory, pid)]})
                + "\n"
            )
    print(f"{len(records)} records, {len(pairs)} distinct (answer, gold) pairs")
    print("flips:", dict(flips))


if __name__ == "__main__":
    main()
