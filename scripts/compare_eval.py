"""Paired comparison of two evaluations of the same checkpoint on the same problems.

For each problem the difference b - a in pass@1, pass@k, truncation rate and length is taken,
then averaged per level and for the hard tier, with the paired standard error across problems.
Used for protocol changes (decision 009: FP8 vs BF16 KV cache), where the question is whether the
change moves the numbers by more than sampling noise.

    python scripts/compare_eval.py --a $SOE_DATA/eval/<run a> --b $SOE_DATA/eval/<run b> \\
        --out docs/reports/<name>.json
"""

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from soe.utils.jsonl import read_jsonl
from soe.utils.metric.eval_tables import Counts, ProblemResult
from soe.utils.metric.pass_at_k import pass_at_k

Metric = Callable[[Counts], float]


def metrics(k_max: int) -> dict[str, Metric]:
    return {
        "pass@1": lambda c: c.correct / c.n,
        f"pass@{k_max}": lambda c: pass_at_k(c.n, c.correct, k_max),
        "truncation_rate": lambda c: c.truncated / c.n,
        "mean_completion_tokens": lambda c: c.completion_tokens / c.n,
    }


def paired(a: list[float], b: list[float]) -> dict[str, float]:
    """Mean of a and b, mean difference b - a and its standard error across problems."""
    d = [y - x for x, y in zip(a, b, strict=True)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return {
        "a": sum(a) / n,
        "b": sum(b) / n,
        "diff": mean,
        "se": math.sqrt(var / n),
        "problems": n,
    }


def compare(
    a: dict[str, ProblemResult], b: dict[str, ProblemResult], *, main_k: int, hard_k: int
) -> dict[str, dict[str, dict[str, float]]]:
    common = sorted(a.keys() & b.keys())
    groups: dict[str, list[tuple[Counts, Counts]]] = defaultdict(list)
    hard: list[tuple[Counts, Counts]] = []
    for pid in common:
        groups[f"level {a[pid].level.value}"].append((a[pid].main, b[pid].main))
        if a[pid].hard and b[pid].hard:
            hard.append((a[pid].all, b[pid].all))
    out = {}
    for name, pairs in sorted(groups.items()):
        out[name] = {
            m: paired([f(x) for x, _ in pairs], [f(y) for _, y in pairs])
            for m, f in metrics(main_k).items()
        }
    if hard:
        out["hard tier"] = {
            m: paired([f(x) for x, _ in hard], [f(y) for _, y in hard])
            for m, f in metrics(hard_k).items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-k", type=int, default=16)
    parser.add_argument("--hard-k", type=int, default=64)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, ProblemResult]:
        return {r.problem_id: r for r in read_jsonl(path / "per_problem.jsonl", ProblemResult)}

    a, b = load(args.a), load(args.b)
    table = compare(a, b, main_k=args.main_k, hard_k=args.hard_k)
    only = {"a": len(a.keys() - b.keys()), "b": len(b.keys() - a.keys())}
    report = {"a": str(args.a), "b": str(args.b), "problems_in_one_run_only": only, **table}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"problems in one run only: {only}")
    for group, row in table.items():
        cells = "  ".join(
            f"{m} {v['a']:.4f}->{v['b']:.4f} ({v['diff']:+.4f} ± {v['se']:.4f})"
            for m, v in row.items()
            if m != "mean_completion_tokens"
        )
        print(f"{group:10} n={int(row['pass@1']['problems'])}  {cells}")


if __name__ == "__main__":
    main()
