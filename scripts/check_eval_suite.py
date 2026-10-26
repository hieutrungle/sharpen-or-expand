"""Check that every problem file of the frozen evaluation suite matches its pinned sha256.

Run inside the container (or any env with the soe package): exits 1 on any mismatch.
    python scripts/check_eval_suite.py [--suite soe/trainer/config/eval_suite.yaml]
"""

import argparse
import os
import sys
from pathlib import Path

from soe.trainer.config.eval_suite import hash_mismatches, load_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    args = parser.parse_args()
    suite = load_suite(args.suite)
    bad = hash_mismatches(suite, Path(os.environ["SOE_DATA"]))
    sets = len(suite.problem_sets) + (suite.hard_subset is not None)
    print(
        f"{suite.name}: {sets - len(bad)}/{sets} problem sets match their sha256; mismatches: {bad}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
