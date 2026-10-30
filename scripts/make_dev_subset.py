"""The fixed M1.9 development evaluation subset: problem ids only (no text), for `main_eval
--only-problems`.

    SOE_DATA=... python scripts/make_dev_subset.py --out examples/eval/m1_dev_subset.txt

Draws `per_set` problems (seeded) from every problem set of the given levels (A and B: Level C is
never used for any selection or development reading). Problems of the hard subset (64 samples) and
the suite's excluded problems are left out, so every dev problem gets the main tier's 16 samples.
"""

import hashlib
import json
import os
import random
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import hash_mismatches, load_suite
from soe.utils.dataset.problems import Level, Problem
from soe.utils.jsonl import read_jsonl

SUITE = Path("soe/trainer/config/eval_suite.yaml")


class DevSubsetCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/make_dev_subset.py", cli_kebab_case=True
    )

    suite: Path = SUITE
    levels: list[Level] = [Level.A, Level.B]
    per_set: int = 25
    seed: int = 20261030
    out: Path

    def cli_cmd(self) -> None:
        suite = load_suite(self.suite)
        data_root = Path(os.environ["SOE_DATA"])
        mismatched = hash_mismatches(suite, data_root)
        if mismatched:
            raise ValueError(f"problem files differ from the suite: {mismatched}")
        skip = set(suite.excluded_problems)
        if suite.hard_subset is not None:
            hard = read_jsonl(data_root / suite.hard_subset.path, Problem)
            skip |= {p.problem_id for p in hard}
        rng = random.Random(self.seed)
        chosen: dict[str, list[str]] = {}
        for problem_set in suite.problem_sets:
            if problem_set.level not in self.levels:
                continue
            ids = sorted(
                p.problem_id
                for p in read_jsonl(data_root / problem_set.path, Problem)
                if p.problem_id not in skip
            )
            chosen[problem_set.name] = sorted(rng.sample(ids, min(self.per_set, len(ids))))
        lines = "".join(f"{pid}\n" for ids in chosen.values() for pid in ids)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(lines)
        summary = {
            "suite": suite.name,
            "seed": self.seed,
            "per_set": {name: len(ids) for name, ids in chosen.items()},
            "problems": sum(len(ids) for ids in chosen.values()),
            "sha256": hashlib.sha256(lines.encode()).hexdigest(),
        }
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    CliApp.run(DevSubsetCli)
