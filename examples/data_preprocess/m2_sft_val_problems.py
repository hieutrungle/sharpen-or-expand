"""M2.11 diagnostic: the problems of the SFT validation split as a `Problem` file, for main_eval.

    SOE_DATA=... python examples/data_preprocess/m2_sft_val_problems.py \\
        --config examples/data_preprocess/m2_sft_val_problems.yaml

The 500 held-out problems of `val.jsonl` (M2.5), taken from the clean kept problems they were
drawn from, sorted by id. Not an evaluation level and not used for any choice: it shows whether
SFT run #1 is worse in distribution too, or only on the RL validation split (user, 2026-11-05).
Prints the output's sha256 for the suite YAML.
"""

import argparse
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import file_sha256
from soe.trainer.main_rl import checked_data_file
from soe.trainer.sft.data import SftExample
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl, write_jsonl

SHA256 = r"^[0-9a-f]{64}$"


class ValProblemsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    problems: str  # the clean kept problems (M2.5), relative to $SOE_DATA
    problems_sha256: str = Field(pattern=SHA256)
    val: str  # the SFT validation file, relative to $SOE_DATA
    val_sha256: str = Field(pattern=SHA256)
    output: str  # relative to $SOE_DATA


def val_problems(problems: list[Problem], val: list[SftExample]) -> list[Problem]:
    """The problems of the validation examples, once each, sorted by id; all must exist."""
    ids = {e.problem_id for e in val}
    chosen = sorted((p for p in problems if p.problem_id in ids), key=lambda p: p.problem_id)
    if len(chosen) != len(ids):
        raise SystemExit(
            f"{len(ids) - len(chosen)} validation problems are not in the problem file"
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = ValProblemsConfig.model_validate(
        yaml.safe_load(parser.parse_args().config.read_text())
    )
    root = Path(os.environ["SOE_DATA"])
    problems = read_jsonl(checked_data_file(root, config.problems, config.problems_sha256), Problem)
    val = read_jsonl(checked_data_file(root, config.val, config.val_sha256), SftExample)
    chosen = val_problems(problems, val)
    write_jsonl(root / config.output, chosen)
    print(f"{len(chosen)} problems -> {config.output} sha256 {file_sha256(root / config.output)}")


if __name__ == "__main__":
    main()
