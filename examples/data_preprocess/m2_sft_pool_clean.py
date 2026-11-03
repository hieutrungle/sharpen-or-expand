"""M2.3: the decontaminated SFT pool (decision 027): the M2.2 pool minus every problem that a
contamination report lists, applied in rounds until the reports list nothing.

    SOE_DATA=... python examples/data_preprocess/m2_sft_pool_clean.py \\
        --config examples/data_preprocess/m2_sft_pool_clean.yaml

A round's reports are `scripts/contamination_report.py` outputs, all made on the pool left after the
previous round (checked by sha256): Levels A, B, C at the default 20% share, Level C at 10%, and the
RL pool (RL train v3.1 and its validation split). Every pool problem in any listed pair is dropped,
whatever the hand review read it as (user, M2.3: the review's keeps are variants of evaluation
problems). Why rounds: the report ignores n-grams shared by more than 20 pool problems as
boilerplate, so removing problems shrinks that set and can list new pairs. Writes
`pool_clean.jsonl` and a report with counts per round and sha256s. No text.
"""

import json
import os
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import file_sha256
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl, write_jsonl

SHA256 = r"^[0-9a-f]{64}$"


class CleanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: str  # relative to $SOE_DATA
    pool_sha256: str = Field(pattern=SHA256)
    rounds: list[list[Path]] = Field(min_length=1)  # per round, its contamination reports
    output: str  # relative to $SOE_DATA
    report: Path  # repo path


def listed(report: dict[str, list[dict[str, str]]]) -> set[str]:
    """Pool problems in any pair of a contamination report."""
    return {p["train"] for p in report["pairs"]}


def clean(problems: list[Problem], dropped: set[str]) -> list[Problem]:
    missing = dropped - {p.problem_id for p in problems}
    if missing:
        raise ValueError(f"dropped ids not in the pool: {sorted(missing)[:5]}")
    return [p for p in problems if p.problem_id not in dropped]


def jsonl_sha256(problems: list[Problem]) -> str:
    """The sha256 the file would have, so a round's reports can be matched to its pool."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pool.jsonl"
        write_jsonl(path, problems)
        return file_sha256(path)


class CleanCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="examples/data_preprocess/m2_sft_pool_clean.py", cli_kebab_case=True
    )

    config: Path

    def cli_cmd(self) -> None:
        config = CleanConfig.model_validate(yaml.safe_load(self.config.read_text()))
        root = Path(os.environ["SOE_DATA"])
        pool_path = root / config.pool
        if file_sha256(pool_path) != config.pool_sha256:
            raise SystemExit(f"{pool_path} does not match its recorded sha256")
        current = read_jsonl(pool_path, Problem)
        rounds = []
        for k, paths in enumerate(config.rounds, start=1):
            sha = jsonl_sha256(current)
            reports = {str(path): json.loads(path.read_text()) for path in paths}
            for name, report in reports.items():
                if report["train"]["sha256"] != sha:
                    raise SystemExit(f"round {k}: {name} was made on another pool than {sha}")
            per_report = {name: listed(report) for name, report in reports.items()}
            dropped = set().union(*per_report.values())
            before = len(current)
            current = clean(current, dropped)
            rounds.append(
                {
                    "round": k,
                    "pool_sha256": sha,
                    "problems": before,
                    "listed": {name: len(ids) for name, ids in per_report.items()},
                    "dropped": len(dropped),
                    "reports_sha256": {str(path): file_sha256(path) for path in paths},
                }
            )
        out = root / config.output
        write_jsonl(out, current)
        report = {
            "pool": {"path": config.pool, "sha256": config.pool_sha256},
            "rounds": rounds,
            "clean": {"path": config.output, "sha256": file_sha256(out), "problems": len(current)},
        }
        config.report.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    CliApp.run(CleanCli)
