"""Evaluation entry point called by slurm/eval.sbatch.

The suite's problem-file hashes are checked first, so a drifted yardstick never produces numbers.
Sampling with vLLM and writing SampleRecord JSONL is M0 work (docs/roadmap/M0.md); until then
this checks the suite and stops.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import hash_mismatches, load_suite


class EvaluateCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_eval", env_prefix="SOE_EVAL_", cli_kebab_case=True
    )

    suite: Path
    checkpoint: str
    output: Path

    def cli_cmd(self) -> None:
        suite = load_suite(self.suite)
        bad = hash_mismatches(suite, Path(os.environ["SOE_DATA"]))
        if bad:
            raise SystemExit(f"refusing to evaluate: problem sets changed or missing: {bad}")
        raise NotImplementedError(f"suite {suite.name} verified; sampling is implemented in M0")


if __name__ == "__main__":
    CliApp.run(EvaluateCli)
