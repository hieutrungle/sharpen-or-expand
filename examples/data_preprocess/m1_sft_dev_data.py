"""Development SFT data for M1.8 (roadmap D9 (a)): wiring only, never in conclusions.

    SOE_DATA=... python examples/data_preprocess/m1_sft_dev_data.py \
        --config examples/data_preprocess/m1_sft_dev_data.yaml

Takes the correct, stopped thinking-mode completions of the M0 format probe (Qwen3.5-4B-Base on
200 RL-train problems), keeps those in decision 006's format (exactly one `</think>`), renders each
problem's user turn with the frozen suite's template, and splits by problem: `val_problems` problems
(seeded) go to val, the rest to train. Writes `train.jsonl` and `val.jsonl` (SftExample lines) under
`output_dir`; the report holds counts, the validation problem ids, the overlap with RL's validation
split and every sha256, but no text.
"""

import json
import os
import random
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import file_sha256, load_suite
from soe.trainer.sft.data import THINK_END, SftExample
from soe.utils.dataset.problems import Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl, write_jsonl

SHA256 = r"^[0-9a-f]{64}$"


class SftDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    probe: str
    probe_sha256: str = Field(pattern=SHA256)
    problems: str
    problems_sha256: str = Field(pattern=SHA256)
    suite: str
    rl_val: str
    output_dir: str
    report: str
    val_problems: int = Field(gt=0)
    seed: int


def checked(data_root: Path, relative: str, sha256: str) -> Path:
    path = data_root / relative
    if file_sha256(path) != sha256:
        raise ValueError(f"{relative} does not match its sha256")
    return path


class SftDataCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="examples/data_preprocess/m1_sft_dev_data.py", cli_kebab_case=True
    )

    config: Path

    def cli_cmd(self) -> None:
        config = SftDataConfig.model_validate(yaml.safe_load(self.config.read_text()))
        data_root = Path(os.environ["SOE_DATA"])
        samples = read_jsonl(checked(data_root, config.probe, config.probe_sha256), SampleRecord)
        problems = {
            p.problem_id: p
            for p in read_jsonl(
                checked(data_root, config.problems, config.problems_sha256), Problem
            )
        }
        prompt = load_suite(Path(config.suite)).prompt
        correct = [s for s in samples if s.correct and s.finish_reason == "stop"]
        formatted = [s for s in correct if s.completion.count(THINK_END) == 1]
        examples = [
            SftExample(
                problem_id=s.problem_id,
                user=prompt.render(problems[s.problem_id].question),
                completion=s.completion,
            )
            for s in formatted
        ]
        ids = sorted({e.problem_id for e in examples})
        val_ids = set(random.Random(config.seed).sample(ids, config.val_problems))
        splits = {
            "train": [e for e in examples if e.problem_id not in val_ids],
            "val": [e for e in examples if e.problem_id in val_ids],
        }
        out = data_root / config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, rows in splits.items():
            path = out / f"{name}.jsonl"
            write_jsonl(path, rows)
            files[name] = {
                "path": f"{config.output_dir}/{name}.jsonl",
                "examples": len(rows),
                "problems": len({e.problem_id for e in rows}),
                "sha256": file_sha256(path),
            }
        rl_val = {p.problem_id for p in read_jsonl(data_root / config.rl_val, Problem)}
        report = {
            "version": config.version,
            "probe_samples": len(samples),
            "correct_and_stopped": len(correct),
            "one_think_end": len(formatted),
            "files": files,
            "val_problem_ids": sorted(val_ids),
            "problems_in_rl_val_split": sorted(set(ids) & rl_val),
        }
        Path(config.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    CliApp.run(SftDataCli)
