"""RL prompt check (M1.2), run in the container because it needs verl and transformers.

For every row of the RL parquet files, the prompt token ids verl's rollout builds (HFModelConfig's
tokenizer and processor, then the Continuous Token builder its agent loop uses) must equal the ids
the evaluation sampler sends to vLLM (the suite's chat template rendering, encoded by the same
tokenizer). It also records prompt lengths, which set `rollout.max_prompt_tokens`: verl silently
cuts text prompts longer than `rollout.prompt_length`.

    sbatch slurm/cpu.sbatch python scripts/check_rl_prompts.py \\
        --data-report docs/reports/m1_datasets_v3.1.json --out docs/reports/m1_rl_prompts.json
"""

import json
import os
import statistics
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import load_suite

MODELS = ("Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base")
ASSETS = Path("scripts/stage_assets.yaml")


def staged_path(hf_id: str) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    revision = next(
        a["revision"] for a in yaml.safe_load(ASSETS.read_text())["assets"] if a["hf_id"] == hf_id
    )
    return snapshot_download(hf_id, revision=revision, local_files_only=True), revision


def length_stats(lengths: list[int]) -> dict[str, float]:
    ordered = sorted(lengths)
    return {
        "max": ordered[-1],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "mean": round(statistics.fmean(ordered), 1),
    }


def check_model(
    hf_id: str, rows: dict[str, list[dict[str, Any]]], chat_kwargs: dict[str, bool]
) -> dict[str, Any]:
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
    from verl.workers.config import HFModelConfig

    path, revision = staged_path(hf_id)
    model = HFModelConfig(path=path)  # loads tokenizer and processor exactly as verl's workers do
    builder = create_continuous_token_builder(
        model.tokenizer,
        hf_model_type=getattr(model.hf_config, "model_type", None),
        chat_template_kwargs=chat_kwargs,
        processor=model.processor,
    )
    result: dict[str, Any] = {
        "revision": revision,
        "builder": type(builder).__name__,
        "processor": type(model.processor).__name__ if model.processor is not None else None,
    }
    for split, split_rows in rows.items():
        lengths, mismatches = [], []
        for row in split_rows:
            messages = [dict(m) for m in row["prompt"]]
            verl_ids = list(builder.build_initial_tokens(messages))
            text = model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            sampler_ids = model.tokenizer.encode(text)  # what vLLM does with the sampler's string
            if not text.endswith("<|im_start|>assistant\n<think>\n"):
                mismatches.append(
                    {"problem_id": row["extra_info"]["problem_id"], "why": "prompt end"}
                )
            elif verl_ids != sampler_ids:
                mismatches.append(
                    {"problem_id": row["extra_info"]["problem_id"], "why": "token ids"}
                )
            lengths.append(len(verl_ids))
        result[split] = {
            "rows": len(split_rows),
            "mismatches": len(mismatches),
            "first_mismatches": mismatches[:5],
            "prompt_tokens": length_stats(lengths),
        }
    return result


class CheckCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python scripts/check_rl_prompts.py", cli_kebab_case=True
    )

    data_report: Path
    out: Path

    def cli_cmd(self) -> None:
        import pyarrow.parquet as pq

        report = json.loads(self.data_report.read_text())
        root = Path(os.environ["SOE_DATA"])
        rows = {
            split: pq.read_table(root / info["parquet"]).to_pylist()
            for split, info in report["files"].items()
        }
        chat_kwargs = load_suite(Path(report["suite"]["path"])).prompt.chat_template_kwargs()
        out = {
            "data_report": str(self.data_report),
            "parquet_sha256": {s: i["parquet_sha256"] for s, i in report["files"].items()},
            "chat_template_kwargs": chat_kwargs,
            "versions": {
                lib: metadata.version(lib) for lib in ("verl", "transformers", "tokenizers")
            },
            "models": {hf_id: check_model(hf_id, rows, chat_kwargs) for hf_id in MODELS},
        }
        self.out.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
        bad = sum(m[s]["mismatches"] for m in out["models"].values() for s in rows)
        if bad:
            raise SystemExit(f"{bad} prompts differ between verl and the evaluation sampler")


if __name__ == "__main__":
    CliApp.run(CheckCli)
