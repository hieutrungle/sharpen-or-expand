"""M2.11 base-start audit: are verl's rollout prompt ids the evaluation sampler's? (container only)

    M=$SOE_DATA/hf/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
    R=$SOE_DATA/datasets/soe/v3.1/rl
    sbatch slurm/cpu.sbatch python scripts/prompt_identity.py --model-dir $M \\
        --split val:$R/verl/val.parquet:$R/val.jsonl \\
        --split train:$R/verl/train.parquet:$R/train.jsonl \\
        --report docs/reports/m2_prompt_identity.json

verl 0.9.1 tokenizes rollout prompts in its agent loop with a continuous-token builder chosen from
the model type and processor. For Qwen3.5 that is the VL builder over the processor, whose chat
template verl's HFModelConfig copies from the tokenizer when the processor has none (base models);
this script does the same. For every parquet row it compares the builder's ids with the evaluation
sampler's rendering (soe.workers.rollout.vllm_sampler: the suite prompt, the chat template with
the generation prompt and thinking mode, encoded without special tokens) of the same problem, and
checks the prompt ends with the thinking prefill (decision 006). Chooses nothing.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from transformers import AutoConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl

PREFILL = "<|im_start|>assistant\n<think>\n"


class PromptIdentityCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/prompt_identity.py", cli_kebab_case=True
    )

    model_dir: Path
    suite: Path = Path("soe/trainer/config/eval_suite.yaml")
    split: list[str]  # "name:verl parquet:Problem jsonl"
    report: Path

    def cli_cmd(self) -> None:
        prompt = load_suite(self.suite).prompt
        kwargs = prompt.chat_template_kwargs()
        tokenizer, processor = hf_tokenizer(str(self.model_dir)), hf_processor(str(self.model_dir))
        if processor is not None and not getattr(processor, "chat_template", None):
            processor.chat_template = tokenizer.chat_template  # as verl's HFModelConfig
        builder = create_continuous_token_builder(
            tokenizer,
            hf_model_type=AutoConfig.from_pretrained(str(self.model_dir)).model_type,
            chat_template_kwargs=kwargs,
            mm_processor_kwargs={},
            processor=processor,
        )
        report: dict[str, Any] = {
            "model_dir": str(self.model_dir),
            "chat_template_kwargs": kwargs,
            "builder": type(builder).__name__,
            "processor": type(processor).__name__,
        }
        for item in self.split:
            name, parquet, problems_file = item.split(":")
            problems = {p.problem_id: p for p in read_jsonl(Path(problems_file), Problem)}
            counts = {"rows": 0, "ids_equal": 0, "user_text_equal": 0, "ends_with_prefill": 0}
            first_mismatch: str | None = None
            for _, row in pd.read_parquet(parquet).iterrows():
                problem = problems[row["extra_info"]["problem_id"]]
                messages = [dict(m) for m in row["prompt"]]
                user = [{"role": "user", "content": prompt.render(problem.question)}]
                text = tokenizer.apply_chat_template(
                    user, tokenize=False, add_generation_prompt=True, **kwargs
                )
                rollout_ids = list(builder.build_initial_tokens(messages))
                equal = rollout_ids == tokenizer.encode(text, add_special_tokens=False)
                counts["rows"] += 1
                counts["ids_equal"] += equal
                counts["user_text_equal"] += messages == user
                counts["ends_with_prefill"] += text.endswith(PREFILL)
                if not equal and first_mismatch is None:
                    first_mismatch = problem.problem_id
            report[name] = counts | {"first_mismatch": first_mismatch}
            print(name, json.dumps(report[name]))
        self.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    CliApp.run(PromptIdentityCli)
