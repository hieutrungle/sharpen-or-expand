"""TRL SFT worker (M1.8): one process per GPU, started by `soe.trainer.main_sft` with torchrun.

    torchrun --standalone --nproc_per_node=N -m soe.trainer.sft.trainer --run-dir runs/<id>

Trains BF16 LoRA (language model only; the config's lora block, as in RL) with TRL 0.27's
SFTTrainer on pre-tokenized examples (soe/trainer/sft/data.py), so the loss mask is ours and TRL
adds no EOS. Liger's fused cross-entropy keeps 18K-token sequences from materialising the logits.
Rank 0 then merges the adapters in FP32 and writes the checkpoint in the base snapshot's layout
(soe/trainer/rl/export.py: vision tower and mtp unchanged, the base's config, tokenizer and chat
template copied) to `$SOE_DATA/checkpoints/<run_id>/step_<N>/hf`, which main_eval and verl's
`init_checkpoint` load.

Image only (torch, transformers, peft, TRL).
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from soe.trainer.config.eval_suite import PromptSpec, load_suite
from soe.trainer.config.experiment import ExperimentConfig, load_config
from soe.trainer.main_rl import checked_data_file, model_path
from soe.trainer.rl.checkpoints import export_dir
from soe.trainer.rl.export import (
    ExportInfo,
    check_trained,
    like_base,
    merge_lora,
    trained_by_prefix,
    write_export,
)
from soe.trainer.sft.data import SftExample, TokenizedExample, fitting, tokenize_example
from soe.utils.jsonl import read_jsonl
from soe.utils.run import read_run_info
from soe.workers.rollout.chat import turn_end_text

SFT_DATA = "sft_data.json"  # per-split example counts, written by rank 0


class Tokenizer:
    """The base model's tokenizer, rendering prompts exactly as evaluation does."""

    def __init__(self, base: Path, prompt: PromptSpec) -> None:
        self.hf: Any = AutoTokenizer.from_pretrained(base)
        self._prompt = prompt
        turn_end = turn_end_text(lambda m: self.hf.apply_chat_template(m, tokenize=False))
        self.turn_end_id = int(self.encode(turn_end)[0])  # <|im_end|>, as the agent loop takes it

    def encode(self, text: str) -> list[int]:
        return list(self.hf.encode(text, add_special_tokens=False))

    def render(self, user: str) -> str:
        rendered: str = self.hf.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **self._prompt.chat_template_kwargs(),
        )
        return rendered

    def tokenize(self, example: SftExample) -> TokenizedExample:
        return tokenize_example(
            self.render(example.user),
            example.completion,
            encode=self.encode,
            turn_end_id=self.turn_end_id,
        )


class MetricsFile(TrainerCallback):  # type: ignore[misc]
    """Trainer logs as `{"step", "data"}` lines, the format of verl's file logger."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero and logs:
            with open(self._path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "data": logs}) + "\n")


def datasets(
    config: ExperimentConfig, tokenizer: Tokenizer, data_root: Path
) -> tuple[dict[str, Dataset], dict[str, dict[str, int]]]:
    assert config.sft is not None
    out: dict[str, Dataset] = {}
    counts: dict[str, dict[str, int]] = {}
    files = {
        "train": (config.data.train_file, config.data.train_sha256),
        "val": (config.data.val_file, config.data.val_sha256),
    }
    for split, (relative, sha256) in files.items():
        examples = read_jsonl(checked_data_file(data_root, relative, sha256), SftExample)
        kept, dropped = fitting([tokenizer.tokenize(e) for e in examples], config.sft.max_length)
        counts[split] = {
            "examples": len(examples),
            "dropped_too_long": dropped,
            "tokens": sum(len(e.input_ids) for e in kept),
            "loss_tokens": sum(sum(e.completion_mask) for e in kept),
        }
        out[split] = Dataset.from_dict(
            {
                "input_ids": [e.input_ids for e in kept],
                "completion_mask": [e.completion_mask for e in kept],
            }
        )
    return out, counts


def export_merged(
    model: Any, base: Path, out_dir: Path, *, run_id: str, step: int, config: ExperimentConfig
) -> ExportInfo:
    """FP32 merge of the trained adapters into the base layout (decision 018's export rules)."""
    assert config.lora is not None
    trained = [name for name, p in model.named_parameters() if p.requires_grad]
    check_trained(trained, is_lora=True)  # language-model adapters only
    full = {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()}
    numels = {k: v.numel() for k, v in full.items()}
    tensors, merged = merge_lora(full, config.lora.alpha / config.lora.rank)
    del full
    layout = like_base(tensors, base)
    info = ExportInfo(
        run_id=run_id,
        step=step,
        rollout_tokens=None,
        source_checkpoint="trl-sft",
        base_model=f"{config.model.hf_id}@{config.model.revision}",
        training="lora",
        lora_rank=config.lora.rank,
        lora_alpha=config.lora.alpha,
        merged_lora_modules=merged,
        trained_tensors=len(trained),
        trained_parameters=sum(numels[k] for k in trained),
        trained_by_prefix=trained_by_prefix(trained, numels),
        tensors_changed_vs_base=layout.changed,
        copied_from_base=layout.copied,
        created_at=datetime.now(UTC),
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    write_export(layout, base, out_dir, info)
    return info


def train(run_dir: Path) -> None:
    config = load_config(run_dir / "config.yaml")
    info = read_run_info(run_dir)
    sft, lora = config.sft, config.lora
    if sft is None or lora is None:
        raise ValueError(f"{config.name} is not a LoRA SFT config")
    data_root = Path(os.environ["SOE_DATA"])
    base = model_path(config, data_root)
    tokenizer = Tokenizer(base, load_suite(Path(config.eval_suite)).prompt)
    data, counts = datasets(config, tokenizer, data_root)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    if rank0:
        (run_dir / SFT_DATA).write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
        print(f"soe sft data: {counts}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=sft.epochs,
        per_device_train_batch_size=sft.batch_size_per_gpu,
        per_device_eval_batch_size=sft.batch_size_per_gpu,
        gradient_accumulation_steps=sft.gradient_accumulation_steps,
        learning_rate=config.optim.learning_rate,
        warmup_steps=config.optim.warmup_steps,
        weight_decay=config.optim.weight_decay,
        lr_scheduler_type=sft.lr_scheduler,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        max_length=None,  # examples are already filtered to sft.max_length
        packing=False,
        completion_only_loss=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",  # the merged checkpoint is written below; SFT resume arrives at M2
        report_to=[],
        seed=info.seed,
        data_seed=info.seed,
        ddp_find_unused_parameters=False,
    )
    segment = os.environ.get("SLURM_JOB_ID", "local")
    metrics = run_dir / "metrics" / f"segment-{segment}.jsonl"
    metrics.parent.mkdir(exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer.hf,
        peft_config=LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            target_modules=lora.target_modules,
            exclude_modules=lora.exclude_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        callbacks=[MetricsFile(metrics)],
    )
    trainer.train()
    if trainer.accelerator.is_main_process:
        step = trainer.state.global_step
        exported = export_merged(
            trainer.accelerator.unwrap_model(trainer.model),
            base,
            export_dir(data_root, info.run_id, step),
            run_id=info.run_id,
            step=step,
            config=config,
        )
        print(f"soe sft: exported step {step}: {exported.model_dump_json()}", flush=True)
    trainer.accelerator.wait_for_everyone()


class SftWorkerCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.sft.trainer", cli_kebab_case=True
    )

    run_dir: Path

    def cli_cmd(self) -> None:
        train(self.run_dir.resolve())


if __name__ == "__main__":
    CliApp.run(SftWorkerCli)
