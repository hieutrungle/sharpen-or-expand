"""TRL SFT worker: one process per GPU, started by `soe.trainer.main_sft` with torchrun.

    torchrun --standalone --nproc_per_node=N -m soe.trainer.sft.trainer --run-dir runs/<id> \
        [--resume-from runs/<id>/trainer/checkpoint-<step>]

TRL 0.27's SFTTrainer on pre-tokenized examples (soe/trainer/sft/data.py), so the loss mask is ours
and TRL adds no EOS; Liger's fused cross-entropy keeps 18K-token sequences from materialising the
logits. Two training modes (arguments in soe/trainer/sft/args.py):

- full_bf16 (M2.6, decision 028): the model is loaded in FP32 (exact master weights, the base's F32
  tensors included), the vision tower frozen, and FSDP2 shards it over the GPUs with BF16 compute
  and FP32 gradient reduction. At the end the full FP32 state dict is gathered to rank 0 and
  written in the base layout.
- lora_bf16 (M1.8, decision 023): BF16 LoRA on the language model, merged in FP32 at the end.

Saves every `sft.save_every_steps` (sharded model, optimizer, scheduler, RNG, trainer state) let a
chained job resume; every training micro-batch logs its example ids (soe/trainer/sft/resume.py).
The export (soe/trainer/rl/export.py: vision tower unchanged, mtp copied, the base's config,
tokenizer and chat template) goes to `$SOE_DATA/checkpoints/<run_id>/step_<N>/hf`, which main_eval
and verl's `init_checkpoint` load. A resumed job whose training had finished only exports.

Image only (torch, transformers, peft, TRL).
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from safetensors import safe_open
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
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
from soe.trainer.config.experiment import ExperimentConfig, TrainingMode, load_config
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
from soe.trainer.sft.args import FSDP_ENVIRONMENT, plain_name, trainer_arguments
from soe.trainer.sft.data import SftExample, TokenizedExample, fitting, tokenize_example
from soe.trainer.sft.resume import TRAINER_DIR, example_key, log_path
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
    """Trainer logs as `{"step", "data"}` lines, the format of verl's file logger.

    Each log line also carries the wall time and rank 0's torch peak memory (M2.8 times steps
    and sizes run #1 from them), and each save adds a `save_seconds` line. The Trainer logs,
    evaluates (which logs too) and then saves, so the time since the last log is the save."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._last_log = time.time()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self._last_log = time.time()
        if state.is_world_process_zero and logs:
            self._write(state.global_step, {**logs, "wall_time": self._last_log, **peak_memory()})

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        if state.is_world_process_zero:
            self._write(state.global_step, {"save_seconds": time.time() - self._last_log})

    def _write(self, step: int, data: dict[str, float]) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps({"step": step, "data": data}) + "\n")


def peak_memory() -> dict[str, float]:
    """This rank's torch peak since the start (nvidia-smi in gpu_memory/ adds NCCL and the rest)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


class SoeSftTrainer(SFTTrainer):  # type: ignore[misc]
    """Three additions to TRL's trainer.

    Liger skips the logits only in training mode; in evaluation it would build the full sequence x
    vocabulary logits (12 GB for one 18K-token example on 2B, job 3223714 OOM). The loss and token
    accuracy come from Liger's fused kernel in both modes, so logits are never needed.

    Each training micro-batch appends the train-set indices of its examples to this rank's log, so
    the data order can be compared across a resume.

    After a validation, every FSDP2 module is set back to idle and resharded, as the end of a
    backward pass does in training. accelerate shards the tied `embed_tokens` and `lm_head` as
    their own units without resharding after forward; a forward without backward leaves them in
    the FORWARD state with their weights gathered, where `reshard()` is a no-op. The save that
    follows swapped the weights back to shards behind FSDP's back, so the next training forward
    skipped the gather and failed (job 3234134 at step 291: the embedding weight was a DTensor).
    Private torch state, pinned at 2.11; the 2-GPU container test covers it."""

    def __init__(self, *args: Any, example_index: dict[str, int], example_log: Path, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._example_index = example_index
        self._example_log = example_log
        example_log.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        metrics: dict[str, float] = super().evaluate(*args, **kwargs)
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                state._training_state = TrainingState.IDLE
                if state._fsdp_param_group is not None:
                    state._fsdp_param_group._training_state = TrainingState.IDLE
                module.reshard()
        return metrics

    def compute_loss(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if model.training:
            self._log_examples(inputs)
        return super().compute_loss(model, {**inputs, "skip_logits": True}, *args, **kwargs)

    def _log_examples(self, inputs: dict[str, Any]) -> None:
        ids, mask = inputs["input_ids"], inputs.get("attention_mask")
        rows = [ids[i] if mask is None else ids[i][mask[i].bool()] for i in range(ids.shape[0])]
        examples = [self._example_index[example_key(row.tolist())] for row in rows]
        record = {
            "step": self.state.global_step + 1,  # the optimizer step this batch counts toward
            "examples": examples,
            "tokens": sum(len(r) for r in rows),
            "loss_tokens": int((inputs["labels"] != -100).sum()),  # TRL's collator masks with -100
        }
        with open(self._example_log, "a") as f:
            f.write(json.dumps(record) + "\n")


def datasets(
    config: ExperimentConfig, tokenizer: Tokenizer, data_root: Path
) -> tuple[dict[str, Dataset], dict[str, dict[str, int]], list[str]]:
    """Tokenized train/val sets, their counts, and the key of every train example (its index is
    the example id the trainer logs)."""
    assert config.sft is not None
    out: dict[str, Dataset] = {}
    counts: dict[str, dict[str, int]] = {}
    train_keys: list[str] = []
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
        if split == "train":
            train_keys = [example_key(e.input_ids) for e in kept]
        out[split] = Dataset.from_dict(
            {
                "input_ids": [e.input_ids for e in kept],
                "completion_mask": [e.completion_mask for e in kept],
            }
        )
    return out, counts, train_keys


def restore_checkpoint_dtypes(model: Any, base: Path) -> list[str]:
    """Loading in BF16 rounds the tensors the checkpoint stores in F32 (Qwen3.5's 18
    linear-attention gate norms; job 3223729). Put their exact F32 values back, so the frozen base
    is the checkpoint itself. Returns the restored names."""
    params = dict(model.named_parameters())
    restored = []
    for path in sorted(base.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                if name in params and f.get_slice(name).get_dtype() == "F32":
                    param = params[name]
                    if param.dtype != torch.float32:
                        param.data = f.get_tensor(name).to(param.device)
                        restored.append(name)
    return restored


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
    layout = like_base(tensors, base, lora_modules=merged)
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


def export_full(
    state: dict[str, torch.Tensor],
    trained: list[str],
    base: Path,
    out_dir: Path,
    *,
    run_id: str,
    step: int,
    config: ExperimentConfig,
) -> ExportInfo:
    """The gathered FP32 state dict in the base layout; only language-model tensors may differ."""
    check_trained(trained, is_lora=False)
    tensors = {plain_name(k): v.detach().to("cpu", torch.float32) for k, v in state.items()}
    numels = {k: v.numel() for k, v in tensors.items()}
    layout = like_base(tensors, base, lora_modules=None)
    info = ExportInfo(
        run_id=run_id,
        step=step,
        rollout_tokens=None,
        source_checkpoint="trl-sft",
        base_model=f"{config.model.hf_id}@{config.model.revision}",
        training="full",
        lora_rank=None,
        lora_alpha=None,
        merged_lora_modules=0,
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


def freeze_vision(model: Any) -> int:
    """Decision 019's rule: only the language model trains. Returns the frozen tensor count."""
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("model.visual."):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def train(run_dir: Path, resume_from: Path | None) -> None:
    config = load_config(run_dir / "config.yaml")
    info = read_run_info(run_dir)
    sft, lora = config.sft, config.lora
    full = config.training_mode is TrainingMode.FULL_BF16
    if sft is None or (lora is None and not full):
        raise ValueError(f"{config.name} is not a full-BF16 or LoRA SFT config")
    data_root = Path(os.environ["SOE_DATA"])
    base = model_path(config, data_root)
    tokenizer = Tokenizer(base, load_suite(Path(config.eval_suite)).prompt)
    data, counts, train_keys = datasets(config, tokenizer, data_root)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        (run_dir / SFT_DATA).write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
        print(f"soe sft data: {counts}", flush=True)

    if full:
        # FP32 load: exact master weights (the checkpoint's F32 tensors stay F32, the BF16 ones are
        # exact in FP32), so no dtype restore is needed and the export writes the base's dtypes.
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32)
        frozen = freeze_vision(model)
        os.environ.update(FSDP_ENVIRONMENT)  # before the Trainer builds accelerate's FSDP plugin
        if rank == 0:
            print(f"soe sft: full BF16, {frozen} vision tensors frozen", flush=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
        restored = restore_checkpoint_dtypes(model, base)
        if rank == 0:
            print(f"soe sft: {len(restored)} F32 checkpoint tensors kept in F32", flush=True)
    args = SFTConfig(**trainer_arguments(config, seed=info.seed, output_dir=run_dir / TRAINER_DIR))
    segment = os.environ.get("SLURM_JOB_ID", "local")
    metrics = run_dir / "metrics" / f"segment-{segment}.jsonl"
    metrics.parent.mkdir(exist_ok=True)
    peft_config = None
    if lora is not None:
        peft_config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            target_modules=lora.target_modules,
            exclude_modules=lora.exclude_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    trainer = SoeSftTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer.hf,
        peft_config=peft_config,
        callbacks=[MetricsFile(metrics)],
        example_index={key: i for i, key in enumerate(train_keys)},
        example_log=log_path(run_dir, segment, rank),
    )
    if full:
        # D6: BF16 compute, FP32 gradient reduction. accelerate's bf16 default reduces in BF16, and
        # TrainingArguments has no field for it; the plugin is read when train() wraps the model.
        trainer.accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16
        )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)

    step = trainer.state.global_step
    out_dir = export_dir(data_root, info.run_id, step)
    main = trainer.accelerator.is_main_process
    if full:
        state = trainer.accelerator.get_state_dict(trainer.model)  # a collective: every rank
        if main and not out_dir.exists():
            trained = [
                plain_name(n) for n, p in trainer.model.named_parameters() if p.requires_grad
            ]
            exported = export_full(
                state, trained, base, out_dir, run_id=info.run_id, step=step, config=config
            )
            print(f"soe sft: exported step {step}: {exported.model_dump_json()}", flush=True)
    elif main and not out_dir.exists():
        exported = export_merged(
            trainer.accelerator.unwrap_model(trainer.model),
            base,
            out_dir,
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
    resume_from: Path | None = None  # a complete trainer checkpoint (main_sft picks it)

    def cli_cmd(self) -> None:
        train(self.run_dir.resolve(), self.resume_from)


if __name__ == "__main__":
    CliApp.run(SftWorkerCli)
