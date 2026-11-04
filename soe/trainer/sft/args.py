"""TRL `SFTConfig` arguments for an SFT config (decisions 023, 028), kept apart from the worker so
the CPU tests check them without torch or TRL.

- full_bf16: FSDP2 through accelerate (`fsdp="full_shard auto_wrap"`, version 2), Qwen3.5 decoder
  layers as wrap units, FSDP-native activation checkpointing (the Trainer flag would add an
  all-gather per layer in backward), sharded checkpoints (`FSDP_STATE_DICT_TYPE`, which
  TrainingArguments does not forward to accelerate's plugin).
- lora_bf16: DDP with the Trainer's gradient checkpointing, as in M1.8.
"""

from pathlib import Path
from typing import Any

from soe.trainer.config.experiment import ExperimentConfig, TrainingMode

DECODER_LAYERS = ["Qwen3_5DecoderLayer"]  # the vision tower (frozen) stays in the root unit
FSDP_ENVIRONMENT = {"FSDP_STATE_DICT_TYPE": "SHARDED_STATE_DICT"}  # read by accelerate's plugin
CHECKPOINT_WRAPPER = "_checkpoint_wrapped_module."  # torch's activation-checkpoint wrapper


def plain_name(name: str) -> str:
    """A parameter name without the activation-checkpoint wrapper's prefix, as in the state dict."""
    return name.replace(CHECKPOINT_WRAPPER, "")


def trainer_arguments(config: ExperimentConfig, *, seed: int, output_dir: Path) -> dict[str, Any]:
    sft = config.sft
    if sft is None:
        raise ValueError(f"{config.name} is not an SFT config")
    full = config.training_mode is TrainingMode.FULL_BF16
    saves = sft.save_every_steps is not None
    args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": sft.epochs,
        "max_steps": sft.max_steps if sft.max_steps is not None else -1,
        "per_device_train_batch_size": sft.batch_size_per_gpu,
        "per_device_eval_batch_size": sft.batch_size_per_gpu,
        "gradient_accumulation_steps": sft.gradient_accumulation_steps,
        "learning_rate": config.optim.learning_rate,
        "warmup_steps": config.optim.warmup_steps,
        "weight_decay": config.optim.weight_decay,
        "max_grad_norm": 1.0,  # D5's clip; the Trainer default, written out
        # The Trainer's AdamW under torch >= 2.8, written out so a library default cannot drift.
        "optim": "adamw_torch_fused",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "lr_scheduler_type": sft.lr_scheduler,
        "bf16": True,
        "use_liger_kernel": True,
        "max_length": None,  # examples are already filtered to sft.max_length
        "packing": False,
        "completion_only_loss": True,
        "logging_steps": 1,
        # Validation at every save, plus the Trainer's one at the last step: run #1 validates at
        # 290, 580 and 799, the three points of decision 029's criterion 1. Runs without saves
        # (development) validate once, at the end of the epoch.
        "eval_strategy": "steps" if saves else "epoch",
        "eval_steps": sft.save_every_steps if saves else None,
        "prediction_loss_only": True,  # validation loss only; no logits are gathered
        "save_strategy": "steps" if saves else "no",
        "save_steps": sft.save_every_steps if saves else 500,  # 500: TRL's default, unused
        "save_total_limit": sft.keep_last if saves else None,
        "save_only_model": False,  # optimizer, scheduler, RNG and trainer state, for resume
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
    }
    if full:
        args |= {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "version": 2,
                "transformer_layer_cls_to_wrap": DECODER_LAYERS,
                "activation_checkpointing": True,
                "reshard_after_forward": True,
            },
            "gradient_checkpointing": False,
        }
    else:
        args |= {
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "ddp_find_unused_parameters": False,
        }
    return args
