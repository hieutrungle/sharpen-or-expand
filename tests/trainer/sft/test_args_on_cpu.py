"""SFTConfig arguments: full BF16 shards with FSDP2 and checkpoints for resume; LoRA as in M1.8."""

import json
import math
from pathlib import Path

from soe.trainer.config.experiment import load_config
from soe.trainer.sft.args import DECODER_LAYERS, plain_name, trainer_arguments


def test_full_bf16_uses_fsdp2_with_native_activation_checkpointing() -> None:
    config = load_config(Path("examples/sft/m2_dev_full_qwen35-2b.yaml"))
    args = trainer_arguments(config, seed=3, output_dir=Path("/runs/r/trainer"))
    assert args["fsdp"] == "full_shard auto_wrap"
    fsdp = args["fsdp_config"]
    assert fsdp["version"] == 2 and fsdp["transformer_layer_cls_to_wrap"] == DECODER_LAYERS
    assert fsdp["activation_checkpointing"] and not args["gradient_checkpointing"]
    sft = config.sft
    assert sft is not None and sft.max_steps is not None
    saves = ("steps", sft.save_every_steps, sft.keep_last)
    assert (args["save_strategy"], args["save_steps"], args["save_total_limit"]) == saves
    assert not args["save_only_model"]  # optimizer, scheduler and RNG state too
    assert (args["max_steps"], args["seed"], args["data_seed"]) == (sft.max_steps, 3, 3)
    assert args["bf16"] and args["use_liger_kernel"] and not args["packing"]
    assert args["gradient_accumulation_steps"] * sft.n_gpus == 32


def test_run1_is_decision_029s_recipe() -> None:
    config = load_config(Path("examples/sft/m2_run1_qwen35-4b.yaml"))
    args = trainer_arguments(config, seed=1, output_dir=Path("/runs/r/trainer"))
    sft = config.sft
    assert sft is not None and not config.development and sft.max_steps is None
    examples = json.loads(Path("docs/reports/m2_sft_data.json").read_text())["files"]["train"]
    steps = math.ceil(
        math.ceil(examples["examples"] / sft.n_gpus) / sft.gradient_accumulation_steps
    )
    assert steps == 799 and args["warmup_steps"] == math.ceil(0.03 * steps)
    assert (args["learning_rate"], args["lr_scheduler_type"], args["weight_decay"]) == (
        1e-5,
        "cosine",
        0.0,
    )
    assert (args["optim"], args["adam_beta1"], args["adam_beta2"], args["adam_epsilon"]) == (
        "adamw_torch_fused",
        0.9,
        0.999,
        1e-8,
    )
    assert args["max_grad_norm"] == 1.0 and args["num_train_epochs"] == 1
    # Validations at 290 and 580, and the Trainer's at the last step 799: criterion 1's three.
    assert (args["eval_strategy"], args["eval_steps"], args["save_steps"]) == ("steps", 290, 290)


def test_lora_keeps_the_m18_wiring() -> None:
    config = load_config(Path("examples/sft/m1_dev_qwen35-2b.yaml"))
    args = trainer_arguments(config, seed=1, output_dir=Path("/runs/r/trainer"))
    assert "fsdp" not in args and args["gradient_checkpointing"]
    assert args["save_strategy"] == "no" and args["max_steps"] == -1
    assert args["eval_strategy"] == "epoch"  # no saves: one validation at the end


def test_plain_name_drops_the_checkpoint_wrapper() -> None:
    wrapped = "model.language_model.layers.3._checkpoint_wrapped_module.mlp.up_proj.weight"
    assert plain_name(wrapped) == "model.language_model.layers.3.mlp.up_proj.weight"
