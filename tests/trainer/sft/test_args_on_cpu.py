"""SFTConfig arguments: full BF16 shards with FSDP2 and checkpoints for resume; LoRA as in M1.8."""

import json
import math
from pathlib import Path

import pytest

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


def test_run2_differs_from_run1_only_by_decision_029s_changes() -> None:
    run1 = load_config(Path("examples/sft/m2_run1_qwen35-4b.yaml"))
    run2 = load_config(Path("examples/sft/m2_run2_qwen35-4b.yaml"))
    a1 = trainer_arguments(run1, seed=1, output_dir=Path("/runs/r/trainer"))
    a2 = trainer_arguments(run2, seed=1, output_dir=Path("/runs/r/trainer"))
    changed = {k for k in a1 if a1[k] != a2[k]}
    assert changed == {"learning_rate", "warmup_steps"}  # the trainer arguments
    assert a2["learning_rate"] == pytest.approx(1e-5 / 3, rel=1e-6)
    data = json.loads(Path("docs/reports/m2_sft_run2_data.json").read_text())
    assert (run2.data.train_file, run2.data.train_sha256) == (data["output"], data["output_sha256"])
    assert run2.data.val_file == run1.data.val_file and run2.data.val_sha256 == run1.data.val_sha256
    sft = run2.sft
    assert sft is not None
    steps = math.ceil(
        math.ceil(data["examples"]["kept"] / sft.n_gpus) / sft.gradient_accumulation_steps
    )
    assert steps == 719 and a2["warmup_steps"] == math.ceil(0.03 * steps)
    others = {"name", "data", "optim"}
    dump1, dump2 = run1.model_dump(), run2.model_dump()
    assert {k for k in dump1 if dump1[k] != dump2[k]} == others


def test_lora_keeps_the_m18_wiring() -> None:
    config = load_config(Path("examples/sft/m1_dev_qwen35-2b.yaml"))
    args = trainer_arguments(config, seed=1, output_dir=Path("/runs/r/trainer"))
    assert "fsdp" not in args and args["gradient_checkpointing"]
    assert args["save_strategy"] == "no" and args["max_steps"] == -1
    assert args["eval_strategy"] == "epoch"  # no saves: one validation at the end


def test_plain_name_drops_the_checkpoint_wrapper() -> None:
    wrapped = "model.language_model.layers.3._checkpoint_wrapped_module.mlp.up_proj.weight"
    assert plain_name(wrapped) == "model.language_model.layers.3.mlp.up_proj.weight"
