"""SFTConfig arguments: full BF16 shards with FSDP2 and checkpoints for resume; LoRA as in M1.8."""

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


def test_lora_keeps_the_m18_wiring() -> None:
    config = load_config(Path("examples/sft/m1_dev_qwen35-2b.yaml"))
    args = trainer_arguments(config, seed=1, output_dir=Path("/runs/r/trainer"))
    assert "fsdp" not in args and args["gradient_checkpointing"]
    assert args["save_strategy"] == "no" and args["max_steps"] == -1


def test_plain_name_drops_the_checkpoint_wrapper() -> None:
    wrapped = "model.language_model.layers.3._checkpoint_wrapped_module.mlp.up_proj.weight"
    assert plain_name(wrapped) == "model.language_model.layers.3.mlp.up_proj.weight"
