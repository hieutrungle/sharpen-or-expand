from soe.trainer.rl.verl_patches import is_vision_parameter


def test_vision_parameters_are_recognised_with_and_without_peft_names() -> None:
    assert is_vision_parameter("model.visual.blocks.0.attn.qkv.weight")  # full training
    assert is_vision_parameter("base_model.model.model.visual.patch_embed.proj.bias")  # LoRA
    assert not is_vision_parameter("model.language_model.layers.0.mlp.up_proj.weight")
    assert not is_vision_parameter("lm_head.weight")
