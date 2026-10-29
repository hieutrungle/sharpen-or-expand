import subprocess
import sys

from soe.workers.rollout.verl_patches import is_vision_parameter

# install() runs in every Ray worker before its GPU is assigned: importing torch or verl there put
# both FSDP ranks on one GPU (job 3222890). It may only register the import hook.
INSTALL_WITHOUT_TORCH = """
import sys
from soe.workers.rollout import verl_patches
verl_patches.install()
verl_patches.install()
hooks = [f for f in sys.meta_path if type(f).__name__ == "_PatchAfterImport"]
assert len(hooks) == 1, hooks
assert "torch" not in sys.modules and "verl" not in sys.modules
assert hooks[0].find_spec("json", None) is None
print("ok")
"""


def test_install_only_registers_an_import_hook() -> None:
    out = subprocess.run(
        [sys.executable, "-c", INSTALL_WITHOUT_TORCH], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok"


def test_vision_parameters_are_recognised_with_and_without_peft_names() -> None:
    assert is_vision_parameter("model.visual.blocks.0.attn.qkv.weight")  # full training
    assert is_vision_parameter("base_model.model.model.visual.patch_embed.proj.bias")  # LoRA
    assert not is_vision_parameter("model.language_model.layers.0.mlp.up_proj.weight")
    assert not is_vision_parameter("lm_head.weight")
