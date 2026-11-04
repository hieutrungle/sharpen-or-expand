"""Full-BF16 SFT under FSDP2 on CPU with gloo (decision 028; M2.6 acceptance): a tiny
Qwen3.5-shaped model trains 4 steps with saves at 2 and 4, a second launch resumes from step 2
with the same example order and learning rate, and the export changes only language-model tensors.
Image only (torch, transformers, TRL); needs the staged 2B snapshot for its config."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from soe.trainer.config.experiment import load_config
from soe.trainer.main_rl import model_path
from soe.trainer.sft.resume import continuity

WORKER = Path(__file__).with_name("fsdp_tiny_worker.py")


def tiny_base(out: Path) -> None:
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    config = load_config(Path("examples/sft/m2_dev_full_qwen35-2b.yaml"))
    snapshot = model_path(config, Path(os.environ["SOE_DATA"]))
    AutoTokenizer.from_pretrained(snapshot).save_pretrained(out)  # the trainer's processing_class
    hf = AutoConfig.from_pretrained(snapshot)
    text, vision = hf.text_config, hf.vision_config
    text.num_hidden_layers, text.layer_types = 2, ["linear_attention", "full_attention"]
    text.hidden_size, text.intermediate_size = 64, 128
    text.num_attention_heads, text.num_key_value_heads = 1, 1
    text.linear_num_key_heads, text.linear_num_value_heads = 1, 1
    vision.depth, vision.hidden_size, vision.intermediate_size = 1, 32, 64
    vision.num_heads, vision.out_hidden_size = 2, 64
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(hf, dtype=torch.float32).to(
        torch.bfloat16
    ).save_pretrained(out)


def launch(tmp: Path, *extra: str) -> None:
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    subprocess.run([*command, str(WORKER), str(tmp), *extra], check=True, timeout=1200)


@pytest.mark.skipif("SOE_DATA" not in os.environ, reason="needs the staged 2B snapshot")
def test_fsdp2_trains_resumes_and_exports_on_cpu(tmp_path: Path) -> None:
    tiny_base(tmp_path / "base")
    launch(tmp_path)
    launch(tmp_path, "resume")
    first, second = (
        json.loads((tmp_path / f"result-{s}.json").read_text()) for s in ("first", "resume")
    )
    assert any(name.startswith("FSDP") for name in first["modules"])  # FSDP2 wrapped the layers
    for result in (first, second):
        export = result["export"]
        assert export["training"] == "full" and export["tensors_changed_vs_base"] > 0
        assert all(
            p.startswith(("model.language_model", "lm_head")) for p in export["trained_by_prefix"]
        )
    gaps = continuity(tmp_path / "run")
    assert gaps.redone_steps == [3, 4] and gaps.mismatched_steps == []
    assert sorted(second["steps"]) == ["1", "2", "3", "4"]  # steps 1-2 come from the saved state
    for step in ("3", "4"):
        before, after = first["steps"][step], second["steps"][step]
        assert after["lr"] == pytest.approx(before["lr"], rel=1e-9)
        assert after["loss"] == pytest.approx(before["loss"], rel=1e-3)
