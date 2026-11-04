"""The full-BF16 SFT path (decision 028; M2.6 acceptance) on a tiny Qwen3.5-shaped model: 4 steps
with saves at 2 and 4, a second launch resumed from step 2 with the same examples, learning rate and
loss, and an export that changes only language-model tensors.

- On CPU (every container run): one process without FSDP, because accelerate enables FSDP only for
  multi-accelerator setups. This covers the FP32 load, the frozen vision tower, `export_full`
  through `like_base`, the example-id log and the trainer's resume.
- On 2 GPUs (M2.7): the exact path, FSDP2 over two ranks with Liger and the fla kernels.

Image only (torch, transformers, TRL); needs the staged 2B snapshot for its config and tokenizer.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from soe.trainer.config.experiment import load_config
from soe.trainer.main_rl import model_path
from soe.trainer.sft.resume import continuity

WORKER = Path(__file__).with_name("fsdp_tiny_worker.py")
needs_data = pytest.mark.skipif("SOE_DATA" not in os.environ, reason="needs the staged 2B snapshot")


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
    model = AutoModelForImageTextToText.from_config(hf, dtype=torch.float32)
    model.to(torch.bfloat16).save_pretrained(out)


def check_logs(run_dir: Path, lengths: range) -> None:
    """M2.8's measurements: tokens and loss tokens per micro-batch, wall time and save time."""
    for path in (run_dir / "examples").glob("*.jsonl"):
        for row in map(json.loads, path.read_text().splitlines()):
            sizes = [lengths[i] for i in row["examples"]]  # the worker's i-th example
            assert row["tokens"] == sum(sizes)
            assert row["loss_tokens"] == sum(n - n // 2 for n in sizes)  # the completion half
    lines = [json.loads(x) for x in (run_dir / "metrics" / "segment-first.jsonl").open()]
    assert all("wall_time" in x["data"] for x in lines if "loss" in x["data"])
    saves = [x["step"] for x in lines if "save_seconds" in x["data"]]
    assert saves == [2, 4] and all(x["data"].get("save_seconds", 0) >= 0 for x in lines)


def run_twice(tmp: Path, fsdp: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    tiny_base(tmp / "base")
    if fsdp:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
        ]
    else:
        command = [sys.executable]
    # Without FSDP the variant is the CPU one: on a GPU node one process would use DataParallel.
    env = os.environ | ({} if fsdp else {"CUDA_VISIBLE_DEVICES": ""})
    for launch in ("first", "resume"):
        extra = [] if fsdp else ["--no-fsdp"]
        subprocess.run(
            [*command, str(WORKER), str(tmp), launch, *extra], check=True, timeout=1200, env=env
        )
    first, second = (
        json.loads((tmp / f"result-{s}.json").read_text()) for s in ("first", "resume")
    )
    for result in (first, second):
        export = result["export"]
        assert export["training"] == "full" and export["tensors_changed_vs_base"] > 0
        prefixes = export["trained_by_prefix"]
        assert prefixes and all(p.startswith(("model.language_model", "lm_head")) for p in prefixes)
    check_logs(tmp / "run", lengths=range(20, 36, 2) if fsdp else range(20, 28, 2))
    gaps = continuity(tmp / "run")
    assert gaps.redone_steps == [3, 4] and gaps.mismatched_steps == []
    assert sorted(second["steps"]) == ["1", "2", "3", "4"]  # steps 1-2 come from the saved state
    for step in ("3", "4"):
        before, after = first["steps"][step], second["steps"][step]
        assert after["lr"] == pytest.approx(before["lr"], rel=1e-9)
        assert after["loss"] == pytest.approx(before["loss"], rel=1e-3)
    return first, second


@needs_data
def test_full_sft_path_resumes_and_exports_on_cpu(tmp_path: Path) -> None:
    run_twice(tmp_path, fsdp=False)


@needs_data
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="FSDP2 needs 2 GPUs (M2.7)")
def test_fsdp2_resumes_and_exports_on_two_gpus(tmp_path: Path) -> None:
    first, _ = run_twice(tmp_path, fsdp=True)
    assert any(name.startswith("FSDP") for name in first["modules"])  # decoder layers sharded
