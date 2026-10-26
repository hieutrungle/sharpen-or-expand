"""GPU stack check, run inside the container on a compute node (slurm/check_container.sbatch).

Checks: imports and versions of the pinned stack, CUDA and BF16 on every visible GPU, offline
loading of each staged model's config and tokenizer, and optionally a few vLLM tokens.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml

STACK = ("torch", "vllm", "verl", "transformers", "trl", "peft", "flash_attn", "fla", "ray")


def check_imports() -> None:
    for name in STACK:
        module = importlib.import_module(name)
        print(f"ok import {name} {getattr(module, '__version__', '?')}")


def check_gpus() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA not available (missing --nv?)"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gib = props.total_memory / 2**30
        print(f"ok gpu {i}: {props.name}, {memory_gib:.0f} GiB, sm_{props.major}{props.minor}")
    assert torch.cuda.is_bf16_supported(), "BF16 not supported"
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    print(f"ok bf16 matmul, norm {float((x @ x).norm()):.1f}")


def check_flash_attn() -> None:
    # Importing is not enough: run a kernel so a build lacking this GPU's architecture fails here.
    import torch
    from flash_attn import flash_attn_func

    q, k, v = (torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = flash_attn_func(q, k, v, causal=True)
    print(f"ok flash-attn kernel, output {tuple(out.shape)}")


def check_model_offline(hf_id: str, revision: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be 1 on compute nodes"
    config = AutoConfig.from_pretrained(hf_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    print(f"ok offline {hf_id}: {config.model_type}, vocab {len(tokenizer)}")


def check_vllm_generate(hf_id: str, revision: str) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model=hf_id, revision=revision, max_model_len=1024, gpu_memory_utilization=0.5)
    out = llm.generate(["1 + 1 ="], SamplingParams(max_tokens=8, temperature=0.0))
    print(f"ok vllm {hf_id}: {out[0].outputs[0].text!r}")


def staged_models(assets_path: Path) -> dict[str, str]:
    assets = yaml.safe_load(assets_path.read_text())["assets"]
    return {a["hf_id"]: a["revision"] for a in assets if a["repo_type"] == "model"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("scripts/stage_assets.yaml"))
    parser.add_argument(
        "--vllm", metavar="HF_ID", help="also generate a few tokens with this model"
    )
    args = parser.parse_args()
    models = staged_models(args.assets)
    check_imports()
    check_gpus()
    check_flash_attn()
    for hf_id, revision in models.items():
        check_model_offline(hf_id, revision)
    if args.vllm:
        check_vllm_generate(args.vllm, models[args.vllm])
    print("check_container: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
