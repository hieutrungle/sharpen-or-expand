"""GPU stack check, run inside the container on a compute node (slurm/check_container.sbatch).

Checks: imports and versions of the pinned stack, CUDA and BF16 on every visible GPU, offline
loading of each staged model's config and tokenizer, and optionally a few vLLM tokens.
"""

import argparse
import importlib
import json
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
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be 1 on compute nodes"
    # vLLM resolves models with snapshot_download, which offline rejects a partial snapshot.
    snapshot_download(hf_id, revision=revision, local_files_only=True)
    config = AutoConfig.from_pretrained(hf_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    print(f"ok offline {hf_id}: {config.model_type}, vocab {len(tokenizer)}")


def check_vllm_generate(hf_id: str, revision: str) -> None:
    from vllm import LLM, SamplingParams

    # Qwen3.5 and Gemma 4 are multimodal checkpoints; the study is text-only. Without this, vLLM
    # profiles the vision encoder on a dummy video at startup (out of memory on a 16 GB GPU).
    llm = LLM(
        model=hf_id,
        revision=revision,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        language_model_only=True,
    )
    out = llm.generate(["1 + 1 ="], SamplingParams(max_tokens=8, temperature=0.0))
    print(f"ok vllm {hf_id}: {out[0].outputs[0].text!r}")


def pinned_models(assets_path: Path) -> dict[str, str]:
    assets = yaml.safe_load(assets_path.read_text())["assets"]
    return {a["hf_id"]: a["revision"] for a in assets if a["repo_type"] == "model"}


def staged_commits(data_root: Path) -> dict[str, str]:
    """hf_id -> commit actually downloaded, from the manifest written by stage_offline.py."""
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    assets = json.loads(manifest_path.read_text())["assets"]
    return {hf_id: entry["commit_sha"] for hf_id, entry in assets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("scripts/stage_assets.yaml"))
    parser.add_argument(
        "--vllm", metavar="HF_ID", help="also generate a few tokens with this model"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip pinned models that are not staged (local development); FASTER jobs omit this",
    )
    args = parser.parse_args()
    models = pinned_models(args.assets)
    staged = staged_commits(Path(os.environ["SOE_DATA"]))
    wrong = {m: staged[m] for m in models if m in staged and staged[m] != models[m]}
    if wrong:
        raise SystemExit(f"staged at a different revision than scripts/stage_assets.yaml: {wrong}")
    missing = [m for m in models if m not in staged]
    if missing and not args.allow_missing:
        raise SystemExit(
            f"not staged: {missing} (run scripts/stage_offline.py, see docs/cluster.md)"
        )
    check_imports()
    check_gpus()
    check_flash_attn()
    for hf_id, revision in models.items():
        if hf_id in missing:
            print(f"skip {hf_id}: not staged (--allow-missing)")
        else:
            check_model_offline(hf_id, revision)
    if args.vllm:
        check_vllm_generate(args.vllm, models[args.vllm])
    print("check_container: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
