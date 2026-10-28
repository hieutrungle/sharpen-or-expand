"""Export a verl FSDP2 actor checkpoint to an HF directory that main_eval can load (decision 018).

verl's own model_merger is not used: it does not merge LoRA (its "HF model" of a LoRA run is the
base) and it casts to BF16 before merging. This export:
1. loads every rank's shards in FP32 and joins them (FSDP2 shards each tensor on dim 0);
2. merges each LoRA adapter as W0 + (alpha / r) * B @ A in FP32;
3. counts the trained parameters from the optimizer state (a parameter that never got a gradient
   has none), and requires them all to be LoRA matrices or language-model weights;
4. requires every other tensor (vision tower) to equal the base bit for bit, copies the base's mtp.*
   head unchanged (transformers does not load it, so RL never trained it), and drops the tied
   lm_head copy;
5. writes the tensors in the base's dtypes and file layout, copies the base's other files, adds
   soe_export.json, and renames the directory into place, so a half-written export never exists.

Image only (torch, safetensors).
"""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard

from soe.trainer.rl.checkpoints import PEFT_PREFIX, hf_name, is_language_model, lora_parts


class ExportError(RuntimeError):
    pass


class ExportInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    step: int
    rollout_tokens: int | None
    source_checkpoint: str
    base_model: str
    training: str  # "lora" or "full"
    lora_rank: int | None
    lora_alpha: int | None
    merged_lora_modules: int
    trained_tensors: int  # from the optimizer state
    trained_parameters: int
    trained_by_prefix: dict[str, int]
    tensors_changed_vs_base: int
    copied_from_base: list[str]  # untrained tensors taken from the base snapshot (mtp.*)
    created_at: datetime


def _rank_files(actor_dir: Path, kind: str) -> list[Path]:
    files = sorted(actor_dir.glob(f"{kind}_world_size_*_rank_*.pt"))
    if not files:
        raise ExportError(f"no {kind} shards in {actor_dir}")
    world = int(files[0].name.split("_")[3])
    expected = [actor_dir / f"{kind}_world_size_{world}_rank_{r}.pt" for r in range(world)]
    if files != sorted(expected):
        raise ExportError(f"incomplete {kind} shards in {actor_dir}: {[f.name for f in files]}")
    return expected


def load_full_state_dict(actor_dir: Path) -> dict[str, torch.Tensor]:
    """The unsharded FP32 state dict, in the checkpoint's key order."""
    shards = [
        torch.load(f, map_location="cpu", weights_only=False)
        for f in _rank_files(actor_dir, "model")
    ]
    full: dict[str, torch.Tensor] = {}
    for key, first in shards[0].items():
        if not isinstance(first, DTensor):
            full[key] = first.float()
            continue
        (placement,) = first.placements  # FSDP2 without tensor parallelism: a 1-D mesh
        if isinstance(placement, Replicate):
            full[key] = first._local_tensor.float()
        elif isinstance(placement, Shard):
            locals_ = [shard[key]._local_tensor.float() for shard in shards]
            full[key] = torch.cat(locals_, dim=placement.dim)
        else:
            raise ExportError(f"unsupported placement {placement} for {key}")
        if tuple(full[key].shape) != tuple(first.shape):
            raise ExportError(
                f"{key}: joined shape {tuple(full[key].shape)} != {tuple(first.shape)}"
            )
    return full


def trained_keys(actor_dir: Path, state_dict_keys: list[str]) -> list[str]:
    """State-dict keys of the parameters that have optimizer state, i.e. were ever updated.

    torch's optimizer numbers parameters in `model.parameters()` order, which is the state-dict
    order without the tied lm_head duplicate."""
    optim = torch.load(_rank_files(actor_dir, "optim")[0], map_location="cpu", weights_only=False)
    param_ids = [i for group in optim["param_groups"] for i in group["params"]]
    params = [k for k in state_dict_keys if k.removeprefix(PEFT_PREFIX) != "lm_head.weight"]
    if len(params) != len(param_ids):
        raise ExportError(
            f"{len(params)} parameters in the state dict but {len(param_ids)} in the optimizer: "
            "the index-to-name mapping would be wrong"
        )
    name_of = dict(zip(param_ids, params, strict=True))
    return [name_of[i] for i, state in optim["state"].items() if state]


def merge_lora(full: dict[str, torch.Tensor], scale: float) -> tuple[dict[str, torch.Tensor], int]:
    """HF-named FP32 tensors with every adapter merged; also the number of merged modules."""
    tensors: dict[str, torch.Tensor] = {}
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in full.items():
        parts = lora_parts(key)
        if parts is None:
            tensors[hf_name(key)] = value
        else:
            weight, part = parts
            adapters.setdefault(weight, {})[part] = value
    for weight, ab in adapters.items():
        if set(ab) != {"A", "B"} or weight not in tensors:
            raise ExportError(f"incomplete LoRA adapter for {weight}")
        tensors[weight] = tensors[weight] + scale * (ab["B"] @ ab["A"])
    return tensors, len(adapters)


def _read_base(base_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors: dict[str, torch.Tensor] = {}
    file_of: dict[str, str] = {}
    for path in sorted(base_dir.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                tensors[name] = f.get_tensor(name)
                file_of[name] = path.name
    return tensors, file_of


def _prefix(name: str) -> str:
    return ".".join(name.split(".")[:2])


def export_checkpoint(
    actor_dir: Path,
    base_dir: Path,
    out_dir: Path,
    *,
    run_id: str,
    step: int,
    rollout_tokens: int | None,
    base_model: str,
) -> ExportInfo:
    if out_dir.exists():
        raise ExportError(f"{out_dir} exists already")
    full = load_full_state_dict(actor_dir)
    numels = {key: value.numel() for key, value in full.items()}
    trained = trained_keys(actor_dir, list(full))
    is_lora = any(lora_parts(k) is not None for k in full)
    for key in trained:
        parts = lora_parts(key)
        if is_lora:
            allowed = parts is not None and is_language_model(parts[0])
        else:
            allowed = is_language_model(hf_name(key))
        if not allowed:
            raise ExportError(
                f"{key} was trained; only language-model weights (LoRA: adapters) may be"
            )

    lora_meta: dict[str, int] = {}
    if is_lora:
        lora_meta = json.loads((actor_dir / "lora_train_meta.json").read_text())
    scale = lora_meta["lora_alpha"] / lora_meta["r"] if is_lora else 0.0
    tensors, merged = (
        merge_lora(full, scale) if is_lora else ({hf_name(k): v for k, v in full.items()}, 0)
    )
    del full

    base, file_of = _read_base(base_dir)
    config = json.loads((base_dir / "config.json").read_text())
    tied = bool(
        config.get("tie_word_embeddings", config.get("text_config", {}).get("tie_word_embeddings"))
    )
    if "lm_head.weight" in tensors and "lm_head.weight" not in base:
        if not tied or not torch.equal(
            tensors["lm_head.weight"], tensors["model.language_model.embed_tokens.weight"]
        ):
            raise ExportError("lm_head is not tied to the embedding but the base has no lm_head")
        del tensors["lm_head.weight"]

    unknown = sorted(set(tensors) - set(base))
    if unknown:
        raise ExportError(f"tensors not in the base: {unknown[:5]}")
    copied = sorted(set(base) - set(tensors))
    if any(not name.startswith("mtp.") for name in copied):
        raise ExportError(f"base tensors missing from the checkpoint: {copied[:5]}")

    out: dict[str, torch.Tensor] = {}
    changed = 0
    for name, base_tensor in base.items():
        if name in copied:
            out[name] = base_tensor
            continue
        value = tensors.pop(name)
        if tuple(value.shape) != tuple(base_tensor.shape):
            raise ExportError(
                f"{name}: shape {tuple(value.shape)} != base {tuple(base_tensor.shape)}"
            )
        value = value.to(base_tensor.dtype).contiguous()
        if not torch.equal(value, base_tensor):
            if not is_language_model(name):
                raise ExportError(f"{name} differs from the base but is outside the language model")
            changed += 1
        out[name] = value

    trained_by_prefix: dict[str, int] = {}
    for key in trained:
        prefix = _prefix(hf_name(key))
        trained_by_prefix[prefix] = trained_by_prefix.get(prefix, 0) + numels[key]

    info = ExportInfo(
        run_id=run_id,
        step=step,
        rollout_tokens=rollout_tokens,
        source_checkpoint=str(actor_dir),
        base_model=base_model,
        training="lora" if is_lora else "full",
        lora_rank=lora_meta.get("r"),
        lora_alpha=lora_meta.get("lora_alpha"),
        merged_lora_modules=merged,
        trained_tensors=len(trained),
        trained_parameters=sum(numels[key] for key in trained),
        trained_by_prefix=trained_by_prefix,
        tensors_changed_vs_base=changed,
        copied_from_base=copied,
        created_at=datetime.now(UTC),
    )

    tmp = out_dir.parent / f".{out_dir.name}.tmp-{os.getpid()}"
    tmp.mkdir(parents=True)
    for file_name in sorted(set(file_of.values())):
        part = {n: out[n] for n in out if file_of[n] == file_name}
        save_file(part, str(tmp / file_name), metadata={"format": "pt"})
    for path in base_dir.iterdir():
        if not path.name.endswith(".safetensors") and path.is_file():
            shutil.copyfile(path, tmp / path.name)  # snapshot files are symlinks into blobs
    (tmp / "soe_export.json").write_text(info.model_dump_json(indent=1) + "\n")
    try:
        tmp.rename(out_dir)
    except OSError:
        if not out_dir.exists():
            raise
        shutil.rmtree(tmp)  # another process (callback or sweep) finished the same step first
    return info
