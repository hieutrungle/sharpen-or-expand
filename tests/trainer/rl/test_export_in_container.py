"""HF export of a verl checkpoint (decision 018) on a tiny synthetic model, plain (non-DTensor)
shards. Needs torch and safetensors: run in the container on a CPU node.

The real-checkpoint checks (step-0 identity runs, the trained M1.3 smoke checkpoint) are run by
scripts/check_export.py.
"""

import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from soe.trainer.rl.export import ExportError, export_checkpoint, like_base

P = "base_model.model."
UP = "model.language_model.layers.0.mlp.up_proj"
EMBED = "model.language_model.embed_tokens.weight"
VISUAL = "model.visual.proj.weight"
MTP = "mtp.fc.weight"


def make_base(root: Path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    tensors = {
        f"{UP}.weight": torch.randn(6, 4).bfloat16(),
        EMBED: torch.randn(10, 4).bfloat16(),
        VISUAL: torch.randn(3, 3).bfloat16(),
        MTP: torch.randn(4, 4).bfloat16(),
        "model.language_model.layers.0.linear_attn.A_log": torch.randn(2).float(),  # F32 in Qwen3.5
    }
    root.mkdir()
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    (root / "tokenizer.json").write_text("{}")
    return tensors


def make_checkpoint(
    root: Path, base: dict[str, torch.Tensor], lora_b: torch.Tensor, trained: list[str]
) -> None:
    """A world-size-1 checkpoint as verl writes it for a PEFT model (FP32, PEFT names)."""
    lora_a = torch.randn(2, 4)
    state = OrderedDict(
        [
            (f"{P}{UP}.base_layer.weight", base[f"{UP}.weight"].float()),
            (f"{P}{UP}.lora_A.default.weight", lora_a),
            (f"{P}{UP}.lora_B.default.weight", lora_b),
            (f"{P}{EMBED}", base[EMBED].float()),
            (
                f"{P}model.language_model.layers.0.linear_attn.A_log",
                base["model.language_model.layers.0.linear_attn.A_log"],
            ),
            (f"{P}{VISUAL}", base[VISUAL].float()),
            (f"{P}lm_head.weight", base[EMBED].float()),  # tied copy
        ]
    )
    root.mkdir(parents=True)
    torch.save(state, root / "model_world_size_1_rank_0.pt")
    params = [k for k in state if k != f"{P}lm_head.weight"]
    optim = {
        "param_groups": [{"params": list(range(len(params)))}],
        "state": {params.index(k): {"step": torch.tensor(1.0)} for k in trained},
    }
    torch.save(optim, root / "optim_world_size_1_rank_0.pt")
    (root / "lora_train_meta.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    torch.save(lora_a, root / "lora_a.pt")  # for the expected value below


def read(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path / "model.safetensors", "pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}  # noqa: SIM118


def run(actor: Path, base: Path, out: Path):  # type: ignore[no-untyped-def]
    return export_checkpoint(
        actor, base, out, run_id="r", step=1, rollout_tokens=7, base_model="tiny"
    )


ADAPTERS = [f"{P}{UP}.lora_A.default.weight", f"{P}{UP}.lora_B.default.weight"]


def test_lora_is_merged_mtp_copied_and_tied_head_dropped(tmp_path: Path) -> None:
    base = make_base(tmp_path / "base")
    lora_b = torch.randn(6, 2)
    make_checkpoint(tmp_path / "actor", base, lora_b, trained=ADAPTERS)
    info = run(tmp_path / "actor", tmp_path / "base", tmp_path / "hf")
    out = read(tmp_path / "hf")
    lora_a = torch.load(tmp_path / "actor" / "lora_a.pt")
    expected = (base[f"{UP}.weight"].float() + 2.0 * (lora_b @ lora_a)).bfloat16()
    assert torch.equal(out[f"{UP}.weight"], expected)
    assert set(out) == set(base)  # no lm_head; mtp present
    assert torch.equal(out[MTP], base[MTP]) and torch.equal(out[VISUAL], base[VISUAL])
    assert out["model.language_model.layers.0.linear_attn.A_log"].dtype == torch.float32
    assert (info.training, info.merged_lora_modules, info.tensors_changed_vs_base) == ("lora", 1, 1)
    assert (info.trained_tensors, info.trained_parameters) == (2, 2 * 4 + 6 * 2)
    assert info.copied_from_base == [MTP]
    assert (tmp_path / "hf" / "tokenizer.json").exists() and (
        tmp_path / "hf" / "soe_export.json"
    ).exists()


def test_zero_lora_b_exports_the_base_bit_for_bit(tmp_path: Path) -> None:
    base = make_base(tmp_path / "base")
    make_checkpoint(tmp_path / "actor", base, torch.zeros(6, 2), trained=ADAPTERS)
    info = run(tmp_path / "actor", tmp_path / "base", tmp_path / "hf")
    out = read(tmp_path / "hf")
    assert all(torch.equal(out[name], base[name]) for name in base)
    assert info.tensors_changed_vs_base == 0


def test_a_trained_or_changed_vision_tower_is_refused(tmp_path: Path) -> None:
    base = make_base(tmp_path / "base")
    make_checkpoint(tmp_path / "a1", base, torch.zeros(6, 2), trained=[*ADAPTERS, f"{P}{VISUAL}"])
    with pytest.raises(ExportError, match="was trained"):
        run(tmp_path / "a1", tmp_path / "base", tmp_path / "hf1")
    changed = dict(base)
    changed[VISUAL] = base[VISUAL] + 1
    make_checkpoint(tmp_path / "a2", changed, torch.zeros(6, 2), trained=ADAPTERS)
    with pytest.raises(ExportError, match="outside the language model"):
        run(tmp_path / "a2", tmp_path / "base", tmp_path / "hf2")
    assert not (tmp_path / "hf1").exists() and not (tmp_path / "hf2").exists()


def test_lora_export_refuses_changed_untrained_tensors(tmp_path: Path) -> None:
    """An F32 tensor that went through BF16 differs from the base although nothing trained it
    (M1.8, job 3223729): with one merged adapter, a second changed tensor is refused."""
    base = make_base(tmp_path / "base")
    tensors = {name: t.float() for name, t in base.items() if not name.startswith("mtp.")}
    tensors[f"{UP}.weight"] = tensors[f"{UP}.weight"] + 1  # the merged adapter
    a_log = "model.language_model.layers.0.linear_attn.A_log"
    tensors[a_log] = base[a_log].bfloat16().float()  # rounded, untrained
    if torch.equal(tensors[a_log], base[a_log]):
        pytest.skip("the random A_log happens to be exact in BF16")
    with pytest.raises(ExportError, match="untrained weights changed"):
        like_base(dict(tensors), tmp_path / "base", lora_modules=1)
    assert like_base(dict(tensors), tmp_path / "base", lora_modules=None).changed == 2
