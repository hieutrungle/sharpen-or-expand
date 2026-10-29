"""Freeze Qwen3.5's vision tower in verl's FSDP actor (decision 019).

verl 0.9.1 declares `actor.freeze_vision_tower` but no code reads it. In full-BF16 runs the vision
tower (0.33B parameters) therefore sits in the optimizer: verl's Qwen3.5 forward runs a dummy image
through it and adds `0.0 * image_embeds.mean()`, so it gets exactly-zero gradients, and AdamW keeps
FP32 gradients and two moment buffers for it (about 4 GB on 4B) that never change a weight. LoRA
runs already freeze it with the rest of the base.

`install` is Ray's `worker_process_setup_hook` (named in the overrides), so it runs in every verl
worker process. It must not import torch or verl: that touches CUDA before Ray assigns the actor its
GPU, and both FSDP ranks then land on one device (job 3222890, "Duplicate GPU detected"). It only
registers an import hook; when verl itself imports its FSDP engine module, `patch_engine` wraps
`FSDPEngine._build_fsdp_module`, which runs before the optimizer is built, to set
`requires_grad=False` on every vision-tower parameter. Torch's optimizer then never creates state
for them, and the export (decision 018) checks that they stayed equal to the base.

CPU-importable; the patch itself runs only in the GPU image.
"""

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

PATCHED_VERL = "0.9.1"  # re-check that freeze_vision_tower is still unimplemented on upgrade
HOOK = "soe.trainer.rl.verl_patches.install"
ENGINE_MODULE = "verl.workers.engine.fsdp.transformer_impl"
_VISION_PREFIXES = ("model.visual.", "base_model.model.model.visual.")


def is_vision_parameter(name: str) -> bool:
    return name.startswith(_VISION_PREFIXES)


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    """Finds ENGINE_MODULE with the normal finders, then patches it right after it executes."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != ENGINE_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        execute = loader.exec_module

        def exec_module(module: ModuleType) -> None:
            execute(module)
            patch_engine(module)

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def install() -> None:
    """Ray worker setup hook: patch the engine now if verl loaded it already, else on import."""
    if ENGINE_MODULE in sys.modules:
        patch_engine(sys.modules[ENGINE_MODULE])
    elif not any(isinstance(f, _PatchAfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport())


def patch_engine(transformer_impl: ModuleType) -> None:
    import verl

    if verl.__version__ != PATCHED_VERL:
        raise RuntimeError(f"verl_patches targets verl {PATCHED_VERL}, found {verl.__version__}")
    engine = transformer_impl.FSDPEngine
    original = engine._build_fsdp_module
    if getattr(original, "soe_freezes_vision", False):
        return

    def build_fsdp_module(self, module):  # type: ignore[no-untyped-def]
        frozen = 0
        for name, parameter in module.named_parameters():
            if is_vision_parameter(name) and parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += 1
        if frozen:
            print(f"soe: froze {frozen} vision-tower parameters before FSDP", flush=True)
        return original(self, module)

    build_fsdp_module.soe_freezes_vision = True  # type: ignore[attr-defined]
    engine._build_fsdp_module = build_fsdp_module
