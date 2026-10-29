"""The vision-tower freeze (decision 019) on verl 0.9.1's real FSDPEngine class. Container only."""

import torch
from torch import nn
from verl.workers.engine.fsdp import transformer_impl

from soe.workers.rollout import verl_patches


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleDict({"visual": nn.Linear(2, 2), "language_model": nn.Linear(2, 2)})


def test_install_freezes_only_the_vision_tower_before_fsdp() -> None:
    engine = transformer_impl.FSDPEngine
    real = engine._build_fsdp_module
    seen: list[nn.Module] = []
    engine._build_fsdp_module = lambda self, module: seen.append(module) or module  # stand-in
    try:
        verl_patches.install()
        verl_patches.install()  # idempotent: Ray may run the hook once per worker process
        module = Tiny()
        engine._build_fsdp_module(object(), module)
        assert seen == [module]  # the original build still runs, once
        grads = {n: p.requires_grad for n, p in module.named_parameters()}
        assert grads == {
            "model.visual.weight": False,
            "model.visual.bias": False,
            "model.language_model.weight": True,
            "model.language_model.bias": True,
        }
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        loss = module.model["language_model"](torch.ones(1, 2)).sum()
        loss = loss + 0.0 * module.model["visual"](torch.ones(1, 2)).mean()  # verl's dummy forward
        loss.backward()
        optimizer.step()
        stateful = {n for n, p in module.named_parameters() if optimizer.state.get(p)}
        assert stateful == {"model.language_model.weight", "model.language_model.bias"}
    finally:
        engine._build_fsdp_module = real
