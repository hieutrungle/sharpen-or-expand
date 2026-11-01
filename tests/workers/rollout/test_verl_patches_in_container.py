"""verl 0.9.1 worker patches on the real classes: the vision-tower freeze (decision 019) on
FSDPEngine, and the phase-memory marks on ActorRolloutRefWorker (M1.10). Container only."""

import inspect

import torch
from torch import nn
from verl.single_controller.base.decorator import MAGIC_ATTR
from verl.workers import engine_workers
from verl.workers.engine.fsdp import transformer_impl

from soe.workers.rollout import phase_memory, verl_patches


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


def test_install_wraps_the_worker_phases_and_keeps_their_dispatch() -> None:
    worker = engine_workers.ActorRolloutRefWorker
    # An earlier install() in this session (the vision test) may have wrapped them already.
    real = {}
    for name in phase_memory.WRAPPED_METHODS:
        method = worker.__dict__[name]
        real[name] = method.__wrapped__ if getattr(method, "soe_phase_memory", False) else method
        setattr(worker, name, real[name])
    try:
        verl_patches.install()
        verl_patches.install()  # idempotent: wrapped once
        for name, original in real.items():
            method = worker.__dict__[name]
            assert method.soe_phase_memory
            assert method.__wrapped__ is original
            assert getattr(method, MAGIC_ATTR) == getattr(original, MAGIC_ATTR)
            assert inspect.iscoroutinefunction(method) == inspect.iscoroutinefunction(original)
        assert inspect.iscoroutinefunction(worker.update_weights)
    finally:
        for name, original in real.items():
            setattr(worker, name, original)
