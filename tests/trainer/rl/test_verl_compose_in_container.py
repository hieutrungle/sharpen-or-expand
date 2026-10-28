"""Compose verl's real ppo_trainer config with the soe overrides (container only: needs verl).

Run on FASTER:
    sbatch slurm/cpu.sbatch python -m pytest -q tests/trainer/rl/test_verl_compose_in_container.py
It catches keys verl does not know, values Hydra parses differently (quoting, lists) and the
LoRA exclude pattern arriving changed, before any GPU time is spent.
"""

import re
from pathlib import Path

import pytest

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.config.experiment import load_config, rollout_protocol
from soe.trainer.rl.verl_overrides import LaunchPaths, verl_overrides

verl = pytest.importorskip("verl")
hydra = pytest.importorskip("hydra")


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/grpo_trainer/m1_smoke_qwen35-2b.yaml",
        "examples/grpo_trainer/m1_dev_qwen35-2b.yaml",
    ],
)
def test_overrides_compose_into_verls_config(config_path: str) -> None:
    from hydra import compose, initialize_config_dir

    config = load_config(Path(config_path))
    if config.budget is not None and config.budget.max_steps is None:
        pytest.skip("token-budget stopping is wired in M1.5")
    protocol = rollout_protocol(config, load_suite(Path(config.eval_suite)))
    p = Path("/x")
    paths = LaunchPaths(
        model=p, train_file=p, val_file=p, checkpoints=p, reward_function=p, agent_loops=p
    )
    overrides = verl_overrides(config, protocol, seed=1, run_id="r", paths=paths)
    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=overrides)
    rollout = composed.actor_rollout_ref.rollout
    assert (rollout.temperature, rollout.top_p, rollout.response_length) == (0.6, 0.95, 16384)
    assert dict(rollout.engine_kwargs.vllm) == {
        "kv_cache_dtype": "fp8",
        "language_model_only": True,
        "worker_extension_cls": "soe.workers.rollout.vllm_patches.SoeColocateWorkerExtension",
    }
    assert dict(composed.data.apply_chat_template_kwargs) == {"enable_thinking": True}
    if config.lora is not None:
        exclude = composed.actor_rollout_ref.model.exclude_modules
        assert exclude == config.lora.exclude_modules
        assert re.fullmatch(exclude, "model.visual.blocks.0.attn.qkv")
        assert not re.fullmatch(exclude, "model.language_model.layers.0.mlp.down_proj")
