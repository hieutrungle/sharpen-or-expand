"""The verl override list: the frozen protocol is always in it, and any change is a visible diff."""

import os
from pathlib import Path

import pytest
import yaml

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.config.experiment import ExperimentConfig, load_config, rollout_protocol
from soe.trainer.rl.verl_overrides import LaunchPaths, hydra_string, verl_overrides

SMOKE = Path("examples/grpo_trainer/m1_smoke_qwen35-2b.yaml")
SNAPSHOT = Path("tests/trainer/rl/m1_smoke_qwen35-2b.overrides.txt")
PATHS = LaunchPaths(
    model=Path("/models/qwen35-2b"),
    run_dir=Path("/runs/r"),
    train_file=Path("/data/train.parquet"),
    val_file=Path("/data/val.parquet"),
    checkpoints=Path("/runs/r/checkpoints"),
    reward_function=Path("/repo/soe/workers/reward_manager/math_reward.py"),
    agent_loops=Path("/repo/soe/workers/rollout/agent_loops.yaml"),
)


def overrides_for(config: ExperimentConfig) -> list[str]:
    protocol = rollout_protocol(config, load_suite(Path(config.eval_suite)))
    return verl_overrides(config, protocol, seed=7, run_id="r", paths=PATHS)


def smoke_with(**blocks: dict[str, object]) -> ExperimentConfig:
    data = yaml.safe_load(SMOKE.read_text())
    for block, changes in blocks.items():
        data[block] = data[block] | changes
    return ExperimentConfig.model_validate(data)


def test_snapshot() -> None:
    # After a deliberate change: SOE_UPDATE_SNAPSHOT=1 pytest <this file>, then review the git diff.
    overrides = overrides_for(load_config(SMOKE))
    if os.environ.get("SOE_UPDATE_SNAPSHOT"):
        SNAPSHOT.write_text("\n".join(overrides) + "\n")
    assert overrides == SNAPSHOT.read_text().splitlines()


def test_frozen_protocol_is_always_set() -> None:
    overrides = set(overrides_for(load_config(SMOKE)))
    required = {
        "actor_rollout_ref.rollout.temperature=1.0",  # decision 017
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
        "actor_rollout_ref.rollout.response_length=16384",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls="
        "soe.workers.rollout.vllm_patches.SoeColocateWorkerExtension",
        "+data.apply_chat_template_kwargs.enable_thinking=true",
        "actor_rollout_ref.rollout.agent.default_agent_loop=soe_single_turn",
        "reward.custom_reward_function.name=compute_score",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "algorithm.use_kl_in_reward=false",
        "actor_rollout_ref.rollout.seed=7",
        "data.truncation=error",
        "trainer.use_v1=true",
    }
    assert required <= overrides
    assert not any("rollout.seed=0" in o or "val_kwargs.seed" in o for o in overrides)


def test_smoke_cap_reaches_both_rollout_and_data() -> None:
    overrides = overrides_for(smoke_with(rollout={"smoke_max_response_tokens": 4096}))
    assert "actor_rollout_ref.rollout.response_length=4096" in overrides
    assert "data.max_response_length=4096" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=6144" in overrides


def test_full_bf16_has_no_lora_keys() -> None:
    data = yaml.safe_load(SMOKE.read_text())
    data["training_mode"] = "full_bf16"
    del data["lora"]
    overrides = overrides_for(ExperimentConfig.model_validate(data))
    assert not any("lora" in o or "target_modules" in o for o in overrides)


def test_only_grpo_is_wired_in_m1() -> None:
    config = smoke_with()
    dr_grpo = config.model_copy(update={"algorithm": "dr_grpo"})
    with pytest.raises(NotImplementedError, match="M3"):
        overrides_for(dr_grpo)


def test_without_max_steps_the_step_total_is_only_an_upper_bound() -> None:
    # Decision 020: the checkpoint callback lowers verl's total to the token budget's last step.
    data = yaml.safe_load(SMOKE.read_text())
    del data["budget"]["max_steps"]
    overrides = overrides_for(ExperimentConfig.model_validate(data))
    assert "trainer.total_training_steps=null" in overrides
    assert "actor_rollout_ref.actor.optim.lr_scheduler_type=constant" in overrides
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0" in overrides


def test_hydra_string_keeps_backslashes_and_refuses_quotes() -> None:
    assert hydra_string(r"(model\.visual|mtp)\..*") == r"'(model\.visual|mtp)\..*'"
    with pytest.raises(ValueError):
        hydra_string("it's")
