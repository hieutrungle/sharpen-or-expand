from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.config.experiment import ExperimentConfig, rollout_protocol

EXAMPLE = Path("examples/grpo_trainer/m1_dev_qwen35-2b.yaml")
SUITE = Path("soe/trainer/config/eval_suite.yaml")


def example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text())
    return data


def with_block(block: str, **changes: Any) -> dict[str, Any]:
    data = example()
    return data | {block: data[block] | changes}


def test_unknown_key_rejected() -> None:
    data = example() | {"lerning_rate": 1e-5}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_qlora_only_for_development() -> None:
    data = example() | {"training_mode": "qlora", "development": False, "init_checkpoint": "sft"}
    with pytest.raises(ValidationError, match="QLoRA"):
        ExperimentConfig.model_validate(data)


def test_non_dev_rl_needs_frozen_sft() -> None:
    data = example() | {"development": False}
    with pytest.raises(ValidationError, match="frozen SFT"):
        ExperimentConfig.model_validate(data)


def test_lora_block_matches_mode() -> None:
    data = example() | {"training_mode": "full_bf16"}
    with pytest.raises(ValidationError, match="lora block"):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("exclude", ["model\\.visual\\..*", "mtp\\..*", "nothing"])
def test_lora_must_exclude_vision_and_mtp(exclude: str) -> None:
    with pytest.raises(ValidationError, match="vision tower and MTP"):
        ExperimentConfig.model_validate(with_block("lora", exclude_modules=exclude))


def test_lora_must_not_exclude_the_language_model() -> None:
    with pytest.raises(ValidationError, match="must not exclude the language model"):
        ExperimentConfig.model_validate(with_block("lora", exclude_modules=".*"))


def test_dapo_needs_components_and_clip_higher_consistency() -> None:
    with pytest.raises(ValidationError, match="dapo block"):
        ExperimentConfig.model_validate(example() | {"algorithm": "dapo"})
    components = {
        "dynamic_sampling": True,
        "clip_higher": True,
        "token_level_loss": True,
        "overlong_shaping": False,
    }
    with pytest.raises(ValidationError, match="clip_ratio_high >"):
        ExperimentConfig.model_validate(example() | {"algorithm": "dapo", "dapo": components})
    loss = example()["policy_loss"] | {"clip_ratio_high": 0.28}
    ok = ExperimentConfig.model_validate(
        example() | {"algorithm": "dapo", "dapo": components, "policy_loss": loss}
    )
    assert ok.dapo is not None and ok.dapo.clip_higher


def test_decoupled_clip_without_dapo_rejected() -> None:
    with pytest.raises(ValidationError, match="must equal"):
        ExperimentConfig.model_validate(with_block("policy_loss", clip_ratio_high=0.28))


def test_no_kl_term_can_be_configured() -> None:
    # Decision 013: no RL algorithm has a KL term, so the field does not exist.
    with pytest.raises(ValidationError, match="kl_coef"):
        ExperimentConfig.model_validate(with_block("policy_loss", kl_coef=0.001))


def test_rs_grpo_needs_beta() -> None:
    with pytest.raises(ValidationError, match="rs_grpo_beta"):
        ExperimentConfig.model_validate(example() | {"algorithm": "rs_grpo"})
    cfg = ExperimentConfig.model_validate(example() | {"algorithm": "rs_grpo", "rs_grpo_beta": 2.0})
    assert cfg.rs_grpo_beta == 2.0


def test_stop_rule_needs_both_fields() -> None:
    with pytest.raises(ValidationError, match="both"):
        ExperimentConfig.model_validate(with_block("budget", stop_patience_evals=3))


@pytest.mark.parametrize("block", ["checkpoint", "validation", "hardware"])
def test_rl_needs_checkpoint_validation_and_hardware(block: str) -> None:
    data = example()
    del data[block]
    with pytest.raises(ValidationError, match=f"{block} is required for RL"):
        ExperimentConfig.model_validate(data)


def test_sft_has_no_rl_blocks() -> None:
    with pytest.raises(ValidationError, match="rollout is required for RL"):
        ExperimentConfig.model_validate(example() | {"algorithm": "sft"})


def test_name_must_be_directory_safe() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(example() | {"name": "GRPO dev/run"})


def test_data_hashes_must_be_sha256() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(with_block("data", train_sha256="abc"))


def test_hardware_gpus_must_split_into_rollout_replicas() -> None:
    with pytest.raises(ValidationError, match="multiple of rollout_tensor_parallel"):
        ExperimentConfig.model_validate(with_block("hardware", n_gpus=3, rollout_tensor_parallel=2))


def test_protocol_comes_from_the_suite() -> None:
    config = ExperimentConfig.model_validate(example())
    protocol = rollout_protocol(config, load_suite(SUITE))
    assert protocol.max_response_tokens == 16384
    assert protocol.kv_cache_dtype == "fp8"
    assert protocol.prompt.chat_template_kwargs() == {"enable_thinking": True}
    # Decision 017: rollouts use the config's sampler, validation the suite's.
    rollout, validation = protocol.rollout, protocol.validation
    assert (rollout.temperature, rollout.top_p, rollout.top_k) == (1.0, 1.0, -1)
    assert (validation.temperature, validation.top_p, validation.top_k) == (0.6, 0.95, -1)


def test_rollout_sampler_is_required_and_bounded() -> None:
    data = example()
    del data["rollout"]["temperature"]
    with pytest.raises(ValidationError, match="temperature"):
        ExperimentConfig.model_validate(data)
    with pytest.raises(ValidationError, match="top_p"):
        ExperimentConfig.model_validate(with_block("rollout", top_p=1.5))


def test_updates_per_step_is_required_and_splits_whole_groups() -> None:
    data = example()
    del data["rollout"]["updates_per_step"]
    with pytest.raises(ValidationError, match="updates_per_step"):
        ExperimentConfig.model_validate(data)
    with pytest.raises(ValidationError, match="divisible"):
        ExperimentConfig.model_validate(with_block("rollout", updates_per_step=5))  # 64 prompts
    config = ExperimentConfig.model_validate(with_block("rollout", updates_per_step=16))
    assert config.rollout is not None and config.rollout.updates_per_step == 16


def test_learning_rate_zero_only_for_development_identity_runs() -> None:
    # Decision 018: the export identity test needs a checkpoint the update left unchanged.
    data = with_block("optim", learning_rate=0.0)
    data["budget"] = {k: v for k, v in data["budget"].items() if k != "max_hours"}  # dev-only too
    assert ExperimentConfig.model_validate(data).optim.learning_rate == 0.0
    with pytest.raises(ValidationError, match="learning_rate 0"):
        ExperimentConfig.model_validate(
            data | {"development": False, "init_checkpoint": "checkpoints/sft/hf"}
        )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(with_block("optim", learning_rate=-1e-5))


def test_max_hours_only_for_development() -> None:
    data = with_block("budget", max_hours=20.0)
    assert ExperimentConfig.model_validate(data).budget is not None
    with pytest.raises(ValidationError, match="max_hours"):
        ExperimentConfig.model_validate(
            data | {"development": False, "init_checkpoint": "checkpoints/sft/hf"}
        )


def test_smoke_cap_only_lowers_and_only_for_development() -> None:
    suite = load_suite(SUITE)
    smoke = ExperimentConfig.model_validate(with_block("rollout", smoke_max_response_tokens=4096))
    assert rollout_protocol(smoke, suite).max_response_tokens == 4096
    above = ExperimentConfig.model_validate(with_block("rollout", smoke_max_response_tokens=32768))
    with pytest.raises(ValueError, match="above the suite's cap"):
        rollout_protocol(above, suite)
    data = with_block("rollout", smoke_max_response_tokens=4096) | {
        "development": False,
        "init_checkpoint": "sft",
    }
    with pytest.raises(ValidationError, match="development runs only"):
        ExperimentConfig.model_validate(data)


def test_top_k_and_validation_sampling_cannot_be_set_in_the_config() -> None:
    # Decision 017: only rollout temperature / top-p are RL settings; the rest is the suite's.
    with pytest.raises(ValidationError, match="top_k"):
        ExperimentConfig.model_validate(with_block("rollout", top_k=20))
    with pytest.raises(ValidationError, match="temperature"):
        ExperimentConfig.model_validate(with_block("validation", temperature=1.0))
