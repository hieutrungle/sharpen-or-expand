from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from soe.trainer.config.experiment import ExperimentConfig

EXAMPLE = Path("examples/grpo_trainer/m1_dev_qwen35-2b.yaml")


def example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text())
    return data


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
    loss = example()["policy_loss"] | {"clip_ratio_high": 0.28}
    with pytest.raises(ValidationError, match="must equal"):
        ExperimentConfig.model_validate(example() | {"policy_loss": loss})


def test_rs_grpo_needs_beta() -> None:
    with pytest.raises(ValidationError, match="rs_grpo_beta"):
        ExperimentConfig.model_validate(example() | {"algorithm": "rs_grpo"})
    cfg = ExperimentConfig.model_validate(example() | {"algorithm": "rs_grpo", "rs_grpo_beta": 2.0})
    assert cfg.rs_grpo_beta == 2.0


def test_budget_checkpoint_interval_and_stop_rule() -> None:
    budget = example()["budget"] | {"checkpoint_every_rollout_tokens": 10**12}
    with pytest.raises(ValidationError, match="checkpoint interval"):
        ExperimentConfig.model_validate(example() | {"budget": budget})
    budget = example()["budget"] | {"stop_patience_evals": 3}
    with pytest.raises(ValidationError, match="both"):
        ExperimentConfig.model_validate(example() | {"budget": budget})


def test_sft_has_no_rl_blocks() -> None:
    with pytest.raises(ValidationError, match="rollout is required for RL"):
        ExperimentConfig.model_validate(example() | {"algorithm": "sft"})


def test_name_must_be_directory_safe() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(example() | {"name": "GRPO dev/run"})
