"""Experiment configs. One YAML file per experiment, validated before any GPU time is spent.

The validators encode the study's standing rules (docs/milestone-plan.md), so a config that would
confound the algorithm comparison fails at load time instead of after a 70-hour run.
"""

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Algorithm(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    DR_GRPO = "dr_grpo"
    DAPO = "dapo"
    RS_GRPO = "rs_grpo"


RL_ALGORITHMS = frozenset({Algorithm.GRPO, Algorithm.DR_GRPO, Algorithm.DAPO, Algorithm.RS_GRPO})


class TrainingMode(StrEnum):
    FULL_BF16 = "full_bf16"
    LORA_BF16 = "lora_bf16"
    QLORA = "qlora"  # development only


class Lane(StrEnum):
    A = "A"  # 2xA100: every 4B / Gemma training run
    B = "B"  # 2xA40: 2B development, evaluation, preference generation, small sweeps


class ModelSpec(FrozenModel):
    hf_id: str
    revision: str = Field(min_length=40, max_length=40, description="Hub commit sha")


class LoraSpec(FrozenModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    target_modules: str | list[str]


class OptimSpec(FrozenModel):
    learning_rate: float = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)


class RolloutSpec(FrozenModel):
    prompts_per_step: int = Field(gt=0)
    group_size: int = Field(ge=2)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    max_prompt_tokens: int = Field(gt=0)
    max_response_tokens: int = Field(gt=0)


class PolicyLossSpec(FrozenModel):
    clip_ratio_low: float = Field(gt=0)
    clip_ratio_high: float = Field(gt=0)
    kl_coef: float = Field(ge=0)
    entropy_coef: float = Field(ge=0)


class DapoComponents(FrozenModel):
    """The four DAPO components; leave-one-out ablations flip exactly one of them."""

    dynamic_sampling: bool
    clip_higher: bool
    token_level_loss: bool
    overlong_shaping: bool


class BudgetSpec(FrozenModel):
    """Runs are compared at equal rollout-token budgets, never equal steps."""

    rollout_tokens: int = Field(gt=0)
    checkpoint_every_rollout_tokens: int = Field(gt=0)
    # G1 stopping rule: stop when in-loop validation pass@1 has not improved by stop_min_delta
    # over stop_patience_evals evaluations. Unset until gate G1 fixes it.
    stop_patience_evals: int | None = Field(default=None, gt=0)
    stop_min_delta: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.checkpoint_every_rollout_tokens > self.rollout_tokens:
            raise ValueError("checkpoint interval is larger than the whole rollout budget")
        if (self.stop_patience_evals is None) != (self.stop_min_delta is None):
            raise ValueError("set both stop_patience_evals and stop_min_delta, or neither")
        return self


class DataSpec(FrozenModel):
    """Paths are relative to the data root ($SOE_DATA); existence is checked at launch."""

    train_file: str
    val_file: str


class ExperimentConfig(FrozenModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")  # used in run directory names
    milestone: str = Field(pattern=r"^m\d$")
    development: bool  # true for 2B/dev runs, which never enter conclusions
    lane: Lane
    model: ModelSpec
    init_checkpoint: str | None  # the frozen SFT checkpoint, relative to $SOE_DATA
    training_mode: TrainingMode
    lora: LoraSpec | None = None
    algorithm: Algorithm
    optim: OptimSpec
    rollout: RolloutSpec | None = None
    policy_loss: PolicyLossSpec | None = None
    budget: BudgetSpec | None = None
    dapo: DapoComponents | None = None
    rs_grpo_beta: float | None = Field(default=None, gt=0)
    data: DataSpec
    eval_suite: (
        str  # the frozen suite YAML (soe/trainer/config/eval_suite.yaml), relative to repo root
    )

    @model_validator(mode="after")
    def _check_standing_rules(self) -> Self:
        uses_lora = self.training_mode in (TrainingMode.LORA_BF16, TrainingMode.QLORA)
        if uses_lora != (self.lora is not None):
            raise ValueError("a lora block is required for LoRA modes and forbidden otherwise")
        if self.training_mode is TrainingMode.QLORA and not self.development:
            raise ValueError("QLoRA is for development runs only")
        starts_from_sft = self.algorithm in RL_ALGORITHMS or self.algorithm is Algorithm.DPO
        if starts_from_sft and not self.development and self.init_checkpoint is None:
            raise ValueError(
                "every non-development RL/DPO run starts from the frozen SFT checkpoint"
            )
        is_rl = self.algorithm in RL_ALGORITHMS
        for block in ("rollout", "policy_loss", "budget"):
            if is_rl != (getattr(self, block) is not None):
                raise ValueError(f"{block} is required for RL algorithms and forbidden otherwise")
        if (self.algorithm is Algorithm.DAPO) != (self.dapo is not None):
            raise ValueError("the dapo block is required for DAPO and forbidden otherwise")
        if (self.algorithm is Algorithm.RS_GRPO) != (self.rs_grpo_beta is not None):
            raise ValueError("rs_grpo_beta is required for RS-GRPO and forbidden otherwise")
        if self.policy_loss is not None:
            self._check_clip_consistency(self.policy_loss)
        return self

    def _check_clip_consistency(self, loss: PolicyLossSpec) -> None:
        # Clip-Higher exists only in DAPO; elsewhere a decoupled clip would be a silent confound.
        wants_clip_higher = self.dapo is not None and self.dapo.clip_higher
        is_clip_higher = loss.clip_ratio_high > loss.clip_ratio_low
        if wants_clip_higher and not is_clip_higher:
            raise ValueError("dapo.clip_higher needs clip_ratio_high > clip_ratio_low")
        if not wants_clip_higher and loss.clip_ratio_high != loss.clip_ratio_low:
            raise ValueError("clip_ratio_high must equal clip_ratio_low without Clip-Higher")


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
