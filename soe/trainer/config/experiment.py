"""Experiment configs. One YAML file per experiment, validated before any GPU time is spent.

The validators encode the study's standing rules (docs/milestone-plan.md), so a config that would
confound the algorithm comparison fails at load time instead of after a 70-hour run.

The rollout protocol (prompt, thinking mode, KV-cache dtype, response cap) is not repeated here: it
is read from the frozen evaluation suite the config names (`rollout_protocol`), so RL rollouts and
evaluation cannot drift apart.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from soe.trainer.config.eval_suite import EvalSuite, PromptSpec

SHA256_PATTERN = r"^[0-9a-f]{64}$"

# Qwen3.5 checkpoints are multimodal and carry a multi-token-prediction head. LoRA must adapt only
# the language model, so its exclude pattern must match these names and not the language-model one.
# (Module names as peft sees them; Gemma 4 names are added when M8 needs them.)
NON_LANGUAGE_MODULES = (
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.merger.linear_fc1",
    "mtp.layers.0.mlp.down_proj",
)
LANGUAGE_MODULE = "model.language_model.layers.0.mlp.down_proj"


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
    A = "A"  # 4xA40 on fc098: every 4B / Gemma training run (2xA100 only on request)
    B = "B"  # A40 pairs on fc096/fc097: 2B development; sampling runs on 1-GPU workers


class ModelSpec(FrozenModel):
    hf_id: str
    revision: str = Field(min_length=40, max_length=40, description="Hub commit sha")


class LoraSpec(FrozenModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    target_modules: str | list[str]
    # A regex over full module names (peft's `exclude_modules`); must cover NON_LANGUAGE_MODULES.
    exclude_modules: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        exclude = re.compile(self.exclude_modules)
        missed = [name for name in NON_LANGUAGE_MODULES if not exclude.fullmatch(name)]
        if missed:
            raise ValueError(
                f"lora.exclude_modules must exclude the vision tower and MTP: {missed}"
            )
        if exclude.fullmatch(LANGUAGE_MODULE):
            raise ValueError("lora.exclude_modules must not exclude the language model")
        return self


class OptimSpec(FrozenModel):
    # 0 only for development identity runs: the M1.4 export test needs a checkpoint whose weights
    # the update left unchanged (decision 018).
    learning_rate: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)


class RolloutSpec(FrozenModel):
    """Batch shape and the rollout sampler. Rollouts sample at the temperature / top-p given here,
    the same for every algorithm (decision 017); top-k stays off as in evaluation. Prompt, thinking
    mode, KV-cache dtype and the response cap come from the suite; in-loop validation samples
    like evaluation."""

    prompts_per_step: int = Field(gt=0)
    group_size: int = Field(ge=2)
    temperature: float = Field(gt=0, le=2)
    top_p: float = Field(gt=0, le=1)
    max_prompt_tokens: int = Field(
        gt=0
    )  # from the measured maximum (docs/reports/m1_rl_prompts.json)
    # Development smoke runs only: a lower cap than the suite's (decision 004 forbids a higher one).
    smoke_max_response_tokens: int | None = Field(default=None, gt=0)


class PolicyLossSpec(FrozenModel):
    """No KL term exists for any RL algorithm (decision 013), so there is no field for one."""

    clip_ratio_low: float = Field(gt=0)
    clip_ratio_high: float = Field(gt=0)
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
    # Development only (smoke tests, memory-study cells): stop after this many steps.
    max_steps: int | None = Field(default=None, gt=0)
    # G1 stopping rule: stop when in-loop validation pass@1 has not improved by stop_min_delta
    # over stop_patience_evals evaluations. Unset until gate G1 fixes it.
    stop_patience_evals: int | None = Field(default=None, gt=0)
    stop_min_delta: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.stop_patience_evals is None) != (self.stop_min_delta is None):
            raise ValueError("set both stop_patience_evals and stop_min_delta, or neither")
        return self


class CheckpointSpec(FrozenModel):
    """Decision 013: save every N steps (N set so saves land every 2-4 h); each checkpoint records
    its cumulative rollout tokens; runs are compared at the checkpoint nearest each token mark."""

    save_every_steps: int = Field(gt=0)
    keep_last: int = Field(ge=1)  # FSDP shards kept for resume; HF exports for evaluation all stay


class ValidationSpec(FrozenModel):
    """In-loop pass@1 on the validation split (decision 012), at every checkpoint, sampled like
    evaluation (the suite's temperature 0.6, top-p 0.95, 16K; decision 017), not like rollouts."""

    max_problems: int | None = Field(default=None, gt=0)  # None: the whole split
    samples_per_problem: int = Field(gt=0)


class HardwareSpec(FrozenModel):
    """Speed and memory knobs: how fast a step runs, never the algorithm or its batch."""

    n_gpus: int = Field(gt=0)
    rollout_tensor_parallel: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, lt=1)
    param_offload: bool
    optimizer_offload: bool
    max_token_len_per_gpu: int = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.n_gpus % self.rollout_tensor_parallel:
            raise ValueError("n_gpus must be a multiple of rollout_tensor_parallel")
        return self


class DataSpec(FrozenModel):
    """Paths are relative to the data root ($SOE_DATA); hashes are checked at launch."""

    train_file: str
    train_sha256: str = Field(pattern=SHA256_PATTERN)
    val_file: str
    val_sha256: str = Field(pattern=SHA256_PATTERN)


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
    checkpoint: CheckpointSpec | None = None
    validation: ValidationSpec | None = None
    hardware: HardwareSpec | None = None
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
        for block in ("rollout", "policy_loss", "budget", "checkpoint", "validation", "hardware"):
            if is_rl != (getattr(self, block) is not None):
                raise ValueError(f"{block} is required for RL algorithms and forbidden otherwise")
        if (self.algorithm is Algorithm.DAPO) != (self.dapo is not None):
            raise ValueError("the dapo block is required for DAPO and forbidden otherwise")
        if (self.algorithm is Algorithm.RS_GRPO) != (self.rs_grpo_beta is not None):
            raise ValueError("rs_grpo_beta is required for RS-GRPO and forbidden otherwise")
        if self.policy_loss is not None:
            self._check_clip_consistency(self.policy_loss)
        smoke = self.rollout is not None and self.rollout.smoke_max_response_tokens is not None
        if smoke and not self.development:
            raise ValueError("smoke_max_response_tokens is for development runs only")
        if self.budget is not None and self.budget.max_steps is not None and not self.development:
            raise ValueError("budget.max_steps is for development runs only")
        if self.optim.learning_rate == 0 and not self.development:
            raise ValueError("learning_rate 0 is for development identity runs only")
        return self

    def _check_clip_consistency(self, loss: PolicyLossSpec) -> None:
        # Clip-Higher exists only in DAPO; elsewhere a decoupled clip would be a silent confound.
        wants_clip_higher = self.dapo is not None and self.dapo.clip_higher
        is_clip_higher = loss.clip_ratio_high > loss.clip_ratio_low
        if wants_clip_higher and not is_clip_higher:
            raise ValueError("dapo.clip_higher needs clip_ratio_high > clip_ratio_low")
        if not wants_clip_higher and loss.clip_ratio_high != loss.clip_ratio_low:
            raise ValueError("clip_ratio_high must equal clip_ratio_low without Clip-Higher")


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampler:
    temperature: float
    top_p: float
    top_k: int  # -1: off


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutProtocol:
    """What RL rollouts share with evaluation (decisions 004, 006, 009, 014), plus the samplers."""

    prompt: PromptSpec  # template and enable_thinking
    max_response_tokens: int
    kv_cache_dtype: str
    # The config's temperature / top-p with the suite's top-k (decision 017).
    rollout: Sampler
    # The suite's sampler: validation, the G1 stopping signal, measures what evaluation measures.
    # The suite's per-request seed is for evaluation only; both take the engine seed (run seed).
    validation: Sampler


def rollout_protocol(config: ExperimentConfig, suite: EvalSuite) -> RolloutProtocol:
    if config.rollout is None:
        raise ValueError(f"{config.name} is not an RL config")
    cap = suite.sampling.max_tokens
    smoke = config.rollout.smoke_max_response_tokens
    if smoke is not None and smoke > cap:
        raise ValueError(f"smoke_max_response_tokens {smoke} is above the suite's cap {cap}")
    return RolloutProtocol(
        prompt=suite.prompt,
        max_response_tokens=cap if smoke is None else smoke,
        kv_cache_dtype=suite.sampling.kv_cache_dtype,
        rollout=Sampler(
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            top_k=suite.sampling.top_k,
        ),
        validation=Sampler(
            temperature=suite.sampling.temperature,
            top_p=suite.sampling.top_p,
            top_k=suite.sampling.top_k,
        ),
    )


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
