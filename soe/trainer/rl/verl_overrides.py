"""ExperimentConfig -> Hydra overrides for `python -m verl.trainer.main_ppo` (verl 0.9.1).

A pure function, so every setting a run trains with is visible in one list, written to the run
directory and pinned by a snapshot test. Keys were checked against verl 0.9.1's
`trainer/config/_generated_ppo_trainer.yaml`; a `+` prefix adds a key the schema leaves open.
"""

from dataclasses import dataclass
from pathlib import Path

from soe.trainer.config.experiment import (
    Algorithm,
    ExperimentConfig,
    RolloutProtocol,
    TrainingMode,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LaunchPaths:
    """Absolute paths resolved at launch (offline model snapshot, data, repo files)."""

    model: Path
    run_dir: Path
    train_file: Path
    val_file: Path
    checkpoints: Path
    reward_function: Path  # soe/workers/reward_manager/math_reward.py
    agent_loops: Path  # soe/workers/rollout/agent_loops.yaml


# Decision 001's GRPO row. Dr.GRPO, DAPO and RS-GRPO join in M3, behind the same function.
_ALGORITHM_OVERRIDES: dict[Algorithm, list[str]] = {
    Algorithm.GRPO: [
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=true",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
    ],
}


def hydra_string(value: str) -> str:
    """A single-quoted Hydra string. Hydra keeps backslashes in quoted strings as they are, except
    right before a quote, so only quotes are escaped (a trailing backslash would be ambiguous)."""
    if "'" in value or value.endswith("\\"):
        raise ValueError(f"cannot quote {value!r} for Hydra safely")
    return "'" + value + "'"


def _flag(value: bool) -> str:
    return "true" if value else "false"


def verl_overrides(
    config: ExperimentConfig,
    protocol: RolloutProtocol,
    *,
    seed: int,
    run_id: str,
    paths: LaunchPaths,
) -> list[str]:
    algorithm = _ALGORITHM_OVERRIDES.get(config.algorithm)
    if algorithm is None:
        raise NotImplementedError(f"{config.algorithm} is wired in M3; M1 supports grpo only")
    rollout, loss, checkpoint = config.rollout, config.policy_loss, config.checkpoint
    validation, hardware, budget = config.validation, config.hardware, config.budget
    if not (rollout and loss and checkpoint and validation and hardware and budget):
        raise ValueError(f"{config.name} is not an RL config")
    if budget.max_steps is None:
        # The rollout-token stop (decision 013) arrives with the trainer hook in M1.5/M1.6.
        raise NotImplementedError("token-budget stopping is not wired yet; set budget.max_steps")

    prompt_tokens, response_tokens = rollout.max_prompt_tokens, protocol.max_response_tokens
    sampling = [
        f"temperature={protocol.temperature}",
        f"top_p={protocol.top_p}",
        f"top_k={protocol.top_k}",
    ]
    fsdp = "actor_rollout_ref.actor.fsdp_config"
    vllm = "+actor_rollout_ref.rollout.engine_kwargs.vllm"
    thinking = _flag(bool(protocol.prompt.enable_thinking))
    overrides = [
        *algorithm,
        "algorithm.use_kl_in_reward=false",  # decision 013: no KL, no reference model
        # model
        f"actor_rollout_ref.model.path={paths.model}",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        # actor: one optimizer step per rollout batch
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={rollout.prompts_per_step}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={hardware.max_token_len_per_gpu}",
        f"actor_rollout_ref.actor.optim.lr={config.optim.learning_rate}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={config.optim.warmup_steps}",
        f"actor_rollout_ref.actor.optim.weight_decay={config.optim.weight_decay}",
        f"actor_rollout_ref.actor.clip_ratio_low={loss.clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={loss.clip_ratio_high}",
        f"actor_rollout_ref.actor.entropy_coeff={loss.entropy_coef}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        # The 248K vocabulary makes fp32 logits about 1 MB per token: compute entropy in chunks.
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true",
        f"{fsdp}.param_offload={_flag(hardware.param_offload)}",
        f"{fsdp}.optimizer_offload={_flag(hardware.optimizer_offload)}",
        f"{fsdp}.seed={seed}",
        # rollout: the frozen protocol (decisions 004, 006, 009, 014)
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={rollout.group_size}",
        *(f"actor_rollout_ref.rollout.{s}" for s in sampling),
        f"actor_rollout_ref.rollout.seed={seed}",  # engine seed; no per-request seed (014)
        f"actor_rollout_ref.rollout.prompt_length={prompt_tokens}",
        f"actor_rollout_ref.rollout.response_length={response_tokens}",
        f"actor_rollout_ref.rollout.max_model_len={prompt_tokens + response_tokens}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={hardware.rollout_tensor_parallel}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={hardware.gpu_memory_utilization}",
        "actor_rollout_ref.rollout.calculate_log_probs=true",  # rollout/trainer log-prob gap
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={hardware.max_token_len_per_gpu}",
        f"{vllm}.kv_cache_dtype={protocol.kv_cache_dtype}",
        f"{vllm}.language_model_only=true",  # decision 001: text only
        # Backport of vLLM PR #41602: FP8 KV wake-up on hybrid models (decision 015)
        f"{vllm}.worker_extension_cls=soe.workers.rollout.vllm_patches.SoeColocateWorkerExtension",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={paths.agent_loops}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=soe_single_turn",
        # in-loop validation, sampled like the rollouts (decision 014), on the decision-012 split
        *(f"actor_rollout_ref.rollout.val_kwargs.{s}" for s in sampling),
        f"actor_rollout_ref.rollout.val_kwargs.n={validation.samples_per_problem}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        # data
        f"data.train_files={paths.train_file}",
        f"data.val_files={paths.val_file}",
        "data.prompt_key=prompt",
        f"data.max_prompt_length={prompt_tokens}",
        f"data.max_response_length={response_tokens}",
        f"data.train_batch_size={rollout.prompts_per_step}",
        f"data.val_max_samples={validation.max_problems or -1}",
        f"+data.apply_chat_template_kwargs.enable_thinking={thinking}",
        # Never drop or cut a prompt silently: max_prompt_tokens covers the measured maximum.
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        f"data.seed={seed}",
        # reward: the evaluation verifier (decisions 008, 010) through the soe hook
        f"reward.custom_reward_function.path={paths.reward_function}",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_manager.name=naive",
        # trainer
        "trainer.use_v1=true",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={hardware.n_gpus}",
        "trainer.project_name=sharpen-or-expand",
        f"trainer.experiment_name={run_id}",
        "trainer.logger=[console,file]",
        f"trainer.default_local_dir={paths.checkpoints}",
        "trainer.resume_mode=auto",
        f"trainer.save_freq={checkpoint.save_every_steps}",
        f"trainer.max_actor_ckpt_to_keep={checkpoint.keep_last}",
        f"trainer.test_freq={checkpoint.save_every_steps}",  # validate at every checkpoint
        "trainer.val_before_train=true",
        f"trainer.total_training_steps={budget.max_steps}",
        "trainer.total_epochs=1000",  # the step (later token) budget stops a run, never epochs
        # Hydra's resolved config goes with the run, not into the repo's working directory
        f"hydra.run.dir={paths.run_dir}/hydra",
    ]
    if config.training_mode is TrainingMode.LORA_BF16:
        if config.lora is None:
            raise ValueError("lora_bf16 needs a lora block")
        targets = config.lora.target_modules
        target = targets if isinstance(targets, str) else "[" + ",".join(targets) + "]"
        overrides += [
            f"actor_rollout_ref.model.lora_rank={config.lora.rank}",
            f"actor_rollout_ref.model.lora_alpha={config.lora.alpha}",
            f"actor_rollout_ref.model.target_modules={target}",
            f"actor_rollout_ref.model.exclude_modules={hydra_string(config.lora.exclude_modules)}",
        ]
    elif config.training_mode is not TrainingMode.FULL_BF16:
        raise NotImplementedError(f"{config.training_mode} is not wired (QLoRA is dev-only)")
    return overrides
