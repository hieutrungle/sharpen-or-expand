"""verl agent loop for RL rollouts: verl's single-turn loop plus the evaluation sampler's stop rule.

verl 0.9.1's vLLM server sends no stop tokens of its own, and Qwen3.5 base checkpoints list only
<|endoftext|> as EOS, so without this every rollout would run past <|im_end|> to the length cap.
The loop also flags truncation (no stop token at the end); the reward hook then
gives reward 0. A generation error is recorded so the run stops (soe/workers/rollout/failures.py).

Registered for verl in soe/workers/rollout/agent_loops.yaml. GPU image only (imports verl).
"""

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from soe.trainer.rl.rollout_tokens import record_rollout
from soe.workers.rollout.chat import is_truncated, stop_token_ids, turn_end_text
from soe.workers.rollout.failures import failures_recorded


class StopAtTurnEndAgentLoop(SingleTurnAgentLoop):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        tokenizer = self.tokenizer
        turn_end = turn_end_text(
            lambda messages: tokenizer.apply_chat_template(messages, tokenize=False)
        )

        def encode(text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=False))

        self._stop_ids = stop_token_ids(tokenizer.eos_token_id, turn_end, encode)
        self._turn_end_id = encode(turn_end)[0]

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        params = {**sampling_params, "stop_token_ids": self._stop_ids}
        with failures_recorded("agent_loop"):
            output: AgentLoopOutput = await super().run(params, **kwargs)
            # Rollout-token accounting (decision 018): training rollouts only, by verl step.
            if kwargs["extra_info"]["split"] == "train":
                record_rollout(step=int(kwargs["global_steps"]), tokens=len(output.response_ids))
        ids = output.response_ids
        # For the reward hook's rollout records (soe/trainer/rl/metrics.py).
        output.extra_fields["global_step"] = int(kwargs["global_steps"])
        output.extra_fields["truncated"] = is_truncated(ids, self._stop_ids)
        # Diagnostic: a finished turn should close on <|im_end|>, not on the base model's EOS.
        output.extra_fields["turn_end"] = bool(ids) and ids[-1] == self._turn_end_id
        return output
