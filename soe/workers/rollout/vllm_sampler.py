"""Text-only vLLM sampling with the frozen suite's prompt and sampling rules; every completion is
scored by the same verifier RL rewards use and returned as a SampleRecord.

GPU-only: vLLM is imported when a sampler is built, so CPU tests can import this module's users.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from soe.trainer.config.eval_suite import PromptSpec, SamplingSpec
from soe.utils.dataset.problems import Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.reward_score.math_verify import score_response
from soe.workers.rollout.chat import stop_token_ids, turn_end_text

# Room for the prompt on top of max_tokens; the longest suite prompt is far below this.
PROMPT_TOKEN_ALLOWANCE = 2048


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineSettings:
    """vLLM engine knobs: they change speed and memory use, never which tokens are sampled."""

    gpu_memory_utilization: float = 0.9
    max_num_seqs: int | None = None  # None: vLLM's default
    log_stats: bool = False  # periodic throughput, running/waiting requests, KV-cache use

    def llm_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "disable_log_stats": not self.log_stats,
        }
        if self.max_num_seqs is not None:
            kwargs["max_num_seqs"] = self.max_num_seqs
        return kwargs


class VllmSampler:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None,
        max_tokens: int,
        engine: "EngineSettings | None" = None,
        kv_cache_dtype: str = "auto",
    ) -> None:
        from vllm import LLM

        self.engine = engine or EngineSettings()

        # Qwen3.5 and Gemma 4 are multimodal checkpoints; the study is text-only, and without
        # this vLLM also loads and profiles the vision encoder.
        self._llm = LLM(
            model=model,
            revision=revision,
            max_model_len=max_tokens + PROMPT_TOKEN_ALLOWANCE,
            language_model_only=True,
            # Not an engine knob: a quantized KV cache changes which tokens are sampled.
            kv_cache_dtype=kv_cache_dtype,
            **self.engine.llm_kwargs(),
        )
        self._tokenizer: Any = self._llm.get_tokenizer()
        turn_end = turn_end_text(
            lambda messages: self._tokenizer.apply_chat_template(messages, tokenize=False)
        )
        self.stop_token_ids = stop_token_ids(
            self._tokenizer.eos_token_id,
            turn_end,
            lambda text: self._tokenizer.encode(text, add_special_tokens=False),
        )

    def render(self, question: str, prompt: PromptSpec) -> str:
        text = prompt.render(question)
        if not prompt.use_chat_template:
            return text
        rendered: str = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            **prompt.chat_template_kwargs(),
        )
        return rendered

    def sample(
        self,
        problems: Sequence[Problem],
        *,
        prompt: PromptSpec,
        sampling: SamplingSpec,
        n: int,
        checkpoint: str,
    ) -> list[SampleRecord]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,  # per request: every problem gets the same random stream
            stop_token_ids=self.stop_token_ids,
        )
        outputs = self._llm.generate(
            [self.render(p.question, prompt) for p in problems], params, use_tqdm=False
        )
        records = []
        for problem, output in zip(problems, outputs, strict=True):
            for index, completion in enumerate(output.outputs):
                if completion.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"{problem.problem_id}: finish {completion.finish_reason}")
                verdict = score_response(completion.text, problem.answer)
                records.append(
                    SampleRecord(
                        problem_id=problem.problem_id,
                        sample_index=index,
                        checkpoint=checkpoint,
                        completion=completion.text,
                        extracted_answer=verdict.extracted,
                        # decision 004: a truncated response is incorrect even if it has a box
                        correct=verdict.correct and completion.finish_reason == "stop",
                        completion_tokens=len(completion.token_ids),
                        finish_reason=completion.finish_reason,
                    )
                )
        return records
