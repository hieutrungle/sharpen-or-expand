"""SFT examples and their tokenization (decision 006 format; M1.8).

An example is one problem's user turn (the frozen suite's template, as evaluation renders it) and a
completion: the text a model generates after the generation prompt, i.e. `reasoning</think>` then
the answer. Training sees exactly what rollouts and evaluation see: the prompt rendered by the chat
template with the suite's `enable_thinking` (ending in `<think>\n`), then the completion's tokens
and `<|im_end|>`, the turn end every sampler stops on. Only the completion tokens (and the turn end)
carry loss; TRL sets the label of every token outside `completion_mask` to -100.

No torch, TRL or transformers import, so this runs in the CPU tests.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

THINK_END = "</think>"


class SftExample(BaseModel):
    """One line of an SFT data file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_id: str
    user: str  # the suite-rendered user turn (template applied, chat template not)
    completion: str  # model text after the generation prompt, without the turn-end token


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedExample:
    input_ids: list[int]
    completion_mask: list[int]  # 1 where the token carries loss


def tokenize_example(
    prompt: str, completion: str, *, encode: Callable[[str], list[int]], turn_end_id: int
) -> TokenizedExample:
    """Prompt and completion are encoded apart and joined, so the prompt's tokens are exactly the
    ones the sampler conditions on; no boundary merge can move a token between them."""
    prompt_ids = encode(prompt)
    completion_ids = [*encode(completion), turn_end_id]
    return TokenizedExample(
        input_ids=prompt_ids + completion_ids,
        completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
    )


def fitting(
    examples: Sequence[TokenizedExample], max_length: int
) -> tuple[list[TokenizedExample], int]:
    """Examples that fit in max_length, and how many were dropped (never truncated: a cut example
    would teach reasoning that never reaches an answer or a stop)."""
    kept = [e for e in examples if len(e.input_ids) <= max_length]
    return kept, len(examples) - len(kept)
