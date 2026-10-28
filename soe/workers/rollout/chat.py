"""Chat-template helpers that need no GPU stack (the tokenizer is passed in as plain callables)."""

from collections.abc import Callable, Collection, Sequence
from typing import Any

# A string no model emits on its own, so it marks where the assistant reply sits in a rendered turn.
_REPLY_MARKER = "SOE_REPLY_MARKER"


def turn_end_text(render_conversation: Callable[[list[dict[str, str]]], str]) -> str:
    """The text the chat template puts right after an assistant reply, e.g. "<|im_end|>" (Qwen).

    Base checkpoints often list only <|endoftext|> as EOS, so a sampler that stops on EOS alone
    runs past the end of the assistant turn until max_tokens, and truncation counts as incorrect.
    """
    rendered = render_conversation(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": _REPLY_MARKER}]
    )
    if rendered.count(_REPLY_MARKER) != 1:
        raise ValueError("the chat template did not render the assistant reply exactly once")
    after = rendered.split(_REPLY_MARKER, 1)[1].strip()
    if not after:
        raise ValueError("the chat template puts nothing after an assistant reply")
    return after


def stop_token_ids(
    eos_token_id: int, turn_end: str, encode: Callable[[str], list[int]]
) -> list[int]:
    """EOS plus the single token that closes an assistant turn."""
    ids = encode(turn_end)
    if len(ids) != 1:
        raise ValueError(f"expected the turn end {turn_end!r} to be one token, got {ids}")
    return sorted({eos_token_id, ids[0]})


def is_truncated(response_ids: Sequence[int], stop_ids: Collection[int]) -> bool:
    """True when generation did not end on a stop token, i.e. it hit the length cap (decision 004:
    a truncated response is incorrect). vLLM keeps the stop token at the end of the output."""
    return not response_ids or response_ids[-1] not in stop_ids


def flat_cache_tensors(kv_caches: Sequence[Any]) -> list[Any]:
    """vLLM's per-layer KV caches as one flat list. Full-attention layers hold one tensor;
    hybrid models' linear-attention layers (Qwen3.5's DeltaNet) hold a list of state tensors.
    None entries are dropped."""
    flat: list[Any] = []
    for entry in kv_caches:
        if entry is None:
            continue
        if isinstance(entry, list | tuple):
            flat.extend(flat_cache_tensors(entry))
        else:
            flat.append(entry)
    return flat
