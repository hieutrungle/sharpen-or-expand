import pytest

from soe.workers.rollout.chat import stop_token_ids, turn_end_text


def qwen_like(messages: list[dict[str, str]]) -> str:
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def test_turn_end_is_the_text_after_the_assistant_reply() -> None:
    assert turn_end_text(qwen_like) == "<|im_end|>"


def test_turn_end_needs_something_after_the_reply() -> None:
    with pytest.raises(ValueError, match="nothing after"):
        turn_end_text(lambda messages: "".join(m["content"] for m in messages))


def test_stop_ids_add_the_turn_end_token_to_eos() -> None:
    vocab = {"<|im_end|>": [248046]}
    assert stop_token_ids(248044, "<|im_end|>", vocab.__getitem__) == [248044, 248046]


def test_stop_ids_reject_a_multi_token_turn_end() -> None:
    with pytest.raises(ValueError, match="one token"):
        stop_token_ids(1, "<end>", lambda text: [5, 6])
