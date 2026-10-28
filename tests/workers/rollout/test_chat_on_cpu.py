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


def test_truncated_means_not_ended_on_a_stop_token() -> None:
    from soe.workers.rollout.chat import is_truncated

    stops = [248044, 248046]
    assert not is_truncated([1, 2, 248046], stops)
    assert not is_truncated([1, 248044], stops)
    assert is_truncated([1, 2, 3], stops)
    assert is_truncated([], stops)


def test_flat_cache_tensors_unpacks_hybrid_layer_state_lists() -> None:
    from soe.workers.rollout.chat import flat_cache_tensors

    full, s1, s2, s3 = object(), object(), object(), object()
    assert flat_cache_tensors([None, full, [s1, s2], None, (s3,)]) == [full, s1, s2, s3]
    assert flat_cache_tensors([]) == []
