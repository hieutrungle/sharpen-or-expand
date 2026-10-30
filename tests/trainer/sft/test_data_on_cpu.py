from soe.trainer.sft.data import TokenizedExample, fitting, tokenize_example

TURN_END = 99


def encode(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_only_completion_and_turn_end_carry_loss() -> None:
    example = tokenize_example("ab", "xyz", encode=encode, turn_end_id=TURN_END)
    assert example.input_ids == [97, 98, 120, 121, 122, TURN_END]
    assert example.completion_mask == [0, 0, 1, 1, 1, 1]


def test_prompt_tokens_are_the_prompt_encoded_alone() -> None:
    # A tokenizer that would merge across the boundary ("b" + "x" -> one token) must not move a
    # token between prompt and completion: the two are encoded apart.
    def merging(text: str) -> list[int]:
        return [1000] if text == "bx" else encode(text)

    example = tokenize_example("b", "x", encode=merging, turn_end_id=TURN_END)
    assert example.input_ids[:1] == merging("b") and example.completion_mask == [0, 1, 1]


def test_long_examples_are_dropped_and_counted() -> None:
    short = TokenizedExample(input_ids=[1, 2], completion_mask=[0, 1])
    long = TokenizedExample(input_ids=[1, 2, 3], completion_mask=[0, 1, 1])
    assert fitting([short, long, short], max_length=2) == ([short, short], 1)
