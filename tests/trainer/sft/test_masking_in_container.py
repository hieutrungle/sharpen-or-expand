"""With the real 2B tokenizer and TRL 0.27's collator, only the completion and the turn end carry
loss (M1.8 acceptance: prompt tokens have label -100). Image only; needs the staged 2B snapshot."""

import os
from pathlib import Path

import pytest
import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from soe.trainer.config.eval_suite import load_suite
from soe.trainer.config.experiment import load_config
from soe.trainer.main_rl import model_path
from soe.trainer.sft.data import SftExample
from soe.trainer.sft.trainer import Tokenizer

CONFIG = Path("examples/sft/m1_dev_qwen35-2b.yaml")
IM_END = 248046  # <|im_end|>, the turn end every sampler stops on (decision 006)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    if "SOE_DATA" not in os.environ:
        pytest.skip("needs $SOE_DATA with the staged 2B snapshot")
    config = load_config(CONFIG)
    suite = load_suite(Path(config.eval_suite))
    return Tokenizer(model_path(config, Path(os.environ["SOE_DATA"])), suite.prompt)


def test_prompt_tokens_get_label_minus_100(tokenizer: Tokenizer) -> None:
    suite = load_suite(Path(load_config(CONFIG).eval_suite))
    example = SftExample(
        problem_id="p",
        user=suite.prompt.render("What is 1 + 1?"),
        completion="One and one make two.\n</think>\n\nThe answer is \\boxed{2}.",
    )
    rendered = tokenizer.render(example.user)
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")  # enable_thinking (decision 006)
    tokenized = tokenizer.tokenize(example)
    n_prompt = tokenized.completion_mask.index(1)
    assert tokenized.input_ids[:n_prompt] == tokenizer.encode(rendered)

    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.hf.pad_token_id)
    batch = collator(
        [{"input_ids": tokenized.input_ids, "completion_mask": tokenized.completion_mask}]
    )
    labels = batch["labels"][0]
    ids = torch.tensor(tokenized.input_ids)
    assert (labels[:n_prompt] == -100).all()
    assert torch.equal(labels[n_prompt:], ids[n_prompt:])
    assert int(labels[-1]) == IM_END == tokenizer.turn_end_id
    completion = tokenizer.hf.decode(ids[n_prompt:-1])
    assert completion == example.completion
