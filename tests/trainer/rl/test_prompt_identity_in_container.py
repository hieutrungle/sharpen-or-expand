"""verl's rollout prompt ids equal the evaluation sampler's (decision 031); image and $SOE_DATA.

Starting RL from the base model relies on the prompt alone for the answer format (thinking prefill,
boxed answer), so a verl, transformers or template change that alters the rollout prompt would
change what every algorithm learns from without any error. verl 0.9.1 tokenizes rollout prompts in
its agent loop with a continuous-token builder; for Qwen3.5 it is the VL builder over the
processor, whose chat template verl's HFModelConfig copies from the tokenizer for base models.

    T=tests/trainer/rl/test_prompt_identity_in_container.py
    sbatch slurm/cpu.sbatch python -m pytest -q $T
"""

import os
from pathlib import Path
from typing import Any

import pytest

from soe.trainer.config.eval_suite import PromptSpec, load_suite
from soe.trainer.config.experiment import load_config
from soe.trainer.main_rl import model_path

verl = pytest.importorskip("verl")

CONFIG = Path("examples/grpo_trainer/m1_throughput_bf0_qwen35-4b.yaml")  # 4B, init_checkpoint null
PREFILL = "<|im_start|>assistant\n<think>\n"
QUESTIONS = [
    "What is 3+4?",
    "Let $f(x)=\\frac{x^2}{1+x}$; find $\\lim_{x\\to\\infty} f(x)/x$.\n\nAs a fraction.",
    "Évaluez $\\sum_{k=1}^{n} k^3$ — en fonction de $n$. 数学 ∫₀¹ x dx",
    "  Leading spaces, trailing newline\n",
]


@pytest.fixture(scope="module")
def setup() -> tuple[Any, Any, PromptSpec, Path]:
    if "SOE_DATA" not in os.environ:
        pytest.skip("needs $SOE_DATA with the staged 4B snapshot")
    from transformers import AutoConfig
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    config = load_config(CONFIG)
    prompt = load_suite(Path(config.eval_suite)).prompt
    model = model_path(config, Path(os.environ["SOE_DATA"]))
    tokenizer, processor = hf_tokenizer(str(model)), hf_processor(str(model))
    if processor is not None and not getattr(processor, "chat_template", None):
        processor.chat_template = tokenizer.chat_template  # as verl's HFModelConfig
    builder = create_continuous_token_builder(
        tokenizer,
        hf_model_type=AutoConfig.from_pretrained(str(model)).model_type,
        chat_template_kwargs=prompt.chat_template_kwargs(),
        mm_processor_kwargs={},
        processor=processor,
    )
    return tokenizer, builder, prompt, Path(os.environ["SOE_DATA"]) / config.data.val_file


def eval_text(tokenizer: Any, prompt: PromptSpec, question: str) -> str:
    """As soe.workers.rollout.vllm_sampler.VllmSampler.render."""
    text: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.render(question)}],
        tokenize=False,
        add_generation_prompt=True,
        **prompt.chat_template_kwargs(),
    )
    return text


@pytest.mark.parametrize("question", QUESTIONS)
def test_rollout_prompt_ids_equal_evaluation_prompt_ids(
    setup: tuple[Any, Any, PromptSpec, Path], question: str
) -> None:
    tokenizer, builder, prompt, _ = setup
    text = eval_text(tokenizer, prompt, question)
    assert text.endswith(PREFILL)
    messages = [{"role": "user", "content": prompt.render(question)}]
    assert list(builder.build_initial_tokens(messages)) == tokenizer.encode(
        text, add_special_tokens=False
    )


def test_every_rl_validation_prompt_matches(setup: tuple[Any, Any, PromptSpec, Path]) -> None:
    import pandas as pd

    tokenizer, builder, prompt, val_file = setup
    for _, row in pd.read_parquet(val_file).iterrows():
        messages = [dict(m) for m in row["prompt"]]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **prompt.chat_template_kwargs()
        )
        assert text.endswith(PREFILL)
        assert list(builder.build_initial_tokens(messages)) == tokenizer.encode(
            text, add_special_tokens=False
        )
