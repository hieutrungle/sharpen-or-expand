"""Worker for test_fsdp_in_container.py: the full-BF16 SFT path (decision 028) on a tiny
Qwen3.5-shaped model, under torchrun with gloo on CPU. Not collected by pytest (no test_ prefix).

    torchrun --standalone --nproc_per_node=2 tests/trainer/sft/fsdp_tiny_worker.py <dir> [resume]

<dir>/base holds the tiny base (written by the test); the run writes <dir>/run and the export
<dir>/export. CPU-only stand-ins, each marked: torch kernels instead of fla/causal-conv1d (Triton
needs a GPU), Liger off (also Triton), so the forward gets no `skip_logits`.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from soe.trainer.config.experiment import ExperimentConfig
from soe.trainer.sft.args import FSDP_ENVIRONMENT, plain_name, trainer_arguments
from soe.trainer.sft.resume import example_key, log_path
from soe.trainer.sft.trainer import SoeSftTrainer, export_full, freeze_vision

for kernel in (
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
):
    setattr(qwen3_5, kernel, None)  # CPU stand-in: the torch implementations


class CpuSftTrainer(SoeSftTrainer):
    def compute_loss(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if model.training:
            self._log_examples(inputs)
        return SFTTrainer.compute_loss(self, model, inputs, *args, **kwargs)  # no Liger on CPU


def config() -> ExperimentConfig:
    import yaml

    data = yaml.safe_load(Path("examples/sft/m2_dev_full_qwen35-2b.yaml").read_text())
    sft = data["sft"] | {"gradient_accumulation_steps": 1, "save_every_steps": 2, "max_steps": 4}
    return ExperimentConfig.model_validate(data | {"sft": sft})


def main() -> None:
    root, resume = Path(sys.argv[1]), len(sys.argv) > 2
    run_dir, base = root / "run", root / "base"
    rank = int(os.environ["RANK"])
    generator = torch.Generator().manual_seed(0)
    rows = [
        torch.randint(10, 200, (int(n),), generator=generator).tolist() for n in range(20, 36, 2)
    ]
    masks = [[0] * (len(r) // 2) + [1] * (len(r) - len(r) // 2) for r in rows]
    train = Dataset.from_dict({"input_ids": rows, "completion_mask": masks})
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32)
    freeze_vision(model)
    os.environ.update(FSDP_ENVIRONMENT)
    cfg = config()
    kwargs = trainer_arguments(cfg, seed=1, output_dir=run_dir / "trainer")
    kwargs |= {"use_liger_kernel": False, "eval_strategy": "no", "use_cpu": True}  # CPU stand-ins
    segment = "resume" if resume else "first"
    trainer = CpuSftTrainer(
        model=model,
        args=SFTConfig(**kwargs),
        train_dataset=train,
        processing_class=AutoTokenizer.from_pretrained(base),
        example_index={example_key(r): i for i, r in enumerate(rows)},
        example_log=log_path(run_dir, segment, rank),
    )
    trainer.accelerator.state.fsdp_plugin.mixed_precision_policy = (
        torch.distributed.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16
        )
    )
    # The resumed launch restarts from the step-2 save and redoes steps 3 and 4.
    checkpoint = run_dir / "trainer" / "checkpoint-2" if resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    state = trainer.accelerator.get_state_dict(trainer.model)
    if trainer.accelerator.is_main_process:
        trained = [plain_name(n) for n, p in trainer.model.named_parameters() if p.requires_grad]
        out = root / f"export-{segment}"  # the real run writes $SOE_DATA/checkpoints/<id>/...
        info = export_full(state, trained, base, out, run_id="tiny", step=4, config=cfg)
        steps = {h["step"]: h for h in trainer.state.log_history if "loss" in h}
        result = {
            "export": json.loads(info.model_dump_json()),
            "modules": sorted({type(m).__name__ for m in trainer.model.modules()}),
            "steps": {s: {"loss": h["loss"], "lr": h["learning_rate"]} for s, h in steps.items()},
        }
        (root / f"result-{segment}.json").write_text(json.dumps(result))
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
