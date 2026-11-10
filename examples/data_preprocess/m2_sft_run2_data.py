"""M2.10 run #2 data (decision 029 item 6, criterion 3's change): the SFT train file without the
examples above the 90th percentile of train token lengths. `val.jsonl` is unchanged.

    sbatch slurm/cpu.sbatch python examples/data_preprocess/m2_sft_run2_data.py \\
        --config examples/data_preprocess/m2_sft_run2_data.yaml

Lengths are counted as main_sft tokenizes (chat template, completion, turn end). The percentile is
the nearest rank: the length at rank ceil(0.9 N) of the sorted lengths; an example longer than it
is dropped, one equal to it is kept. Lines are copied unchanged, in their original order. Container
only (transformers). The report holds counts, tokens, the cut and sha256s, no text.
"""

import argparse
import json
import math
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import file_sha256, load_suite
from soe.trainer.main_rl import checked_data_file
from soe.trainer.sft.data import SftExample
from soe.trainer.sft.sampling import pinned_revision
from soe.utils.jsonl import read_jsonl, write_jsonl

SHA256 = r"^[0-9a-f]{64}$"


class Run2DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: str  # run #1's train file, relative to $SOE_DATA
    input_sha256: str = Field(pattern=SHA256)
    model: str
    suite: Path
    percentile: float = Field(gt=0, lt=100)
    output: str  # relative to $SOE_DATA
    report: Path  # repo path


def cut_length(lengths: list[int], percentile: float) -> int:
    """Nearest-rank percentile: the length at rank ceil(p/100 * N) of the sorted lengths."""
    ordered = sorted(lengths)
    return ordered[math.ceil(percentile / 100 * len(ordered)) - 1]


def main() -> None:
    from huggingface_hub import snapshot_download  # offline, from the staged snapshot

    from soe.trainer.sft.trainer import Tokenizer  # container only (transformers)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = Run2DataConfig.model_validate(yaml.safe_load(parser.parse_args().config.read_text()))
    root = Path(os.environ["SOE_DATA"])
    examples = read_jsonl(checked_data_file(root, config.input, config.input_sha256), SftExample)
    base = snapshot_download(
        config.model, revision=pinned_revision(config.model), local_files_only=True
    )
    tokenizer = Tokenizer(Path(base), load_suite(config.suite).prompt)
    tokenized = [tokenizer.tokenize(e) for e in examples]
    lengths = [len(t.input_ids) for t in tokenized]
    cut = cut_length(lengths, config.percentile)
    kept = [i for i, n in enumerate(lengths) if n <= cut]
    write_jsonl(root / config.output, [examples[i] for i in kept])
    report = {
        "input": config.input,
        "input_sha256": config.input_sha256,
        "percentile": config.percentile,
        "cut_tokens": cut,
        "examples": {
            "input": len(examples),
            "kept": len(kept),
            "dropped": len(examples) - len(kept),
        },
        "problems": {
            "input": len({e.problem_id for e in examples}),
            "kept": len({examples[i].problem_id for i in kept}),
        },
        "tokens": {"input": sum(lengths), "kept": sum(lengths[i] for i in kept)},
        "loss_tokens_kept": sum(sum(tokenized[i].completion_mask) for i in kept),
        "max_tokens_kept": max(lengths[i] for i in kept),
        "output": config.output,
        "output_sha256": file_sha256(root / config.output),
    }
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
