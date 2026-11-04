"""M2.8 worst-case development data: the longest examples of an SFT file, by tokens.

    python examples/data_preprocess/m2_sft_longest.py \\
        --config examples/data_preprocess/m2_sft_longest.yaml

Lines are copied unchanged, longest first, so a short 4B probe trains only on the sequences that
set run #1's peak memory. Lengths are counted as main_sft tokenizes (chat template, turn end).
Container only (transformers); paths under $SOE_DATA except the report (repo path).
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import file_sha256, load_suite
from soe.trainer.main_rl import checked_data_file
from soe.trainer.sft.data import SftExample
from soe.trainer.sft.sampling import pinned_revision
from soe.utils.jsonl import read_jsonl, write_jsonl


class LongestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str
    suite: Path
    count: int = Field(gt=0)
    output: str
    report: Path


def longest(lengths: list[int], count: int) -> list[int]:
    """Indices of the `count` longest examples, longest first (ties keep file order)."""
    return sorted(range(len(lengths)), key=lambda i: -lengths[i])[:count]


def main() -> None:
    from huggingface_hub import snapshot_download  # offline, from the staged snapshot

    from soe.trainer.sft.trainer import Tokenizer  # container only (transformers)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = LongestConfig.model_validate(yaml.safe_load(parser.parse_args().config.read_text()))
    root = Path(os.environ["SOE_DATA"])
    examples = read_jsonl(checked_data_file(root, config.input, config.input_sha256), SftExample)
    base = snapshot_download(
        config.model, revision=pinned_revision(config.model), local_files_only=True
    )
    tokenizer = Tokenizer(Path(base), load_suite(config.suite).prompt)
    lengths = [len(tokenizer.tokenize(e).input_ids) for e in examples]
    chosen = longest(lengths, config.count)
    write_jsonl(root / config.output, [examples[i] for i in chosen])
    report = {
        "input": config.input,
        "output": config.output,
        "output_sha256": file_sha256(root / config.output),
        "count": len(chosen),
        "tokens": {"max": lengths[chosen[0]], "min": lengths[chosen[-1]]},
        "over_16384_in_input": sum(n > 16384 for n in lengths),
        "over_16384_kept": sum(lengths[i] > 16384 for i in chosen),
    }
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
