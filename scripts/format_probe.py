"""Compare chat-template thinking modes on RL-train problems before fixing one in the suite.

Qwen3.5's generation prompt ends with an empty think block by default and an open "<think>\\n" with
enable_thinking=true. This samples a seeded draw of RL-train problems (never evaluation data) in
both modes with the suite's sampling rules and prints answer rate, truncation, accuracy and length.
Lane B only:
    sbatch slurm/lane_b.sbatch python scripts/format_probe.py --model Qwen/Qwen3.5-4B-Base
"""

import argparse
import json
import os
import random
import statistics
from pathlib import Path

import yaml

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.workers.rollout.vllm_sampler import VllmSampler

CHUNK_PROBLEMS = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--problems", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1, help="vLLM seed; 0 is the evaluation seed")
    parser.add_argument("--draw-seed", type=int, default=20261026)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["false", "true"],
        default=["false", "true"],
        help="enable_thinking values to sample",
    )
    args = parser.parse_args()

    data_root = Path(os.environ["SOE_DATA"])
    suite = load_suite(args.suite)
    pool = read_jsonl(data_root / "datasets/soe/v1/rl/train.jsonl", Problem)
    problems = random.Random(args.draw_seed).sample(
        sorted(pool, key=lambda p: p.problem_id), args.problems
    )
    assets = yaml.safe_load(Path("scripts/stage_assets.yaml").read_text())["assets"]
    revision = next(a["revision"] for a in assets if a["hf_id"] == args.model)
    sampling = suite.sampling.model_copy(update={"seed": args.seed})
    sampler = VllmSampler(args.model, revision=revision, max_tokens=sampling.max_tokens)
    print("stop token ids:", sampler.stop_token_ids, flush=True)
    out_dir = data_root / "probes" / f"format_{args.model.replace('/', '_')}"

    modes = {f"enable_thinking={m == 'true'}": m == "true" for m in args.modes}
    prompts = {k: suite.prompt.model_copy(update={"enable_thinking": m}) for k, m in modes.items()}
    for key, prompt in prompts.items():
        print(f"{key} prompt:", repr(sampler.render("<Q>", prompt)))
        (out_dir / f"{key}.jsonl").unlink(missing_ok=True)
    seen: dict[str, list[SampleRecord]] = {key: [] for key in modes}
    # Both modes per chunk, so a time limit still leaves a paired comparison on disk.
    for start in range(0, len(problems), CHUNK_PROBLEMS):
        chunk = problems[start : start + CHUNK_PROBLEMS]
        for key, prompt in prompts.items():
            records = sampler.sample(
                chunk, prompt=prompt, sampling=sampling, n=args.samples, checkpoint=args.model
            )
            write_jsonl(out_dir / f"{key}.jsonl", records, append=True)
            if not seen[key]:
                first = records[0]
                print(f"--- {key} {first.problem_id} correct={first.correct}")
                print(first.completion[:700], "\n[...]\n", first.completion[-300:])
            seen[key] += records
        summary = {key: summarize(records) for key, records in seen.items()}
        print(f"after {start + len(chunk)} problems:", json.dumps(summary), flush=True)
        (out_dir / f"summary_{'_'.join(args.modes)}.json").write_text(json.dumps(summary, indent=1))


def summarize(records: list[SampleRecord]) -> dict[str, float]:
    tokens = [r.completion_tokens for r in records]
    n = len(records)
    return {
        "samples": n,
        "answer_rate": round(sum(r.extracted_answer is not None for r in records) / n, 3),
        "truncation_rate": round(sum(r.finish_reason == "length" for r in records) / n, 3),
        "accuracy": round(sum(r.correct for r in records) / n, 3),
        "mean_tokens": round(statistics.mean(tokens)),
        "median_tokens": round(statistics.median(tokens)),
        "closes_think": round(sum("</think>" in r.completion for r in records) / n, 3),
    }


if __name__ == "__main__":
    main()
