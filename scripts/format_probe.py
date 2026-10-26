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
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.workers.rollout.vllm_sampler import VllmSampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--problems", type=int, default=200)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1, help="vLLM seed; 0 is the evaluation seed")
    parser.add_argument("--draw-seed", type=int, default=20261026)
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
    print("stop token ids:", sampler.stop_token_ids)
    out_dir = data_root / "probes" / f"format_{args.model.replace('/', '_')}"

    summary = {}
    for enable_thinking in (False, True):
        prompt = suite.prompt.model_copy(update={"enable_thinking": enable_thinking})
        print(
            f"\n=== enable_thinking={enable_thinking}\nprompt:", repr(sampler.render("<Q>", prompt))
        )
        records = sampler.sample(
            problems, prompt=prompt, sampling=sampling, n=args.samples, checkpoint=args.model
        )
        write_jsonl(out_dir / f"thinking_{enable_thinking}.jsonl", records)
        tokens = [r.completion_tokens for r in records]
        summary[f"enable_thinking={enable_thinking}"] = {
            "samples": len(records),
            "answer_rate": round(
                sum(r.extracted_answer is not None for r in records) / len(records), 3
            ),
            "truncation_rate": round(
                sum(r.finish_reason == "length" for r in records) / len(records), 3
            ),
            "accuracy": round(sum(r.correct for r in records) / len(records), 3),
            "mean_tokens": round(statistics.mean(tokens)),
            "median_tokens": round(statistics.median(tokens)),
            "closes_think": round(
                sum("</think>" in r.completion for r in records) / len(records), 3
            ),
        }
        for r in records[
            : 2 * args.samples : args.samples
        ]:  # first sample of the first two problems
            print(
                f"--- {r.problem_id} correct={r.correct} finish={r.finish_reason}",
                f"tokens={r.completion_tokens}",
            )
            print(r.completion[:700], "\n[...]\n", r.completion[-300:])
    print("\nSUMMARY", json.dumps(summary, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
