"""Steady-state vLLM throughput for one engine setting, before a full evaluation (input to G1).

Samples a seeded draw of RL-train problems (never evaluation data) with the frozen suite's prompt
and sampling (seed 1) in one large call, with vLLM's periodic stats on: generation tokens/s,
running/waiting requests, KV-cache use, preemptions. Stop it with an outer `timeout`; the log is
the result. One job per setting, side by side:
    sbatch --time=00:40:00 slurm/lane_b.sbatch timeout 1500 python scripts/throughput_probe.py \\
        --model Qwen/Qwen3.5-4B-Base --gpu-memory-utilization 0.95 --max-num-seqs 512
"""

import argparse
import os
import random
from pathlib import Path

import yaml

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Problem
from soe.utils.jsonl import read_jsonl
from soe.workers.rollout.vllm_sampler import EngineSettings, VllmSampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", type=Path, default=Path("soe/trainer/config/eval_suite.yaml"))
    parser.add_argument("--problems", type=int, default=256)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--draw-seed", type=int, default=20261026)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    pool = read_jsonl(Path(os.environ["SOE_DATA"]) / "datasets/soe/v1/rl/train.jsonl", Problem)
    problems = random.Random(args.draw_seed).sample(
        sorted(pool, key=lambda p: p.problem_id), args.problems
    )
    assets = yaml.safe_load(Path("scripts/stage_assets.yaml").read_text())["assets"]
    revision = next(a["revision"] for a in assets if a["hf_id"] == args.model)
    engine = EngineSettings(
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        log_stats=True,
    )
    sampler = VllmSampler(
        args.model, revision=revision, max_tokens=suite.sampling.max_tokens, engine=engine
    )
    try:  # what the scheduler resolved; private API, so only printed for the log
        scheduler = sampler._llm.llm_engine.vllm_config.scheduler_config
        print(f"PROBE scheduler max_num_seqs={scheduler.max_num_seqs}", flush=True)
        print(f"PROBE scheduler max_num_batched_tokens={scheduler.max_num_batched_tokens}")
    except AttributeError as e:
        print("PROBE scheduler config not readable:", e)
    print(f"PROBE engine {engine} problems={args.problems} samples={args.samples}", flush=True)
    sampling = suite.sampling.model_copy(update={"seed": 1})
    sampler.sample(
        problems, prompt=suite.prompt, sampling=sampling, n=args.samples, checkpoint=args.model
    )
    print("PROBE finished before the timeout", flush=True)


if __name__ == "__main__":
    main()
