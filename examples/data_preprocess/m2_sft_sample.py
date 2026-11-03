"""M2.4/M2.5: sample the SFT pool with Qwen3.5-4B-Base (roadmap D1 (c), D4), then measure yield.

sample  GPU (Lane B, any 1-GPU type). Draws problems `[draw_start, draw_stop)` of a seeded
        permutation of the clean pool, so the pilot (the first 1,000) and the full run (the next
        ones) never overlap and the full run reuses the pilot's samples. Samples each problem
        `samples_per_problem` times with the frozen suite's prompt, thinking mode, temperature,
        top-p, FP8 KV cache and 16K cap, and the config's seed. Shards are claimed like
        main_eval's; SampleRecord JSONL and run info go under $SOE_DATA/<work_dir>/.
stats   CPU. Yield of `kept_completions` (soe/trainer/sft/data.py) per problem, with a bootstrap
        interval, plus accuracy, truncation, format rates, kept lengths and yield by difficulty.
        Ids and numbers only.

    SBATCH_GRES=gpu:a30:1 sbatch --time=04:00:00 --job-name=soe-sft-pilot slurm/lane_b.sbatch \
        python examples/data_preprocess/m2_sft_sample.py --config <yaml> sample
    sbatch slurm/cpu.sbatch python examples/data_preprocess/m2_sft_sample.py --config <yaml> stats
"""

import argparse
import json
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from soe.trainer.config.eval_suite import file_sha256, load_suite
from soe.trainer.sft.data import THINK_END, kept_completions
from soe.trainer.sft.sampling import SampleConfig, drawn, pinned_revision, read_samples
from soe.utils.dataset.records import SampleRecord, SamplingRunInfo
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.run import gpu_names, library_versions
from soe.utils.shards import claim_shard, mark_done, shard_path

CHUNK_PROBLEMS = 512  # x 4 samples = 2,048 sequences per vLLM call, as main_eval
LENGTH_BINS = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
BOOTSTRAP_DRAWS = 2000


def run_sample(config: SampleConfig, shard: int | None, data_root: Path) -> None:
    from soe.workers.rollout.vllm_sampler import EngineSettings, VllmSampler  # GPU image only

    suite = load_suite(config.suite)
    sampling = suite.sampling.model_copy(update={"seed": config.sampling_seed})
    problems = drawn(config, data_root)
    revision = pinned_revision(config.model)
    engine = EngineSettings()
    sampler = VllmSampler(
        config.model,
        revision=revision,
        max_tokens=sampling.max_tokens,
        engine=engine,
        kv_cache_dtype=sampling.kv_cache_dtype,
    )
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    info = SamplingRunInfo(
        checkpoint=config.model,
        revision=revision,
        suite_name=suite.name,
        problem_sets=[config.problems],
        samples_per_problem=config.samples_per_problem,
        seed=sampling.seed,
        chat_template_kwargs=suite.prompt.chat_template_kwargs(),
        stop_token_ids=sampler.stop_token_ids,
        engine=asdict(engine),
        git_commit=git.stdout.strip(),
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpus=gpu_names(),
        library_versions=library_versions(),
    )
    print("rendered prompt:", repr(sampler.render("<Q>", suite.prompt)), flush=True)
    work = data_root / config.work_dir
    owner = f"job {info.slurm_job_id} on {info.hostname} {info.gpus}"
    for k in [shard] if shard is not None else range(config.num_shards):
        if shard is None and not claim_shard(work, k, owner):
            continue
        out = shard_path(work, k, config.num_shards)
        done: dict[str, int] = defaultdict(int)
        for record in read_jsonl(out, SampleRecord) if out.exists() else []:
            done[record.problem_id] += 1
        if any(n != config.samples_per_problem for n in done.values()):
            raise SystemExit(f"{out} has incomplete problems; inspect before resuming")
        todo = [p for p in problems[k :: config.num_shards] if p.problem_id not in done]
        for start in range(0, len(todo), CHUNK_PROBLEMS):
            chunk = todo[start : start + CHUNK_PROBLEMS]
            began = time.monotonic()
            records = sampler.sample(
                chunk,
                prompt=suite.prompt,
                sampling=sampling,
                n=config.samples_per_problem,
                checkpoint=config.model,
            )
            write_jsonl(out, records, append=True)
            tokens = sum(r.completion_tokens for r in records)
            print(
                f"shard {k}: {start + len(chunk)}/{len(todo)} problems; "
                f"{tokens / (time.monotonic() - began):.0f} tok/s",
                flush=True,
            )
        write_jsonl(work / f"runinfo-{k}-of-{config.num_shards}.jsonl", [info], append=True)
        mark_done(work, k)
    print("no shards left", flush=True)


def bootstrap_mean(values: list[int], seed: int) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling problems."""
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_DRAWS))
    return means[int(0.025 * BOOTSTRAP_DRAWS)], means[int(0.975 * BOOTSTRAP_DRAWS) - 1]


def yield_stats(
    by_problem: dict[str, list[SampleRecord]],
    difficulty: dict[str, float],
    max_kept: int,
    seed: int,
) -> dict[str, Any]:
    records = [r for rs in by_problem.values() for r in rs]
    stopped = [r for r in records if r.finish_reason == "stop"]
    kept = {pid: kept_completions(rs, max_kept) for pid, rs in by_problem.items()}
    per_problem = [len(k) for k in kept.values()]
    low, high = bootstrap_mean(per_problem, seed)
    lengths = [r.completion_tokens for ks in kept.values() for r in ks]
    by_difficulty: dict[str, list[int]] = defaultdict(list)
    for pid, ks in kept.items():
        by_difficulty[f"{difficulty[pid]:g}"].append(len(ks))
    return {
        "problems": len(by_problem),
        "samples": len(records),
        "accuracy": mean(r.correct for r in records),
        "truncation_rate": 1 - len(stopped) / len(records),
        "stopped_boxed_rate": mean(r.extracted_answer is not None for r in stopped),
        "stopped_one_think_end_rate": mean(r.completion.count(THINK_END) == 1 for r in stopped),
        "kept_per_problem": {"mean": mean(per_problem), "ci95": [low, high]},
        "problems_with_a_kept": sum(n > 0 for n in per_problem),
        "kept": len(lengths),
        "kept_tokens": {
            "mean": mean(lengths) if lengths else 0.0,
            "histogram_upper_edges": {
                str(edge): sum(prev < n <= edge for n in lengths)
                for prev, edge in zip([0, *LENGTH_BINS], LENGTH_BINS, strict=False)
            },
        },
        "kept_per_problem_by_difficulty": {
            d: {"problems": len(v), "mean": mean(v)}
            for d, v in sorted(by_difficulty.items(), key=lambda kv: float(kv[0]))
        },
    }


def run_stats(config: SampleConfig, data_root: Path, out: Path) -> None:
    work = data_root / config.work_dir
    by_problem = read_samples(config, data_root)
    lines = (data_root / config.meta).read_text().splitlines()
    difficulty = {m["problem_id"]: float(m["difficulty"]) for m in map(json.loads, lines)}
    stats = yield_stats(by_problem, difficulty, config.max_kept_per_problem, config.draw_seed)
    report = {
        "version": config.version,
        "config": config.model_dump(mode="json"),
        "shard_files_sha256": {
            f.name: file_sha256(f)
            for f in (shard_path(work, k, config.num_shards) for k in range(config.num_shards))
        },
        **stats,
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if "by_difficulty" not in k}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("stage", choices=("sample", "stats"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--out", type=Path, help="stats report path")
    args = parser.parse_args()
    config = SampleConfig.model_validate(yaml.safe_load(args.config.read_text()))
    data_root = Path(os.environ["SOE_DATA"])
    if args.stage == "sample":
        run_sample(config, args.shard, data_root)
    else:
        if args.out is None:
            raise SystemExit("stats needs --out")
        run_stats(config, data_root, args.out)


if __name__ == "__main__":
    main()
