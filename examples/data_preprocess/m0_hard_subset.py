"""Select the M0.6 hard subset (decision 005) in two stages.

sample  GPU (Lane B). The selector model samples every candidate problem with the frozen
        suite's prompt and sampling rules, except the selection seed. Writes SampleRecord
        JSONL shards and one run-info line per job under $SOE_DATA/<work_dir>/. A restarted
        job skips problems already in its shard.
select  CPU. Checks the shards, applies the rule and writes the problem file plus a report
        with ids, counts and sha256 values (no problem text, so it can be committed).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import EvalSuite, hash_mismatches, load_suite
from soe.utils.dataset.hard_subset import HardSubsetRule, select_hard_subset
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord, SamplingRunInfo
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.run import gpu_names, library_versions

ASSETS = Path("scripts/stage_assets.yaml")
CHUNK_PROBLEMS = 64  # one generate() call; a crash loses at most this much work


class RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_correct: int
    fallback_max_correct: int
    min_size: int
    target_size: int
    draw_seed: int


class HardSubsetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    suite: Path
    selector: str
    candidate_sets: list[str]
    samples_per_problem: int = Field(ge=1)
    sampling_seed: int
    min_answer_rate: float = Field(ge=0, le=1)
    rule: RuleSpec
    work_dir: str
    output: str
    report: Path


def load_config(path: Path) -> HardSubsetConfig:
    return HardSubsetConfig.model_validate(yaml.safe_load(path.read_text()))


def check_against_suite(config: HardSubsetConfig, suite: EvalSuite) -> None:
    """The pre-registered constraints of decision 005; raises on any violation."""
    sets = {p.name: p for p in suite.problem_sets}
    unknown = [name for name in config.candidate_sets if name not in sets]
    if unknown:
        raise ValueError(f"candidate sets not in the suite: {unknown}")
    level_c = [n for n in config.candidate_sets if sets[n].level in (Level.C1, Level.C2)]
    if level_c:
        raise ValueError(f"Level C is never used for selection: {level_c}")
    if config.sampling_seed == suite.sampling.seed:
        raise ValueError("the selection seed must differ from the evaluation seed")
    if config.samples_per_problem != suite.main.samples_per_problem:
        raise ValueError("selection uses the main tier's samples per problem")


def pinned_revision(hf_id: str) -> str:
    assets = yaml.safe_load(ASSETS.read_text())["assets"]
    revision: str = next(a["revision"] for a in assets if a["hf_id"] == hf_id)
    return revision


def candidates(
    config: HardSubsetConfig, suite: EvalSuite, data_root: Path
) -> tuple[list[Problem], dict[str, str]]:
    """Candidate problems sorted by id, and the problem set each came from (the draw's strata;
    Problem.source is not enough: Level B problems have source "dapo")."""
    sets = {p.name: p for p in suite.problem_sets}
    problems: list[Problem] = []
    problem_set: dict[str, str] = {}
    for name in config.candidate_sets:
        for problem in read_jsonl(data_root / sets[name].path, Problem):
            problems.append(problem)
            problem_set[problem.problem_id] = name
    if len(problem_set) != len(problems):
        raise ValueError("problem ids must be unique across the candidate sets")
    return sorted(problems, key=lambda p: p.problem_id), problem_set


def shard_path(work: Path, shard: int, num_shards: int) -> Path:
    return work / f"samples-{shard}-of-{num_shards}.jsonl"


def run_sample(config: HardSubsetConfig, shard: int, num_shards: int, data_root: Path) -> None:
    from soe.workers.rollout.vllm_sampler import VllmSampler  # GPU stack, image only

    suite = load_suite(config.suite)
    check_against_suite(config, suite)
    bad = hash_mismatches(suite, data_root)
    if bad:
        raise SystemExit(f"refusing to sample: problem sets changed or missing: {bad}")
    work = data_root / config.work_dir
    out = shard_path(work, shard, num_shards)
    mine = candidates(config, suite, data_root)[0][shard::num_shards]
    done = Counter(r.problem_id for r in read_jsonl(out, SampleRecord)) if out.exists() else {}
    partial = [pid for pid, n in done.items() if n != config.samples_per_problem]
    if partial:
        raise SystemExit(f"{out} has incomplete problems {partial[:5]}; inspect before resuming")
    todo = [p for p in mine if p.problem_id not in done]
    print(f"shard {shard}/{num_shards}: {len(mine)} problems, {len(todo)} to sample", flush=True)

    revision = pinned_revision(config.selector)
    sampling = suite.sampling.model_copy(update={"seed": config.sampling_seed})
    sampler = VllmSampler(config.selector, revision=revision, max_tokens=sampling.max_tokens)
    info = SamplingRunInfo(
        checkpoint=config.selector,
        revision=revision,
        suite_name=suite.name,
        problem_sets=config.candidate_sets,
        samples_per_problem=config.samples_per_problem,
        seed=sampling.seed,
        chat_template_kwargs=suite.prompt.chat_template_kwargs(),
        stop_token_ids=sampler.stop_token_ids,
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpus=gpu_names(),
        library_versions=library_versions(),
    )
    write_jsonl(work / f"runinfo-{shard}-of-{num_shards}.jsonl", [info], append=True)
    print("rendered prompt:", repr(sampler.render(todo[0].question if todo else "", suite.prompt)))

    for start in range(0, len(todo), CHUNK_PROBLEMS):
        chunk = todo[start : start + CHUNK_PROBLEMS]
        began = time.monotonic()
        records = sampler.sample(
            chunk,
            prompt=suite.prompt,
            sampling=sampling,
            n=config.samples_per_problem,
            checkpoint=config.selector,
        )
        write_jsonl(out, records, append=True)
        tokens = sum(r.completion_tokens for r in records)
        print(
            f"{start + len(chunk)}/{len(todo)} problems; {tokens / (time.monotonic() - began):.0f}"
            f" tok/s; truncated {sum(r.finish_reason == 'length' for r in records)}/{len(records)}",
            flush=True,
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_stats(
    records: list[SampleRecord], correct: dict[str, int], rule: RuleSpec
) -> dict[str, Any]:
    histogram = Counter(correct.values())
    return {
        "problems": len(correct),
        "answer_rate": round(
            sum(r.extracted_answer is not None for r in records) / len(records), 4
        ),
        "truncation_rate": round(
            sum(r.finish_reason == "length" for r in records) / len(records), 4
        ),
        "pass_at_1": round(sum(r.correct for r in records) / len(records), 4),
        "mean_completion_tokens": round(sum(r.completion_tokens for r in records) / len(records)),
        "eligible_at_max_correct": sum(c <= rule.max_correct for c in correct.values()),
        "eligible_at_fallback": sum(c <= rule.fallback_max_correct for c in correct.values()),
        "correct_histogram": {str(c): histogram[c] for c in sorted(histogram)},
    }


def run_select(config: HardSubsetConfig, data_root: Path) -> int:
    suite = load_suite(config.suite)
    check_against_suite(config, suite)
    work = data_root / config.work_dir
    shards = sorted(work.glob("samples-*-of-*.jsonl"))
    infos = [
        i for p in sorted(work.glob("runinfo-*.jsonl")) for i in read_jsonl(p, SamplingRunInfo)
    ]
    if {i.seed for i in infos} != {config.sampling_seed}:
        raise SystemExit(f"selection samples used seeds {sorted({i.seed for i in infos})}")
    records = [r for path in shards for r in read_jsonl(path, SampleRecord)]
    by_problem: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_problem[record.problem_id].append(record)
    problems, stratum = candidates(config, suite, data_root)
    wanted = list(range(config.samples_per_problem))
    incomplete = [
        p.problem_id
        for p in problems
        if sorted(r.sample_index for r in by_problem.get(p.problem_id, [])) != wanted
    ]
    if incomplete or len(by_problem) != len(problems):
        raise SystemExit(f"{len(incomplete)} candidates lack exactly {wanted[-1] + 1} samples")

    correct = {pid: sum(r.correct for r in rs) for pid, rs in by_problem.items()}
    per_set = {}
    for name in config.candidate_sets:
        ids = [pid for pid in correct if stratum[pid] == name]
        per_set[name] = set_stats(
            [r for pid in ids for r in by_problem[pid]],
            {pid: correct[pid] for pid in ids},
            config.rule,
        )
    overall = set_stats(records, correct, config.rule)
    report: dict[str, Any] = {
        "version": config.version,
        "selector": {"hf_id": config.selector, "revision": pinned_revision(config.selector)},
        "sampling_seed": config.sampling_seed,
        "samples_per_problem": config.samples_per_problem,
        "rule": config.rule.model_dump(),
        "git_commits": sorted({i.git_commit for i in infos}),
        "gpus": sorted({g for i in infos for g in i.gpus}),
        "chat_template_kwargs": infos[0].chat_template_kwargs,
        "stop_token_ids": infos[0].stop_token_ids,
        "sample_files_sha256": {p.name: file_sha256(p) for p in shards},
        "overall": overall,
        "per_set": per_set,
    }
    halted = overall["answer_rate"] < config.min_answer_rate
    if not halted:
        rule = HardSubsetRule(**config.rule.model_dump())
        subset = select_hard_subset(correct, stratum, rule)
        chosen = set(subset.problem_ids)
        out = data_root / config.output
        write_jsonl(out, [p for p in problems if p.problem_id in chosen])
        selected_by_set = Counter(stratum[pid] for pid in chosen)
        for name in config.candidate_sets:
            per_set[name]["selected"] = selected_by_set[name]
        report |= {
            "max_correct_used": subset.max_correct_used,
            "eligible": subset.eligible,
            "selected": len(chosen),
            "output": config.output,
            "output_sha256": file_sha256(out),
            "selected_problem_ids": subset.problem_ids,
        }
    report |= {"halted_low_answer_rate": halted, "correct_counts": dict(sorted(correct.items()))}
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in report
                if k not in ("correct_counts", "selected_problem_ids", "per_set")
            },
            indent=1,
        )
    )
    for name, stats in per_set.items():
        print(name, {k: v for k, v in stats.items() if k != "correct_histogram"})
    if halted:
        print(
            f"HALT: answer rate {overall['answer_rate']} < {config.min_answer_rate} (decision 005)"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    stages = parser.add_subparsers(dest="stage", required=True)
    sample = stages.add_parser("sample")
    sample.add_argument("--shard", type=int, required=True)
    sample.add_argument("--num-shards", type=int, required=True)
    stages.add_parser("select")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(os.environ["SOE_DATA"])
    if args.stage == "sample":
        run_sample(config, args.shard, args.num_shards, data_root)
        return 0
    return run_select(config, data_root)


if __name__ == "__main__":
    sys.exit(main())
