"""Evaluation entry point: sample the frozen suite for one checkpoint, then build its tables.

    --stage sample  GPU. Checks every problem-file hash, then claims shards of the suite (see
                    soe.utils.shards) and writes SampleRecord JSONL plus run info per shard under
                    --output. Run several jobs at once on any GPU type; `--shard k` finishes one.
    --stage report  CPU. Checks completeness and writes per_problem.jsonl and report.json:
                    pass@k / maj@k per problem set and per level (main tier) and on the hard subset.

Sampling uses only the suite's settings (seed included); nothing here can override them. The output
directory's plan.json pins checkpoint, suite file and sharding, so jobs never mix two evaluations.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import EvalSuite, hash_mismatches, load_suite
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord, SamplingRunInfo
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.metric.eval_tables import ProblemResult, problem_result, tier_table
from soe.utils.reward_score.math_verify import answers_equivalent
from soe.utils.run import gpu_names, library_versions
from soe.utils.shards import claim_shard, mark_done, shard_path, unfinished_shards

ASSETS = Path("scripts/stage_assets.yaml")
# Completions per generate() call: large enough that the few longest completions, which finish
# alone at the end of a call, cost little throughput; small enough that a crash loses little.
SEQUENCES_PER_CALL = 2048


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalUnit:
    problem: Problem
    problem_set: str
    level: Level
    n: int  # samples: the main tier's, or the hard tier's on the hard subset
    hard: bool


class EvalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    suite_name: str
    suite_file_sha256: str
    checkpoint: str
    revision: str | None
    num_shards: int
    limit_per_set: int | None  # smoke tests only; full evaluations have None
    units: int
    samples: int


def eval_units(suite: EvalSuite, data_root: Path, limit_per_set: int | None) -> list[EvalUnit]:
    """Every (problem, number of samples) of the suite, sorted by problem id."""
    hard_ids: set[str] = set()
    if suite.hard_subset is not None:
        hard_ids = {p.problem_id for p in read_jsonl(data_root / suite.hard_subset.path, Problem)}
    units = []
    for problem_set in suite.problem_sets:
        problems = sorted(
            read_jsonl(data_root / problem_set.path, Problem), key=lambda p: p.problem_id
        )
        for problem in problems[:limit_per_set]:
            hard = problem.problem_id in hard_ids
            n = (suite.hard if hard else suite.main).samples_per_problem
            units.append(
                EvalUnit(
                    problem=problem,
                    problem_set=problem_set.name,
                    level=problem_set.level,
                    n=n,
                    hard=hard,
                )
            )
    ids = [u.problem.problem_id for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError("problem ids must be unique across the suite")
    if limit_per_set is None and not hard_ids <= set(ids):
        raise ValueError("every hard-subset problem must belong to a suite problem set")
    return sorted(units, key=lambda u: u.problem.problem_id)


def calls(units: list[EvalUnit]) -> list[list[EvalUnit]]:
    """Group units by sample count (one n per vLLM call), then cut into calls of bounded size."""
    by_n: dict[int, list[EvalUnit]] = defaultdict(list)
    for unit in units:
        by_n[unit.n].append(unit)
    out = []
    for n, group in sorted(by_n.items()):
        size = max(1, SEQUENCES_PER_CALL // n)
        out += [group[i : i + size] for i in range(0, len(group), size)]
    return out


def pinned_revision(checkpoint: str) -> str | None:
    """The stage_assets.yaml revision for a Hub model id; None for a local checkpoint directory."""
    for asset in yaml.safe_load(ASSETS.read_text())["assets"]:
        if asset["hf_id"] == checkpoint:
            revision: str = asset["revision"]
            return revision
    if not Path(checkpoint).is_dir():
        raise ValueError(f"{checkpoint} is neither a staged Hub model nor a local directory")
    return None


def write_or_check_plan(output: Path, plan: EvalPlan) -> None:
    path = output / "plan.json"
    output.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as f:  # exclusive create: the first job writes the plan
            f.write(plan.model_dump_json(indent=1) + "\n")
    except FileExistsError:
        existing = EvalPlan.model_validate_json(path.read_text())
        if existing != plan:
            raise SystemExit(
                f"{path} describes a different evaluation:\n{existing}\nvs\n{plan}"
            ) from None


def make_plan(
    suite_path: Path,
    suite: EvalSuite,
    checkpoint: str,
    units: list[EvalUnit],
    num_shards: int,
    limit_per_set: int | None,
) -> EvalPlan:
    return EvalPlan(
        suite_name=suite.name,
        suite_file_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        checkpoint=checkpoint,
        revision=pinned_revision(checkpoint),
        num_shards=num_shards,
        limit_per_set=limit_per_set,
        units=len(units),
        samples=sum(u.n for u in units),
    )


def run_sample(cli: "EvaluateCli", suite: EvalSuite, units: list[EvalUnit], plan: EvalPlan) -> None:
    from soe.workers.rollout.vllm_sampler import EngineSettings, VllmSampler  # GPU image only

    engine = EngineSettings(
        gpu_memory_utilization=cli.gpu_memory_utilization, max_num_seqs=cli.max_num_seqs
    )
    sampler = VllmSampler(
        cli.checkpoint, revision=plan.revision, max_tokens=suite.sampling.max_tokens, engine=engine
    )
    info = SamplingRunInfo(
        checkpoint=cli.checkpoint,
        revision=plan.revision,
        suite_name=suite.name,
        problem_sets=[p.name for p in suite.problem_sets],
        samples_per_problem=suite.main.samples_per_problem,
        seed=suite.sampling.seed,
        chat_template_kwargs=suite.prompt.chat_template_kwargs(),
        stop_token_ids=sampler.stop_token_ids,
        engine=asdict(engine),
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpus=gpu_names(),
        library_versions=library_versions(),
    )
    print("rendered prompt:", repr(sampler.render("<Q>", suite.prompt)), flush=True)
    owner = f"job {info.slurm_job_id} on {info.hostname} {info.gpus}"
    shards = [cli.shard] if cli.shard is not None else range(plan.num_shards)
    for k in shards:  # claim lazily, one shard at a time, so faster GPUs take more shards
        if cli.shard is None and not claim_shard(cli.output, k, owner):
            continue
        out = shard_path(cli.output, k, plan.num_shards)
        done: dict[str, int] = defaultdict(int)
        for record in read_jsonl(out, SampleRecord) if out.exists() else []:
            done[record.problem_id] += 1
        mine = units[k :: plan.num_shards]
        partial = [u.problem.problem_id for u in mine if 0 < done[u.problem.problem_id] != u.n]
        if partial:
            raise SystemExit(f"{out}: incomplete problems {partial[:5]}; inspect before resuming")
        todo = [u for u in mine if done[u.problem.problem_id] == 0]
        print(
            f"shard {k}/{plan.num_shards}: {len(mine)} problems, {len(todo)} to sample", flush=True
        )
        for call in calls(todo):
            began = time.monotonic()
            records = sampler.sample(
                [u.problem for u in call],
                prompt=suite.prompt,
                sampling=suite.sampling,
                n=call[0].n,
                checkpoint=cli.checkpoint,
            )
            write_jsonl(out, records, append=True)
            tokens = sum(r.completion_tokens for r in records)
            print(
                f"shard {k}: {len(call)} problems x {call[0].n}; "
                f"{tokens / (time.monotonic() - began):.0f} tok/s; "
                f"truncated {sum(r.finish_reason == 'length' for r in records)}/{len(records)}",
                flush=True,
            )
        write_jsonl(cli.output / f"runinfo-{k}-of-{plan.num_shards}.jsonl", [info], append=True)
        mark_done(cli.output, k)
    print("no shards left", flush=True)


def group_by(
    results: list[ProblemResult], key: Callable[[ProblemResult], str]
) -> dict[str, list[ProblemResult]]:
    grouped: dict[str, list[ProblemResult]] = defaultdict(list)
    for result in results:
        grouped[key(result)].append(result)
    return dict(sorted(grouped.items()))


@lru_cache(maxsize=1 << 20)
def same_answer(a: str, b: str) -> bool:
    """Verifier equivalence for maj@k grouping (decision 004); cached, answers repeat a lot."""
    return answers_equivalent(a, b)


def run_report(cli: "EvaluateCli", suite: EvalSuite, units: list[EvalUnit], plan: EvalPlan) -> int:
    left = unfinished_shards(cli.output, plan.num_shards)
    if left:
        print(f"not finished: shards {left}")
        return 1
    by_problem: dict[str, list[SampleRecord]] = defaultdict(list)
    for k in range(plan.num_shards):
        for record in read_jsonl(shard_path(cli.output, k, plan.num_shards), SampleRecord):
            by_problem[record.problem_id].append(record)
    results = []
    for unit in units:
        records = by_problem.pop(unit.problem.problem_id, [])
        if len(records) != unit.n:
            raise SystemExit(
                f"{unit.problem.problem_id}: {len(records)} samples, expected {unit.n}"
            )
        ks = suite.hard.ks if unit.hard else suite.main.ks
        results.append(
            problem_result(
                records,
                problem_set=unit.problem_set,
                level=unit.level,
                hard=unit.hard,
                n_main=suite.main.samples_per_problem,
                maj_ks=ks,
                same=same_answer,
            )
        )
    if by_problem:
        raise SystemExit(f"samples for problems outside the plan: {sorted(by_problem)[:5]}")
    write_jsonl(cli.output / "per_problem.jsonl", results)

    per_set = {
        name: tier_table(rs, suite.main.ks, hard=False)
        for name, rs in group_by(results, lambda r: r.problem_set).items()
    }
    per_level = {
        level: tier_table(rs, suite.main.ks, hard=False)
        for level, rs in group_by(results, lambda r: r.level.value).items()
    }
    hard = [r for r in results if r.hard]
    hard_tables = None
    if hard:
        hard_tables = {
            "all": tier_table(hard, suite.hard.ks, hard=True),
            "per_level": {
                level: tier_table(rs, suite.hard.ks, hard=True)
                for level, rs in group_by(hard, lambda r: r.level.value).items()
            },
        }
    report = {
        "plan": plan.model_dump(),
        "main": {"per_set": per_set, "per_level": per_level},
        "hard": hard_tables,
    }
    (cli.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"{'set':18} {'n':>5} "
        + " ".join(f"{'p@' + str(k):>6}" for k in suite.main.ks)
        + f" {'maj@' + str(suite.main.ks[-1]):>7} {'trunc':>6}"
    )
    rows = [*per_set.items(), *((f"level {k}", v) for k, v in per_level.items())]
    for name, row in rows:
        print(
            f"{name:18} {int(row['problems']):5} "
            + " ".join(f"{row[f'pass@{k}']:6.3f}" for k in suite.main.ks)
            + f" {row[f'maj@{suite.main.ks[-1]}']:7.3f} {row['truncation_rate']:6.3f}"
        )
    return 0


class EvaluateCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.trainer.main_eval", env_prefix="SOE_EVAL_", cli_kebab_case=True
    )

    suite: Path
    checkpoint: str  # a Hub id from scripts/stage_assets.yaml, or a local checkpoint directory
    output: Path
    stage: Literal["sample", "report"] = "sample"
    shard: int | None = None  # sample this shard only (default: claim free ones)
    num_shards: int = 32
    limit_per_set: int | None = None  # smoke tests: the first N problems (by id) of each set
    # vLLM speed knobs only (recorded per shard); sampling always comes from the suite
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int | None = None

    def cli_cmd(self) -> None:
        suite = load_suite(self.suite)
        data_root = Path(os.environ["SOE_DATA"])
        bad = hash_mismatches(suite, data_root)
        if bad:
            raise SystemExit(f"refusing to evaluate: problem sets changed or missing: {bad}")
        units = eval_units(suite, data_root, self.limit_per_set)
        plan = make_plan(
            self.suite, suite, self.checkpoint, units, self.num_shards, self.limit_per_set
        )
        write_or_check_plan(self.output, plan)
        if self.stage == "sample":
            run_sample(self, suite, units, plan)
        elif run_report(self, suite, units, plan):
            raise SystemExit(1)


if __name__ == "__main__":
    CliApp.run(EvaluateCli)
