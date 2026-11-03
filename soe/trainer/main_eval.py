"""Evaluation entry point: sample the frozen suite for one checkpoint, then build its tables.

    --stage sample  GPU. Checks every problem-file hash, then claims shards of the suite (see
                    soe.utils.shards) and writes SampleRecord JSONL plus run info per shard under
                    --output. Run several jobs at once on any GPU type; `--shard k` finishes one.
    --stage report  CPU. Checks completeness and writes per_problem.jsonl and report.json:
                    pass@k / maj@k per problem set and per level (main tier) and on the hard subset.
                    With --rescore-from DIR --rescore-suite OLD.yaml it scores DIR's samples, drawn
                    under OLD, under --suite instead, which may only add exclusions (decision 026).

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
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.eval_suite import EvalSuite, file_sha256, hash_mismatches, load_suite
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord, SamplingRunInfo
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.metric.eval_tables import ProblemResult, problem_result, tier_table
from soe.utils.reward_score.math_verify import (
    answers_equivalent,
    score_response,
    timed_out_comparisons,
)
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


class RescoreSource(BaseModel):
    """Where a re-scored evaluation's samples came from (decision 026): an evaluation sampled under
    an earlier suite that differs only in its name and in problems it excludes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    directory: str
    plan_sha256: str
    suite_name: str
    suite_file_sha256: str
    shard_files_sha256: dict[str, str]
    dropped_problems: list[str]  # sampled there, excluded here
    dropped_samples: int


class EvalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    suite_name: str
    suite_file_sha256: str
    checkpoint: str
    revision: str | None
    num_shards: int
    limit_per_set: int | None  # smoke tests only; full evaluations have None
    only_problems_sha256: str | None = None  # partial re-evaluation of listed problems (e.g. 007)
    units: int
    samples: int
    rescored_from: RescoreSource | None = None  # set only by `--rescore-from` (decision 026)


def eval_units(
    suite: EvalSuite,
    data_root: Path,
    limit_per_set: int | None,
    *,
    only: frozenset[str] | None = None,
) -> list[EvalUnit]:
    """Every (problem, number of samples) of the suite, sorted by problem id; with `only`, just
    those problems (each must exist)."""
    hard_ids: set[str] = set()
    if suite.hard_subset is not None:
        hard_ids = {p.problem_id for p in read_jsonl(data_root / suite.hard_subset.path, Problem)}
    if not set(suite.removed_from_hard_subset) <= hard_ids:
        raise ValueError("removed from the hard subset but not in it")
    hard_ids -= set(suite.removed_from_hard_subset)
    units = []
    for problem_set in suite.problem_sets:
        problems = sorted(
            read_jsonl(data_root / problem_set.path, Problem), key=lambda p: p.problem_id
        )
        kept = [p for p in problems if p.problem_id not in suite.excluded_problems]
        for problem in kept[:limit_per_set]:
            if only is not None and problem.problem_id not in only:
                continue
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
    if only is not None and set(ids) != only:
        raise ValueError(f"not in the suite: {sorted(only - set(ids))[:5]}")
    present = {
        p.problem_id for s in suite.problem_sets for p in read_jsonl(data_root / s.path, Problem)
    }
    unknown = set(suite.excluded_problems) - present
    if unknown:
        raise ValueError(f"excluded problems not in the suite: {sorted(unknown)}")
    hard_ids -= set(suite.excluded_problems)
    if limit_per_set is None and only is None and not hard_ids <= set(ids):
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
    """The first job publishes the plan; every later job must match it.

    The plan is written to a private file and then hard-linked into place: the link is atomic and
    fails if the plan exists, so no job ever reads a half-written plan.json.
    """
    path = output / "plan.json"
    output.mkdir(parents=True, exist_ok=True)
    private = output / f".plan-{os.getpid()}-{platform.node()}.json"
    private.write_text(plan.model_dump_json(indent=1) + "\n")
    try:
        os.link(private, path)
    except FileExistsError:
        existing = EvalPlan.model_validate_json(path.read_text())
        if existing != plan:
            raise SystemExit(
                f"{path} describes a different evaluation:\n{existing}\nvs\n{plan}"
            ) from None
    finally:
        private.unlink()


def make_plan(
    suite_path: Path,
    suite: EvalSuite,
    checkpoint: str,
    units: list[EvalUnit],
    num_shards: int,
    limit_per_set: int | None,
    *,
    only_problems_sha256: str | None = None,
    rescored_from: RescoreSource | None = None,
) -> EvalPlan:
    return EvalPlan(
        suite_name=suite.name,
        suite_file_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        checkpoint=checkpoint,
        revision=pinned_revision(checkpoint),
        num_shards=num_shards,
        limit_per_set=limit_per_set,
        only_problems_sha256=only_problems_sha256,
        units=len(units),
        samples=sum(u.n for u in units),
        rescored_from=rescored_from,
    )


def run_sample(cli: "EvaluateCli", suite: EvalSuite, units: list[EvalUnit], plan: EvalPlan) -> None:
    from soe.workers.rollout.vllm_sampler import EngineSettings, VllmSampler  # GPU image only

    engine = EngineSettings(
        gpu_memory_utilization=cli.gpu_memory_utilization, max_num_seqs=cli.max_num_seqs
    )
    sampler = VllmSampler(
        cli.checkpoint,
        revision=plan.revision,
        max_tokens=suite.sampling.max_tokens,
        engine=engine,
        kv_cache_dtype=suite.sampling.kv_cache_dtype,
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
        git_commit=git_commit(),
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


def rescored(record: SampleRecord, gold: str) -> SampleRecord:
    """The record as the current verifier judges it. Sampling stores a verdict, but the report
    scores every completion again, so a verifier or extraction fix reaches old samples without
    re-sampling."""
    verdict = score_response(record.completion, gold)
    correct = record.finish_reason == "stop" and verdict.correct
    if (correct, verdict.extracted) == (record.correct, record.extracted_answer):
        return record
    return record.model_copy(update={"correct": correct, "extracted_answer": verdict.extracted})


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def report_tables(results: list[ProblemResult], suite: EvalSuite) -> dict[str, Any]:
    """Main-tier tables per problem set and per level; hard-tier tables overall and per level."""
    main = {
        "per_set": {
            name: tier_table(rs, suite.main.ks, hard=False)
            for name, rs in group_by(results, lambda r: r.problem_set).items()
        },
        "per_level": {
            level: tier_table(rs, suite.main.ks, hard=False)
            for level, rs in group_by(results, lambda r: r.level.value).items()
        },
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
    return {"main": main, "hard": hard_tables}


def print_main_table(main: dict[str, Any], ks: list[int]) -> None:
    print(
        f"{'set':18} {'n':>5} "
        + " ".join(f"{'p@' + str(k):>6}" for k in ks)
        + f" {'maj@' + str(ks[-1]):>7} {'trunc':>6}"
    )
    per_level = main["per_level"].items()
    for name, row in [*main["per_set"].items(), *((f"level {k}", v) for k, v in per_level)]:
        print(
            f"{name:18} {int(row['problems']):5} "
            + " ".join(f"{row[f'pass@{k}']:6.3f}" for k in ks)
            + f" {row[f'maj@{ks[-1]}']:7.3f} {row['truncation_rate']:6.3f}"
        )


def read_samples(directory: Path, num_shards: int) -> dict[str, list[SampleRecord]]:
    by_problem: dict[str, list[SampleRecord]] = defaultdict(list)
    for k in range(num_shards):
        for record in read_jsonl(shard_path(directory, k, num_shards), SampleRecord):
            by_problem[record.problem_id].append(record)
    return by_problem


def run_report(cli: "EvaluateCli", suite: EvalSuite, units: list[EvalUnit], plan: EvalPlan) -> int:
    left = unfinished_shards(cli.output, plan.num_shards)
    if left:
        print(f"not finished: shards {left}")
        return 1
    return write_report(cli.output, suite, units, plan, read_samples(cli.output, plan.num_shards))


def common_suite(*evaluations: Path) -> str:
    """The suite every evaluation directory was scored under. Reports of different suites (v3 and
    v4, decision 026) are never compared problem by problem, so a mix is refused."""
    names = {
        str(d): EvalPlan.model_validate_json((d / "plan.json").read_text()).suite_name
        for d in evaluations
    }
    if len(set(names.values())) != 1:
        raise SystemExit(f"evaluations scored under different suites: {names}")
    return next(iter(names.values()))


def newly_excluded(old: EvalSuite, new: EvalSuite) -> frozenset[str]:
    """The problems `new` excludes beyond `old`, if samples drawn under `old` can be scored under
    `new` without re-sampling: everything that decides what is sampled (problem files, hard
    subset, sampling, prompt, tiers) and how it is scored is identical, and `new` only adds
    exclusions (decision 026)."""
    if not set(old.excluded_problems) <= set(new.excluded_problems):
        raise ValueError(f"{new.name} drops exclusions of {old.name}; re-sample instead")
    aligned = old.model_copy(update={"name": new.name, "excluded_problems": new.excluded_problems})
    if aligned != new:
        fields = [f for f in EvalSuite.model_fields if getattr(aligned, f) != getattr(new, f)]
        raise ValueError(f"{old.name} and {new.name} differ in {fields}; re-sample instead")
    return frozenset(set(new.excluded_problems) - set(old.excluded_problems))


def run_rescore(cli: "EvaluateCli", suite: EvalSuite, units: list[EvalUnit]) -> int:
    """Score an evaluation sampled under an earlier suite as if sampled under `suite` (decision
    026): the old shards are read, samples of newly excluded problems are dropped and counted, and
    the report goes to a new directory whose plan records where its samples came from."""
    assert cli.rescore_from is not None and cli.rescore_suite is not None
    source = cli.rescore_from
    source_plan_text = (source / "plan.json").read_text()
    source_plan = EvalPlan.model_validate_json(source_plan_text)
    old = load_suite(cli.rescore_suite)
    old_sha = hashlib.sha256(cli.rescore_suite.read_bytes()).hexdigest()
    if (old.name, old_sha) != (source_plan.suite_name, source_plan.suite_file_sha256):
        raise SystemExit(f"{cli.rescore_suite} is not the suite {source} was sampled with")
    dropped = newly_excluded(old, suite)
    asked = (cli.checkpoint, cli.limit_per_set)
    if (source_plan.checkpoint, source_plan.limit_per_set) != asked:
        raise SystemExit(f"{source} evaluated {source_plan.checkpoint}, not {cli.checkpoint}")
    if source_plan.only_problems_sha256 is not None:
        raise SystemExit("re-scoring a partial evaluation (--only-problems) is not supported")
    left = unfinished_shards(source, source_plan.num_shards)
    if left:
        raise SystemExit(f"{source}: shards {left} not finished")
    if any(cli.output.glob("samples-*.jsonl")):
        raise SystemExit(f"{cli.output} holds samples; a re-score writes to a new directory")
    by_problem = read_samples(source, source_plan.num_shards)
    dropped_ids = sorted(dropped & set(by_problem))
    dropped_samples = sum(len(by_problem.pop(pid)) for pid in dropped_ids)
    shard_files = [
        shard_path(source, k, source_plan.num_shards) for k in range(source_plan.num_shards)
    ]
    provenance = RescoreSource(
        directory=str(source),
        plan_sha256=hashlib.sha256(source_plan_text.encode()).hexdigest(),
        suite_name=source_plan.suite_name,
        suite_file_sha256=source_plan.suite_file_sha256,
        shard_files_sha256={f.name: file_sha256(f) for f in shard_files},
        dropped_problems=dropped_ids,
        dropped_samples=dropped_samples,
    )
    plan = make_plan(
        cli.suite,
        suite,
        cli.checkpoint,
        units,
        source_plan.num_shards,
        cli.limit_per_set,
        rescored_from=provenance,
    )
    if (plan.units, plan.samples) != (
        source_plan.units - len(dropped_ids),
        source_plan.samples - dropped_samples,
    ):
        raise SystemExit("the re-scored plan does not account for every source sample")
    write_or_check_plan(cli.output, plan)
    print(f"re-scoring {source}: dropped {dropped_samples} samples of {dropped_ids}")
    return write_report(cli.output, suite, units, plan, by_problem)


def write_report(
    output: Path,
    suite: EvalSuite,
    units: list[EvalUnit],
    plan: EvalPlan,
    by_problem: dict[str, list[SampleRecord]],
) -> int:
    results, changed = [], 0
    for unit in units:
        stored = by_problem.pop(unit.problem.problem_id, [])
        records = [rescored(r, unit.problem.answer) for r in stored]
        changed += sum(a.correct != b.correct for a, b in zip(stored, records, strict=True))
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
    write_jsonl(output / "per_problem.jsonl", results)

    commit = git_commit()
    timeouts = [[p[:200], g[:200]] for p, g in timed_out_comparisons()]
    scoring = {
        "git_commit": commit,
        "verdicts_changed_from_sampling": changed,
        "verifier_timeouts": timeouts,  # (predicted, gold) pairs math-verify's time limit decided
    }
    report = {"plan": plan.model_dump(), "scoring": scoring, **report_tables(results, suite)}
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"scored at {commit[:7]}; {changed} verdicts differ from sampling time; "
        f"{len(timeouts)} comparisons hit the verifier's time limit"
    )
    print_main_table(report["main"], suite.main.ks)
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
    only_problems: Path | None = None  # file with one problem id per line: evaluate just those
    # Report stage only (decision 026): score the samples of an evaluation drawn under an earlier
    # suite (`rescore_suite`, its file) that this suite only adds exclusions to; writes to --output.
    rescore_from: Path | None = None
    rescore_suite: Path | None = None
    # vLLM speed knobs only (recorded per shard); sampling always comes from the suite
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int | None = None

    def cli_cmd(self) -> None:
        suite = load_suite(self.suite)
        data_root = Path(os.environ["SOE_DATA"])
        bad = hash_mismatches(suite, data_root)
        if bad:
            raise SystemExit(f"refusing to evaluate: problem sets changed or missing: {bad}")
        only, only_sha = None, None
        if self.only_problems is not None:
            text = self.only_problems.read_bytes()
            only = frozenset(line.strip() for line in text.decode().splitlines() if line.strip())
            only_sha = hashlib.sha256(text).hexdigest()
        units = eval_units(suite, data_root, self.limit_per_set, only=only)
        if (self.rescore_from is None) != (self.rescore_suite is None):
            raise SystemExit("--rescore-from and --rescore-suite go together")
        if self.rescore_from is not None:
            if self.stage != "report" or only is not None:
                raise SystemExit("--rescore-from is a full report-stage run")
            if run_rescore(self, suite, units):
                raise SystemExit(1)
            return
        plan = make_plan(
            self.suite,
            suite,
            self.checkpoint,
            units,
            self.num_shards,
            self.limit_per_set,
            only_problems_sha256=only_sha,
        )
        write_or_check_plan(self.output, plan)
        if self.stage == "sample":
            run_sample(self, suite, units, plan)
        elif run_report(self, suite, units, plan):
            raise SystemExit(1)


if __name__ == "__main__":
    CliApp.run(EvaluateCli)
