"""M2.5: the SFT train and validation files (roadmap D1 (c), D3, D4; decision 027).

    SOE_DATA=... python examples/data_preprocess/m2_sft_data.py \
        --config examples/data_preprocess/m2_sft_data.yaml problems
    (then the pool cleaner on its output: m2_sft_pool_clean.py --config .../m2_sft_kept_clean.yaml)
    sbatch slurm/cpu.sbatch python examples/data_preprocess/m2_sft_data.py \
        --config examples/data_preprocess/m2_sft_data.yaml build

problems  Reads the finished sampling runs (the pilot and the full run: one permutation, contiguous
          draws), keeps each problem's trajectories with `kept_completions` (the M2.4 rule) and
          writes the problems that keep at least one (Problem lines, sorted by id). Decision 027's
          checks then run on exactly these problems, through the pool cleaner, until they list
          nothing: a smaller set has fewer boilerplate n-grams, so the checks can list pairs that
          the full pool hid.
build     Container (the trainer's tokenizer, so lengths are the ones SFT sees). Examples of the
          clean problems only; `val_problems` problems (seeded) go to val. An example longer than
          `max_length` tokens is dropped, never cut. Writes train/val (SftExample lines) and the
          report: counts, tokens, loss tokens, length histogram, drops by reason, per-difficulty
          counts and sha256s. No text.
"""

import argparse
import json
import os
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soe.trainer.config.eval_suite import PromptSpec, file_sha256, load_suite
from soe.trainer.sft.data import SftExample, kept_completions
from soe.trainer.sft.sampling import SampleConfig, drawn, pinned_revision, read_samples
from soe.utils.dataset.problems import Problem
from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl, write_jsonl
from soe.utils.shards import shard_path

SHA256 = r"^[0-9a-f]{64}$"
LENGTH_BINS = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432]
SHARED = ("suite", "model", "problems", "problems_sha256", "meta", "draw_seed", "sampling_seed")


class SftDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    samples: list[Path] = Field(min_length=1)  # sampling configs (repo paths), in draw order
    kept_problems: str  # written by `problems`, relative to $SOE_DATA
    clean_problems: str  # the cleaner's output, relative to $SOE_DATA
    clean_problems_sha256: str | None = Field(default=None, pattern=SHA256)  # set after the cleaner
    clean_report: Path  # the cleaner's report (repo path)
    output_dir: str  # relative to $SOE_DATA
    val_problems: int = Field(gt=0)
    seed: int
    max_length: int = Field(gt=0)  # decision 023's limit, as main_sft applies it
    report: Path  # repo path


def load_runs(paths: list[Path]) -> list[SampleConfig]:
    """The sampling runs, checked to be one permutation of one pool, drawn contiguously from 0."""
    runs = [SampleConfig.model_validate(yaml.safe_load(p.read_text())) for p in paths]
    for run in runs[1:]:
        differ = [f for f in SHARED if getattr(run, f) != getattr(runs[0], f)]
        if differ or run.max_kept_per_problem != runs[0].max_kept_per_problem:
            raise SystemExit(f"{run.version} differs from {runs[0].version} in {differ}")
    stops = [0] + [r.draw_stop for r in runs[:-1]]
    if [r.draw_start for r in runs] != stops:
        raise SystemExit(
            f"draws are not contiguous from 0: {[(r.draw_start, r.draw_stop) for r in runs]}"
        )
    return runs


def kept_by_problem(
    samples: dict[str, list[SampleRecord]], max_kept: int
) -> dict[str, list[SampleRecord]]:
    return {pid: kept_completions(rs, max_kept) for pid, rs in sorted(samples.items())}


def val_ids(problem_ids: list[str], count: int, seed: int) -> set[str]:
    return set(random.Random(seed).sample(sorted(problem_ids), count))


def examples(
    kept: dict[str, list[SampleRecord]], problems: dict[str, Problem], prompt: PromptSpec
) -> list[SftExample]:
    """One example per kept trajectory of each given problem, in problem-id then sample order."""
    return [
        SftExample(
            problem_id=pid, user=prompt.render(problems[pid].question), completion=r.completion
        )
        for pid in sorted(problems)
        for r in kept[pid]
    ]


def histogram(lengths: list[int]) -> dict[str, int]:
    edges = zip([0, *LENGTH_BINS], LENGTH_BINS, strict=False)
    return {str(hi): sum(lo < n <= hi for n in lengths) for lo, hi in edges}


def split_and_measure(
    rows: list[SftExample],
    val: set[str],
    measure: Callable[[SftExample], tuple[int, int]],
    max_length: int,
) -> tuple[dict[str, list[SftExample]], dict[str, list[tuple[int, int]]], int]:
    """Train/val examples that fit, their (tokens, loss tokens), and how many were too long."""
    splits: dict[str, list[SftExample]] = {"train": [], "val": []}
    sizes: dict[str, list[tuple[int, int]]] = {"train": [], "val": []}
    too_long = 0
    for row in rows:
        tokens, loss_tokens = measure(row)
        if tokens > max_length:
            too_long += 1
            continue
        name = "val" if row.problem_id in val else "train"
        splits[name].append(row)
        sizes[name].append((tokens, loss_tokens))
    return splits, sizes, too_long


def run_problems(config: SftDataConfig, data_root: Path) -> None:
    runs = load_runs(config.samples)
    pool = {p.problem_id: p for run in runs for p in drawn(run, data_root)}
    samples = {pid: rs for run in runs for pid, rs in read_samples(run, data_root).items()}
    kept = kept_by_problem(samples, runs[0].max_kept_per_problem)
    with_kept = [pool[pid] for pid, ks in kept.items() if ks]
    out = data_root / config.kept_problems
    write_jsonl(out, with_kept)
    print(json.dumps({"drawn": len(pool), "with_kept": len(with_kept), "sha256": file_sha256(out)}))


def run_build(config: SftDataConfig, data_root: Path) -> None:
    from huggingface_hub import snapshot_download  # offline, from the staged snapshot

    from soe.trainer.sft.trainer import Tokenizer  # container only (transformers)

    runs = load_runs(config.samples)
    clean_path = data_root / config.clean_problems
    if (
        config.clean_problems_sha256 is None
        or file_sha256(clean_path) != config.clean_problems_sha256
    ):
        raise SystemExit(f"{clean_path} does not match clean_problems_sha256")
    clean = {p.problem_id: p for p in read_jsonl(clean_path, Problem)}
    samples = {pid: rs for run in runs for pid, rs in read_samples(run, data_root).items()}
    kept = kept_by_problem(samples, runs[0].max_kept_per_problem)
    with_kept = {pid for pid, ks in kept.items() if ks}
    if not set(clean) <= with_kept:
        raise SystemExit(
            "the clean problems are not a subset of the problems with a kept trajectory"
        )
    suite = load_suite(runs[0].suite)
    base = snapshot_download(
        runs[0].model, revision=pinned_revision(runs[0].model), local_files_only=True
    )
    tokenizer = Tokenizer(Path(base), suite.prompt)

    def measure(example: SftExample) -> tuple[int, int]:
        tokenized = tokenizer.tokenize(example)
        return len(tokenized.input_ids), sum(tokenized.completion_mask)

    val = val_ids(list(clean), config.val_problems, config.seed)
    rows = examples(kept, clean, suite.prompt)
    splits, sizes, too_long = split_and_measure(rows, val, measure, config.max_length)
    out = data_root / config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, split_rows in splits.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        files[name] = {
            "path": f"{config.output_dir}/{name}.jsonl",
            "examples": len(split_rows),
            "problems": len({r.problem_id for r in split_rows}),
            "tokens": sum(t for t, _ in sizes[name]),
            "loss_tokens": sum(lt for _, lt in sizes[name]),
            "sha256": file_sha256(path),
        }
    meta_lines = (data_root / runs[0].meta).read_text().splitlines()
    difficulty = {
        m["problem_id"]: f"{float(m['difficulty']):g}" for m in map(json.loads, meta_lines)
    }
    used = Counter(r.problem_id for rs in splits.values() for r in rs)
    by_difficulty: dict[str, dict[str, int]] = {}
    for pid, n in used.items():
        entry = by_difficulty.setdefault(difficulty[pid], {"problems": 0, "examples": 0})
        entry["problems"] += 1
        entry["examples"] += n
    report = {
        "version": config.version,
        "samples": {
            run.version: {
                "draw": [run.draw_start, run.draw_stop],
                "shard_files_sha256": {
                    f.name: file_sha256(f)
                    for f in (
                        shard_path(data_root / run.work_dir, k, run.num_shards)
                        for k in range(run.num_shards)
                    )
                },
            }
            for run in runs
        },
        "problems_drawn": len(samples),
        "samples_total": sum(len(rs) for rs in samples.values()),
        "kept_per_problem_counts": dict(sorted(Counter(len(ks) for ks in kept.values()).items())),
        "drops": {
            "problems_without_a_kept_trajectory": len(samples) - len(with_kept),
            "problems_listed_by_the_contamination_checks": len(with_kept) - len(clean),
            "trajectories_of_listed_problems": sum(len(kept[p]) for p in with_kept - set(clean)),
            "examples_too_long": too_long,
        },
        "clean_problems": {"path": config.clean_problems, "sha256": config.clean_problems_sha256},
        "clean_report_sha256": file_sha256(config.clean_report),
        "max_length": config.max_length,
        "files": files,
        "length_histogram_upper_edges": histogram([t for s in sizes.values() for t, _ in s]),
        "by_difficulty": dict(sorted(by_difficulty.items(), key=lambda kv: float(kv[0]))),
        "val_problem_ids": sorted(val),
    }
    config.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("val_problem_ids", "samples")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("stage", choices=("problems", "build"))
    args = parser.parse_args()
    config = SftDataConfig.model_validate(yaml.safe_load(args.config.read_text()))
    data_root = Path(os.environ["SOE_DATA"])
    if args.stage == "problems":
        run_problems(config, data_root)
    else:
        run_build(config, data_root)


if __name__ == "__main__":
    main()
