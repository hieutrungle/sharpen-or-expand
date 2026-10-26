# /// script
# requires-python = ">=3.11"
# # socksio is needed only behind a SOCKS proxy (e.g. the agent sandbox); harmless elsewhere.
# dependencies = ["huggingface_hub>=0.34", "pandas>=2.2", "pyarrow>=17", "socksio"]
# ///
"""Reproduce the dataset-candidate table behind decisions 002 and 003.

For every (evaluation set, RL pool) pair it reports how many evaluation problems share any word
13-gram with the pool, and how many are near-duplicates (>= 50% of their 13-grams occur in the
pool). Files are downloaded at the revisions pinned below. Run anywhere with internet:

    uv run scripts/candidate_overlap.py > docs/reports/candidate_overlap.txt

The full contamination report (M0.9) supersedes this once the splits are frozen.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

Loader = Callable[[str], pd.Series]

# (repo, revision, file, column or extractor)
EVAL_SETS: dict[str, tuple[str, str, str, str]] = {
    "A:MATH-500": (
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "test.jsonl",
        "problem",
    ),
    "A:AIME24": (
        "HuggingFaceH4/aime_2024",
        "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "A:AIME25": (
        "MathArena/aime_2025",
        "c94da77eb22bbd6439e62a323bec18493a421302",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "O:OlympiadBench-EN": (
        "Hothan/OlympiadBench",
        "91184b52131e7fc9455fef848035173aea8cc01a",
        "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet",
        "question",
    ),
    "O:Omni-MATH": (
        "KbsdJames/Omni-MATH",
        "40ba231d8f16e29ecd40e6407e2c8640145a8f62",
        "test.jsonl",
        "problem",
    ),
    "O:OlymMATH-EN-HARD": (
        "RUC-AIBOX/OlymMATH",
        "2c6532ea2cf929ac1c421532af5951553eaee727",
        "data/OlymMATH-EN-HARD.jsonl",
        "problem",
    ),
    "O:OlymMATH-EN-EASY": (
        "RUC-AIBOX/OlymMATH",
        "2c6532ea2cf929ac1c421532af5951553eaee727",
        "data/OlymMATH-EN-EASY.jsonl",
        "problem",
    ),
    "C1:AIME26": (
        "MathArena/aime_2026",
        "d61f486805306e6e6b5913504c07763a99e1b112",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "C1:HMMT-Feb26": (
        "MathArena/hmmt_feb_2026",
        "589a53fe152febc0f4f99f3e561007bf8da8e664",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "C2:HMMT-Nov25": (
        "MathArena/hmmt_nov_2025",
        "118dbfb45c4c9467c672268ed55166642897aa46",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "C2:HMMT-Feb25": (
        "MathArena/hmmt_feb_2025",
        "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "C2:BRUMO25": (
        "MathArena/brumo_2025",
        "6de9bc0987f5710c6376ee980e1483dbeb55cfda",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "C2:SMT25": (
        "MathArena/smt_2025",
        "d8e99d83f1583842f46d4953cd960eea5d887e45",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
    "C2:CMIMC25": (
        "MathArena/cmimc_2025",
        "7b3ca7c4b417d31b1f97f781f39fc4f97f1b9185",
        "data/train-00000-of-00001.parquet",
        "problem",
    ),
}
POOLS: dict[str, tuple[str, str, str, str]] = {
    "DAPO-17k": (
        "open-r1/DAPO-Math-17k-Processed",
        "31dd309567e3da778038cc87d868b6097a3ccf68",
        "en/train-00000-of-00001.parquet",
        "prompt",
    ),
    "DeepScaleR": (
        "agentica-org/DeepScaleR-Preview-Dataset",
        "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
        "deepscaler.json",
        "problem",
    ),
    "Skywork-OR1": (
        "Skywork/Skywork-OR1-RL-Data",
        "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
        "data/math-00000-of-00001.parquet",
        "prompt",
    ),
}


def read(repo: str, revision: str, filename: str, column: str) -> pd.Series:
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    if path.endswith(".parquet"):
        frame = pd.read_parquet(path)
    elif path.endswith(".json"):
        frame = pd.DataFrame(json.loads(Path(path).read_text()))
    else:
        frame = pd.read_json(path, lines=True)
    return frame[column].map(as_text)


def as_text(value: object) -> str:
    if isinstance(value, str):
        # DAPO wraps each problem in a fixed instruction; keep only the problem statement.
        match = re.search(r"step by step\..*?\n\n(.*)\n\nRemember to put your answer", value, re.S)
        return match.group(1) if match else value
    return " ".join(message["content"] for message in value)  # chat-format prompts


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)} or {tuple(tokens)}


def main() -> None:
    pools = {name: read(*spec) for name, spec in POOLS.items()}
    pool_grams = {name: set().union(*map(ngrams, texts)) for name, texts in pools.items()}
    for name, texts in pools.items():
        print(f"pool {name}: {len(texts)} problems")
    print(f"\n{'eval set':22s} {'n':>5s}  " + "  ".join(f"{p:>16s}" for p in pools))
    print("(cells: any 13-gram hit / near-duplicate >= 50%)")
    for name, spec in EVAL_SETS.items():
        texts = read(*spec)
        cells = []
        for grams in pool_grams.values():
            fractions = [len(ngrams(t) & grams) / len(ngrams(t)) for t in texts]
            cells.append(f"{sum(f > 0 for f in fractions)}/{sum(f >= 0.5 for f in fractions)}")
        print(f"{name:22s} {len(texts):5d}  " + "  ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
