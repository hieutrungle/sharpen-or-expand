"""Sampling rate per GPU type from shard claims: claim-to-done seconds and completion tokens."""

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_gpu_rates", Path("scripts/eval_gpu_rates.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rates_module = load()


def shard(eval_dir: Path, k: int, gpu: str, seconds: float, tokens: list[int]) -> None:
    info = {"gpus": [f"NVIDIA {gpu}, 49140 MiB"]}
    (eval_dir / f"runinfo-{k}-of-3.jsonl").write_text(json.dumps(info) + "\n")
    lines = [json.dumps({"completion_tokens": t}) for t in tokens]
    (eval_dir / f"samples-{k}-of-3.jsonl").write_text("\n".join(lines) + "\n")
    claim = eval_dir / "claims" / f"shard-{k}"
    claim.mkdir(parents=True)
    (claim / "owner").write_text("job\n")
    (claim / "done").touch()
    os.utime(claim / "owner", (1000.0, 1000.0))
    os.utime(claim / "done", (1000.0 + seconds, 1000.0 + seconds))


def test_rates(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text(json.dumps({"num_shards": 3}))
    shard(tmp_path, 0, "A40", 100.0, [1000, 1000])
    shard(tmp_path, 1, "A40", 300.0, [4000])
    shard(tmp_path, 2, "A10", 200.0, [500, 500])
    result = rates_module.rates(tmp_path)
    assert result["completion_tokens"] == 7000
    assert result["shard_hours"] == pytest.approx(600 / 3600)
    a40, a10 = result["by_gpu"]["A40"], result["by_gpu"]["A10"]
    assert a40["shards"] == 2
    assert a40["tokens_per_second"] == pytest.approx(6000 / 400)  # pooled, not a mean of rates
    assert a40["median_shard_tokens_per_second"] == pytest.approx((20 + 40 / 3) / 2)
    assert a10["tokens_per_second"] == pytest.approx(5.0)
