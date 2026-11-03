"""The G0 schedule: Lane A hours, one evaluation per save in A40-hours, pools, the critical path."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("g0_schedule", Path("scripts/g0_schedule.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g0 = load()


def write(tmp_path: Path) -> object:
    throughput = {
        "runs": [
            {
                "config_name": "run",
                "rate_tokens_per_hour": 10e6,
                "rate_spread": 0.1,
                "length_sensitivity": [
                    {"rate_tokens_per_hour": 8e6},
                    {"rate_tokens_per_hour": 12e6},
                ],
            }
        ]
    }
    steps = {"2": {"training_tokens": 3.6e6, "update_actor_s": 360.0}}
    memory = {"cells": [{"cell": "C", "measured": [{"steps": steps}]}]}
    # the A40 samples 1M tokens per hour, the A10 half that; an evaluation is 24M tokens
    by_gpu = {"A40": {"tokens_per_second": 1e6 / 3600}, "A10": {"tokens_per_second": 0.5e6 / 3600}}
    rates = {"completion_tokens": 24e6, "by_gpu": by_gpu}
    for name, data in (("t", throughput), ("m", memory), ("r", rates)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    inputs = {
        "throughput_report": str(tmp_path / "t.json"),
        "throughput_run": "run",
        "memory_report": str(tmp_path / "m.json"),
        "memory_cell": "C",
        "eval_rates_report": str(tmp_path / "r.json"),
        "queue_factor": 1.25,
        "save_every_hours": 3.0,
        "evals_per_run_cap": 2,
        "lane_b_pools": {"one": {"A40": 1}, "mixed": {"A40": 1, "A10": 2}},
        "mean_response_tokens": [1000, 1000],
        "milestones": [
            # 2 runs of 120M tokens: 10 h at the best rate (12M/h), 15 h at the worst (8M/h)
            {"name": "RL", "rl_runs": [2, 2], "rl_tokens": [120e6, 120e6]},
            # 36M training tokens at 36M/h, plus 1,000 sampled responses and one evaluation
            {
                "name": "SFT",
                "train_tokens": [36e6, 36e6],
                "generated_samples": [1000, 1000],
                "extra_evals": 1,
            },
        ],
    }
    return g0.ScheduleInputs.model_validate(yaml.safe_load(yaml.safe_dump(inputs)))


def test_schedule(tmp_path: Path) -> None:
    result = g0.schedule(write(tmp_path))
    assert result["rl_rate_tokens_per_hour"] == {"rate": 10e6, "low": 8e6, "high": 12e6}
    assert result["train_tokens_per_hour"] == pytest.approx(36e6)
    rl, sft = result["milestones"]
    assert rl["low"]["lane_a_hours"] == pytest.approx(20.0)
    assert rl["high"]["lane_a_hours"] == pytest.approx(30.0)
    assert rl["high"]["lane_a_hours_with_queue_factor"] == pytest.approx(37.5)
    assert result["eval_a40_hours_per_checkpoint"] == pytest.approx(24.0)
    assert result["lane_b_pool_a40_equivalents"] == pytest.approx({"one": 1.0, "mixed": 2.0})
    # a save every 3 h: 4 (10 h) and 5 (15 h) evaluations per run, 24 A40-h each
    assert rl["low"]["lane_b_every_save"]["evaluations"] == 8
    assert rl["high"]["lane_b_every_save"]["evaluations"] == 10
    assert rl["high"]["lane_b_every_save"]["days"] == pytest.approx({"one": 10.0, "mixed": 5.0})
    assert rl["high"]["lane_b_capped"]["evaluations"] == 4
    assert sft["low"]["lane_a_hours"] == pytest.approx(1.0)
    # sampling: 1,000 x 1,000 tokens at 1M tokens per A40-hour = 1 A40-h, plus one evaluation
    assert sft["low"]["lane_b_every_save"]["a40_hours"] == pytest.approx(25.0)
    path = result["totals"]["high"]["lane_b_every_save"]["critical_path_days"]
    # the longer lane of each milestone: Lane B for both
    assert path == pytest.approx({"one": 10.0 + 25 / 24, "mixed": 5.0 + 25 / 48})
