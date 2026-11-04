"""scripts/sft_probe_report.py on a synthetic 4-step SFT run with saves at 2 and 4 (M2.8)."""

import importlib.util
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.rl.segments import SegmentEvent, append_event

CONFIG = Path("examples/sft/m2_probe_qwen35-4b.yaml")
T0 = datetime(2026, 11, 4, tzinfo=UTC)


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sft_probe_report", Path("scripts/sft_probe_report.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load()


def at(seconds: float) -> float:
    return (T0 + timedelta(seconds=seconds)).timestamp()


def make_run(run: Path) -> None:
    for name in ("metrics", "examples", "gpu_memory", "host_memory"):
        (run / name).mkdir(parents=True)
    shutil.copy(CONFIG, run / "config.yaml")
    event = {"segment": "7", "node": "fc098", "hardware": "4 NVIDIA A40", "git_commit": "abc"}
    append_event(run, SegmentEvent(event="start", at=T0, **event))
    append_event(
        run, SegmentEvent(event="end", at=T0 + timedelta(seconds=190), exit="done", **event)
    )
    lines = [  # step 1 ends at 100 s; 10 s steps; saves of 5 s (step 2) and 3 s (step 4)
        (1, {"loss": 0.3, "wall_time": at(100)}),
        (2, {"loss": 0.3, "wall_time": at(110)}),
        (2, {"save_seconds": 5.0}),
        (3, {"loss": 0.3, "wall_time": at(125)}),
        (4, {"loss": 0.3, "wall_time": at(135)}),
        (4, {"eval_loss": 0.3, "eval_runtime": 20.0, "wall_time": at(155)}),
        (4, {"save_seconds": 3.0}),
        (4, {"train_runtime": 160.0, "wall_time": at(160)}),
    ]
    with open(run / "metrics" / "segment-7.jsonl", "w") as f:
        for step, data in lines:
            peaks = {"gpu_peak_allocated_gib": 30.0, "gpu_peak_reserved_gib": 33.0}
            f.write(json.dumps({"step": step, "data": data | (peaks if "loss" in data else {})}))
            f.write("\n")
    for rank in (0, 1):  # two micro-batches per step and rank: 400 tokens, 240 loss tokens a step
        rows = [
            {"step": s, "examples": [0], "tokens": 100, "loss_tokens": 60}
            for s in (1, 1, 2, 2, 3, 3, 4, 4)
        ]
        (run / "examples" / f"segment-7-rank{rank}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    (run / "gpu_memory" / "segment-7.csv").write_text(
        "2026-11-04T00:00:00.000Z,0, 1024, 49152, 0\n2026-11-04T00:00:02.000Z,0, 40960, 49152, 90\n"
        "2026-11-04T00:00:02.000Z,1, 46080, 49152, 90\n"
    )
    gib = 2**20  # kB
    (run / "host_memory" / "segment-7.csv").write_text(
        f"2026-11-04T00:00:00.000Z,{250 * gib},{240 * gib}\n"
        f"2026-11-04T00:00:02.000Z,{250 * gib},{170 * gib}\n"
    )
    train = {
        "examples": 25551,
        "tokens": 120224680,
        "loss_tokens": 117842688,
        "dropped_too_long": 0,
    }
    (run / "sft_data.json").write_text(json.dumps({"train": train}))


def test_probe_report_times_steps_saves_and_memory(tmp_path: Path) -> None:
    make_run(tmp_path / "run")
    out = probe.report(tmp_path / "run")
    assert out["timed_steps"] == [2, 4]
    assert [out["steps"][s]["seconds"] for s in (2, 3, 4)] == pytest.approx([10, 10, 10])
    assert out["tokens_per_hour"] == pytest.approx(1200 / 30 * 3600)
    assert out["loss_tokens_per_hour"] == pytest.approx(720 / 30 * 3600)
    seconds = out["seconds"]
    assert seconds["saves"] == [5.0, 3.0] and seconds["save"] == pytest.approx(4.0)
    assert seconds["start_to_step1"] == pytest.approx(100)
    assert seconds["validation"] == 20.0 and seconds["export"] == pytest.approx(30)
    memory = out["memory"]
    assert memory["torch_rank0_peak_allocated_gib"] == 30.0
    assert memory["nvidia_smi"]["1"]["free_at_peak_gib"] == pytest.approx(3.0)
    assert memory["min_free_gib"] == pytest.approx(3.0) and memory["below_free_margin"]
    assert memory["host"] == {"node_used_at_start_gib": 10.0, "node_peak_used_gib": 80.0}
    run1 = out["run1_estimate"]
    assert out["sequences_per_step"] == 32 and run1["steps"] == 799
    assert run1["training_hours"] == pytest.approx(120224680 / (1200 / 30 * 3600))
    assert run1["save_every_steps_for"] == {"2h": 720, "3h": 1080, "4h": 1440}
