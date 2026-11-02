"""The M1.11 report: R over the job segments, its spread, the budget table and the length fit."""

import importlib.util
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.rl.segments import SegmentEvent, append_event

CONFIG = Path("examples/grpo_trainer/m1_throughput_bf0_qwen35-4b.yaml")
T0 = datetime(2026, 11, 2, tzinfo=UTC)


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "throughput_report", Path("scripts/throughput_report.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load()


def event(kind: str, segment: str, hours: float) -> SegmentEvent:
    return SegmentEvent(
        event=kind,  # type: ignore[arg-type]
        segment=segment,
        at=T0 + timedelta(hours=hours),
        node="fc098",
        hardware="4 NVIDIA A40, 49140 MiB",
        git_commit="abc",
        resumed_from_step=0 if kind == "start" else None,
        exit="done" if kind == "end" else None,
    )


def make_run(tmp_path: Path, step_s: dict[int, float], lengths: dict[int, float]) -> Path:
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    (run / "rollout_tokens").mkdir()
    (run / "config.yaml").write_text(CONFIG.read_text())
    (run / "hardware.txt").write_text("4 NVIDIA A40, 49140 MiB\n")
    records, tokens = [], []
    for s, seconds in step_s.items():
        data = {"timing_s/step": seconds, "response_length/mean": lengths[s]}
        if s % 3 == 0:
            data |= {"timing_s/testing": 360.0, "timing_s/save_checkpoint": 20.0}
        records.append({"step": s, "data": data})
        tokens += [{"step": s, "tokens": 1000}] * 512  # 512,000 tokens per step
    (run / "metrics" / "segment-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (run / "rollout_tokens" / "segment-1.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tokens)
    )
    # two segments, as a resumed run: 1.5 h + 1.0 h of wall-clock, 30 min queued between
    for kind, seg, h in (
        ("start", "1", 0),
        ("end", "1", 1.5),
        ("start", "2", 2.0),
        ("end", "2", 3.0),
    ):
        append_event(run, event(kind, seg, h))
    return run


def test_rate_is_tokens_over_segment_hours_and_the_budget_table_follows() -> None:
    rows = report.budget_table(10e6, 0.1)
    assert [r["rollout_tokens"] for r in rows] == [30e6, 50e6, 100e6, 200e6]
    first = rows[0]
    assert first["hours"] == pytest.approx(3.0)
    assert first["hours_range"] == pytest.approx([30e6 / 11e6, 30e6 / 9e6])
    assert first["jobs"] == 1 and first["hours_with_queue_factor"] == pytest.approx(3.75)
    assert report.budget_table(1e6, 0.0)[3]["jobs"] == math.ceil(200 / 66)


def test_run_throughput(tmp_path: Path) -> None:
    # step seconds = 200 + 0.1 x length exactly, so the fit recovers a = 200, b = 0.1
    lengths = {1: 9000.0, 2: 8000.0, 3: 9000.0, 4: 10000.0, 5: 11000.0}
    step_s = {s: 200 + 0.1 * length for s, length in lengths.items()}
    run = report.run_throughput(make_run(tmp_path, step_s, lengths))
    assert run["wall_clock_hours"] == pytest.approx(2.5)  # queue time between segments excluded
    assert run["rollout_tokens"] == 5 * 512_000
    assert run["rate_tokens_per_hour"] == pytest.approx(5 * 512_000 / 2.5)
    per_step = [512_000 / step_s[s] * 3600 for s in (2, 3, 4, 5)]
    assert run["rate_spread"] == pytest.approx(
        (sum((x - sum(per_step) / 4) ** 2 for x in per_step) / 3) ** 0.5 / (sum(per_step) / 4)
    )
    assert run["validation_passes_s"] == {3: 360.0}
    split = run["time_split_hours"]
    assert split["steps"] == pytest.approx(sum(step_s.values()) / 3600)
    assert split["steps"] + split["validation"] + split["other"] == pytest.approx(2.5)
    fit = run["length_fit"]
    assert fit["a"] == pytest.approx(200) and fit["b"] == pytest.approx(0.1)
    base = {r["length_factor"]: r for r in run["length_sensitivity"]}
    assert base[1.0]["mean_response_length"] == 9000.0  # step 1's length
    rate = lambda length: length / (200 + 0.1 * length)  # noqa: E731
    assert base[0.5]["rate_tokens_per_hour"] == pytest.approx(
        run["rate_tokens_per_hour"] * rate(4500) / rate(9500)
    )
    assert base[0.5]["extrapolated"] and base[1.5]["extrapolated"]
    assert not base[1.0]["extrapolated"]


def test_a_segment_without_an_end_is_refused(tmp_path: Path) -> None:
    run = make_run(
        tmp_path, {s: 1000.0 + s for s in range(1, 5)}, dict.fromkeys(range(1, 5), 9000.0)
    )
    append_event(run, event("start", "3", 4.0))  # a killed job writes no end event
    with pytest.raises(ValueError, match="without an end event"):
        report.run_throughput(run)
