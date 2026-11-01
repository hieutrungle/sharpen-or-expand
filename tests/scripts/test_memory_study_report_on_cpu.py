"""The M1.10 report reads a run's phase marks, sidecar samples, logs and metrics correctly."""

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from soe.workers.rollout.phase_memory import PhaseMark

CELL = Path("examples/grpo_trainer/m1_memory_bl_qwen35-4b.yaml")
T0 = datetime(2026, 10, 31, 12, tzinfo=UTC)


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(f"scripts/{name}.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("memory_study_report")
probe = load("memory_probe_config")


def mark(phase: str, minute: int, rank: int = 0, peak: float = 10.0) -> PhaseMark:
    return PhaseMark(
        phase=phase,
        rank=rank,
        ended_at=T0 + timedelta(minutes=minute),
        max_allocated_gib=peak,
        max_reserved_gib=peak + 1,
        allocated_gib=5.0,
    )


ONE_STEP = [  # weight sync 0-1, generation 1-20, old log-prob 20-25, update 26-40
    ("before_update_weights", 0, 6.0),
    ("update_weights", 1, 7.0),
    ("before_compute_log_prob", 20, 8.0),
    ("compute_log_prob", 25, 20.0),
    ("before_update_actor", 26, 6.0),
    ("update_actor", 40, 30.0),
]

LOG = """(vLLMHttpServer pid=11) INFO [kv_cache_utils.py] GPU KV cache size: 1,234,560 tokens
(vLLMHttpServer pid=11) INFO Maximum concurrency for 18,432 tokens per request: 66.98x
(vLLMHttpServer pid=11) INFO Available KV cache memory: 21.50 GiB
(vLLMHttpServer pid=11) INFO Engine 000: Avg prompt throughput: 1.0 tokens/s, Running: 128 reqs, \
Waiting: 0 reqs, GPU KV cache usage: 41.0%, Prefix cache hit rate: 0.0%
(vLLMHttpServer pid=11) INFO Engine 000: Avg prompt throughput: 1.0 tokens/s, Running: 100 reqs, \
Waiting: 28 reqs, Preemptions: 3, GPU KV cache usage: 99.5%, Prefix cache hit rate: 0.0%
(vLLMHttpServer pid=11) INFO Engine 000: Running: 90 reqs, Waiting: 0 reqs, Preemptions: 2, \
GPU KV cache usage: 80.0%, Prefix cache hit rate: 0.0%
"""


def make_probe_run(root: Path, tp: int, gmu: float, finished: bool) -> Path:
    data = probe.probe_config(yaml.safe_load(CELL.read_text()), tp, gmu)
    run = root / data["name"]
    (run / "metrics").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(data))
    (run / "hardware.txt").write_text("4 NVIDIA A40, 46068 MiB\n")
    (run / "logs").mkdir()
    (run / "phase_memory").mkdir()
    marks = ONE_STEP if finished else ONE_STEP[:3]  # a failed probe stops in old log-prob
    for rank in (0, 1):
        lines = [mark(p, m, rank, peak).model_dump_json() for p, m, peak in marks]
        (run / "phase_memory" / f"segment-9-rank{rank}.jsonl").write_text("\n".join(lines) + "\n")
    gpu_rows = [  # GPU 0 peaks at 40,000 MiB in generation, 30,000 in the update
        f"{(T0 + timedelta(minutes=m)).isoformat().replace('+00:00', 'Z')},0,{used},46068,90"
        for m, used in ((10, 40000), (30, 30000), (45, 1000))
    ]
    (run / "gpu_memory").mkdir()
    (run / "gpu_memory" / "segment-9.csv").write_text("\n".join(gpu_rows) + "\n0,12\n")
    (run / "host_memory").mkdir()
    host = f"{T0.isoformat()},262144000,157286400\n"  # 250 GiB total, 150 GiB available
    (run / "host_memory" / "segment-9.csv").write_text(host)
    log = (
        LOG
        if finished
        else LOG + "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.50 GiB\n"
    )
    (run / "logs" / "segment-9.log").write_text(log)
    if finished:
        record = {
            "step": 1,
            "data": {
                "timing_s/gen": 1000.0,
                "timing_s/old_log_prob": 200.0,
                "timing_s/update_actor": 600.0,
                "timing_s/step": 2000.0,
                "timing_s/testing": 50.0,
                "timing_s/save_checkpoint": 150.0,
                "prompt_length/mean": 100.0,
                "response_length/mean": 11000.0,
            },
        }
        (run / "metrics" / "segment-9.jsonl").write_text(json.dumps(record) + "\n")
        (run / "rollout_tokens").mkdir()
        tokens = [{"step": 1, "tokens": 10000}] * 512
        (run / "rollout_tokens" / "segment-9.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in tokens)
        )
        (run / "DONE").write_text("finished\n")
    return run


def test_probe_is_one_step_with_a_token_validation_and_the_cells_other_fields() -> None:
    cell = yaml.safe_load(CELL.read_text())
    data = probe.probe_config(cell, 2, 0.7)
    assert data["name"] == "m1_memory_bl_qwen35-4b-probe-tp2-gmu70"
    # two steps: AdamW's state exists only after step 1's first update
    assert data["budget"]["max_steps"] == 2 and data["checkpoint"]["save_every_steps"] == 2
    assert data["validation"] == {"max_problems": 8, "samples_per_problem": 1}
    assert data["hardware"] == cell["hardware"] | {
        "rollout_tensor_parallel": 2,
        "gpu_memory_utilization": 0.7,
    }
    assert {
        k: v
        for k, v in data.items()
        if k not in ("name", "budget", "checkpoint", "validation", "hardware")
    } == {
        k: v
        for k, v in cell.items()
        if k not in ("name", "budget", "checkpoint", "validation", "hardware")
    }


def test_kv_cache_lines_and_preemptions_add_up() -> None:
    kv = report.kv_cache(LOG)
    assert kv["kv_cache_tokens_per_replica"] == [1234560]
    assert kv["max_concurrency_per_replica"] == [66.98]
    assert kv["kv_cache_gib_per_replica"] == [21.5]
    assert kv["peak_kv_cache_usage_pct"] == 99.5
    assert kv["preemptions"] == 5


def test_phase_windows_assign_marks_to_steps() -> None:
    marks = [mark(p, m) for p, m, _ in ONE_STEP] + [
        mark("update_weights", 41),
        mark("before_compute_log_prob", 60),
    ]
    windows = report.phase_windows(marks)
    by = {(w["step"], w["phase"]): (w["start"], w["end"]) for w in windows}
    assert by[(1, "generate")] == (T0 + timedelta(minutes=1), T0 + timedelta(minutes=20))
    assert by[(1, "update")] == (T0 + timedelta(minutes=26), T0 + timedelta(minutes=40))
    assert (2, "weight_sync") in by and (2, "generate") in by


def test_finished_probe_rates_memory_and_headroom(tmp_path: Path) -> None:
    run: dict[str, Any] = report.run_report(make_probe_run(tmp_path, 1, 0.6, finished=True))
    step = run["steps"][1]
    assert step["rollout_tokens"] == 5_120_000
    assert step["rollout_tokens_per_s"] == 5120.0
    assert step["training_tokens_per_s"] == (5_120_000 + 100 * 512) / 800
    assert step["step_s_without_validation_and_save"] == 1800.0
    assert run["torch_peak_by_phase"]["update"][0]["max_allocated_gib"] == 30.0
    assert run["torch_peak_by_phase"]["generate"][1]["max_allocated_gib"] == 8.0
    assert run["nvidia_smi_peak"]["by_phase_mib"]["generate"] == {0: 40000.0}
    assert run["nvidia_smi_peak"]["by_phase_mib"]["update"] == {0: 30000.0}
    assert run["host_ram_peak_gib"] == 100.0
    kv = run["kv_cache"]
    assert kv["responses_per_replica"] == 128 and kv["max_model_len"] == 18432
    assert kv["headroom"] == [1234560 / (128 * 18432)]
    assert "failure" not in run


def test_failed_probe_names_the_phase_and_the_failed_allocation(tmp_path: Path) -> None:
    run = report.run_report(make_probe_run(tmp_path, 1, 0.7, finished=False))
    assert run["failure"]["phase"] == "old_log_prob"
    assert any(
        "Tried to allocate 15.50 GiB" in line for line in run["failure"]["out_of_memory_lines"]
    )


def test_grid_marks_lower_values_dominated_and_picks_the_fastest_fit(tmp_path: Path) -> None:
    runs = [
        make_probe_run(tmp_path, 1, 0.7, finished=False),
        make_probe_run(tmp_path, 1, 0.6, finished=True),
        make_probe_run(tmp_path, 2, 0.7, finished=True),
    ]
    study = report.study_report(runs)
    (cell,) = study["cells"]
    assert cell["cell"] == "B-L"
    status = {(p["tp"], p["gpu_memory_utilization"]): p["status"] for p in cell["grid"]}
    assert status[(1, 0.7)] == "failed" and status[(1, 0.6)] == "fits"
    assert status[(1, 0.5)] == status[(1, 0.4)] == "not run, dominated"
    assert status[(2, 0.7)] == "fits" and status[(2, 0.4)] == "not run, dominated"
    assert cell["chosen"]["gpu_memory_utilization"] in (0.6, 0.7)  # equal speed in the fixture


def test_a_probe_reports_only_its_last_step(tmp_path: Path) -> None:
    run = make_probe_run(tmp_path, 1, 0.6, finished=True)
    first = json.loads((run / "metrics" / "segment-9.jsonl").read_text())
    second = {"step": 2, "data": first["data"] | {"timing_s/gen": 500.0}}
    with (run / "metrics" / "segment-9.jsonl").open("a") as out:
        out.write(json.dumps(second) + "\n")
    with (run / "rollout_tokens" / "segment-9.jsonl").open("a") as out:
        out.write("".join(json.dumps({"step": 2, "tokens": 10000}) + "\n" for _ in range(512)))
    result = report.run_report(run)
    assert set(result["steps"]) == {2}
    assert report.probe_speed(result) == 10240.0
