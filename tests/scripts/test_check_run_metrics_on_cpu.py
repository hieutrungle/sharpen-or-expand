"""The run checker passes a complete run and names what is missing in an incomplete one (M1.6)."""

import importlib.util
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

from soe.trainer.main_metrics import MetricsCli
from soe.trainer.rl.metrics import RolloutOutcome
from soe.trainer.rl.segments import SegmentEvent, append_event
from soe.utils.run import RunEfficiency, RunInfo

CONFIG = Path("examples/grpo_trainer/m1_smoke_qwen35-2b.yaml")  # 16 prompts x 8, 3 steps
STEPS, PROMPTS, G = 2, 16, 8
T0 = datetime(2026, 10, 29, tzinfo=UTC)


def checker() -> ModuleType:
    path = Path("scripts/check_run_metrics.py")
    spec = importlib.util.spec_from_file_location("check_run_metrics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(run_dir: Path, verl_keys: tuple[str, ...]) -> None:
    """A finished two-step run as main_rl, the agent loop, the reward hook and verl leave it."""
    run_dir.mkdir()
    shutil.copyfile(CONFIG, run_dir / "config.yaml")
    info = RunInfo(
        run_id=run_dir.name,
        config_name="grpo_m1_smoke_qwen35-2b",
        config_sha256="0" * 64,
        seed=1,
        git_commit="abc",
        git_dirty=False,
        created_at=T0,
        hostname="fc096",
        slurm_job_id="1",
        gpus=["NVIDIA A40, 49140 MiB"] * 2,
        library_versions={"verl": "0.9.1"},
    )
    (run_dir / "run.json").write_text(info.model_dump_json())
    (run_dir / "efficiency.json").write_text(RunEfficiency().model_dump_json())
    common = {"segment": "1", "node": "fc096", "hardware": "2 NVIDIA A40", "git_commit": "abc"}
    append_event(
        run_dir,
        SegmentEvent(
            event="start", at=T0, resumed_from_step=0, library_versions={"verl": "0.9.1"}, **common
        ),
    )
    append_event(
        run_dir, SegmentEvent(event="end", at=T0 + timedelta(hours=2), exit="done", **common)
    )
    for name in ("metrics", "rollout_tokens", "rollouts"):
        (run_dir / name).mkdir()
    verl, tokens, outcomes = [], [], []
    for step in range(1, STEPS + 1):
        lengths = [100 * step + i for i in range(PROMPTS * G)]
        scores = [float(i % 3 == 0) for i in range(PROMPTS * G)]
        for i, (length, score) in enumerate(zip(lengths, scores, strict=True)):
            tokens.append({"step": step, "tokens": length})
            outcome = RolloutOutcome(
                step=step,
                problem_id=f"p{i // G}",
                score=score,
                truncated=False,
                has_box=True,
                turn_end=True,
                answer=str(i % 2),
            )
            outcomes.append(outcome.model_dump())
        data = {key: 1.0 for key in verl_keys}
        data["critic/score/mean"] = sum(scores) / len(scores)
        data["response_length/mean"] = sum(lengths) / len(lengths)
        data["perf/total_num_tokens"] = 1000.0
        verl.append({"step": step, "data": data})
    for name, lines in (
        ("metrics", verl),
        ("rollout_tokens", tokens),
        ("rollouts", outcomes),
    ):
        text = "".join(json.dumps(x) + "\n" for x in lines)
        (run_dir / name / "segment-1.jsonl").write_text(text)


def test_complete_run_passes_and_a_missing_key_is_named(tmp_path: Path) -> None:
    check = checker()
    run = tmp_path / "run"
    fake_run(run, check.VERL_KEYS)
    MetricsCli(run_dir=run, through_step=STEPS).cli_cmd()
    efficiency = RunEfficiency.model_validate_json((run / "efficiency.json").read_text())
    assert (efficiency.through_step, efficiency.training_tokens) == (2, 2000)
    assert efficiency.gpu_hours == 2 * efficiency.wall_clock_hours > 0
    assert check.check_run(run) == []

    # Remove one verl key from step 2: the checker names it.
    path = run / "metrics" / "segment-1.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    del records[1]["data"]["actor/entropy"]
    path.write_text("".join(json.dumps(x) + "\n" for x in records))
    assert check.check_run(run) == ["step 2: verl metric actor/entropy missing"]


def test_a_lost_rollout_fails_the_run(tmp_path: Path) -> None:
    check = checker()
    run = tmp_path / "run"
    fake_run(run, check.VERL_KEYS)
    MetricsCli(run_dir=run, through_step=STEPS).cli_cmd()
    path = run / "rollouts" / "segment-1.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))  # drop one
    assert check.check_run(run) == ["step 2: 127 outcome records, expected 128"]
