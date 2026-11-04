import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from soe.trainer.main_sft import SftLaunchCli, resume_point, worker_command
from soe.trainer.rl.segments import SegmentEvent, append_event


def test_worker_runs_under_torchrun_on_every_gpu() -> None:
    command = worker_command(Path("/runs/r"), 2)
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in command
    assert command[-3:] == ["soe.trainer.sft.trainer", "--run-dir", "/runs/r"]


def start(run_dir: Path) -> None:
    event = SegmentEvent(
        event="start",
        segment="1",
        at=datetime(2026, 11, 4, tzinfo=UTC),
        node="fc096",
        hardware="2 NVIDIA A40",
        git_commit="abc",
    )
    append_event(run_dir, event)


def test_a_started_run_resumes_from_its_newest_complete_checkpoint(tmp_path: Path) -> None:
    assert resume_point(tmp_path) is None  # never started: fresh
    start(tmp_path)
    assert resume_point(tmp_path) is None  # started, nothing saved: from step 0 again
    saved = tmp_path / "trainer" / "checkpoint-6"
    saved.mkdir(parents=True)
    (saved / "trainer_state.json").write_text("{}")
    resume = resume_point(tmp_path)
    assert resume is not None and resume.step == 6
    command = worker_command(tmp_path, 2, resume)
    assert command[-2:] == ["--resume-from", str(saved)]


def test_a_finished_run_is_left_alone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "config.yaml").write_text(
        Path("examples/sft/m2_dev_full_qwen35-2b.yaml").read_text()
    )
    (tmp_path / "DONE").write_text("finished in job 1\n")
    SftLaunchCli(run_dir=tmp_path).cli_cmd()  # returns before the GPU check or any segment event
    assert "already finished" in capsys.readouterr().out
    assert not (tmp_path / "segments.jsonl").exists()
