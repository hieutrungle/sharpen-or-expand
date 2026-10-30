import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from soe.trainer.main_sft import SftLaunchCli, worker_command
from soe.trainer.rl.segments import SegmentEvent, append_event


def test_worker_runs_under_torchrun_on_every_gpu() -> None:
    command = worker_command(Path("/runs/r"), 2)
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in command
    assert command[-3:] == ["soe.trainer.sft.trainer", "--run-dir", "/runs/r"]


def test_a_started_sft_run_is_not_trained_again(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(Path("examples/sft/m1_dev_qwen35-2b.yaml").read_text())
    event = SegmentEvent(
        event="start",
        segment="1",
        at=datetime(2026, 10, 29, tzinfo=UTC),
        node="fc096",
        hardware="2 NVIDIA A40",
        git_commit="abc",
    )
    append_event(tmp_path, event)
    with pytest.raises(RuntimeError, match="SFT resume is not supported"):
        SftLaunchCli(run_dir=tmp_path).cli_cmd()
