import subprocess
from typing import Any

import pytest

from soe.utils import run
from soe.utils.run import check_gpu_count, visible_gpus

LISTING = """GPU 0: NVIDIA A30 (UUID: GPU-1111)
GPU 1: NVIDIA A30 (UUID: GPU-2222)
"""


def test_gpus_are_counted_from_nvidia_smi_listing() -> None:
    assert visible_gpus(LISTING) == 2
    assert visible_gpus("") == 0
    assert visible_gpus("No devices were found\n") == 0


def test_a_job_with_another_gpu_count_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=LISTING, stderr="")

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    check_gpu_count(2)
    with pytest.raises(RuntimeError, match="asks for 4 GPUs but this job has 2"):
        check_gpu_count(4)
