"""Peak GPU memory per training phase in verl's actor worker (M1.10 memory study).

verl 0.9.1 logs one running peak per step (`perf/max_memory_*`, never reset), so it cannot say
which phase needs the memory. `verl_patches` wraps three `ActorRolloutRefWorker` methods with
`wrap`: at each entry and each return, `close_interval` writes torch's peak allocated and reserved
memory since the previous mark, then resets the peak. A mark named `<method>` covers the method's
own run; `before_<method>` covers the time since the previous mark. The one that ends at the entry
of `compute_log_prob` is generation (with reward and any validation pass). vLLM allocates outside
torch's caching allocator, so its memory is in the sidecar's nvidia-smi samples (slurm/common.sh),
not here. A side effect: verl's own `perf/max_memory_*` becomes the peak of the update phase.

One JSONL file per rank under $SOE_PHASE_MEMORY (set by main_rl); nothing is written when unset.
CPU-importable: torch is imported only inside `close_interval`.
"""

import functools
import inspect
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

PHASE_MEMORY_ENV = "SOE_PHASE_MEMORY"
PHASE_MEMORY_DIR = "phase_memory"
WRAPPED_METHODS = ("update_weights", "compute_log_prob", "update_actor")
GIB = 1024**3


class PhaseMark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str
    rank: int
    ended_at: datetime
    max_allocated_gib: float
    max_reserved_gib: float
    allocated_gib: float  # at the mark: what stays resident into the next interval


def mark_path(directory: Path, segment: str, rank: int) -> Path:
    return directory / f"segment-{segment}-rank{rank}.jsonl"


def close_interval(phase: str) -> None:
    directory = os.environ.get(PHASE_MEMORY_ENV)
    if not directory:
        return
    import torch

    if not torch.cuda.is_available():
        return
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    mark = PhaseMark(
        phase=phase,
        rank=rank,
        ended_at=datetime.now(UTC),
        max_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
        max_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
        allocated_gib=torch.cuda.memory_allocated() / GIB,
    )
    path = mark_path(Path(directory), os.environ.get("SLURM_JOB_ID", "local"), rank)
    with path.open("a") as out:
        out.write(mark.model_dump_json() + "\n")
    torch.cuda.reset_peak_memory_stats()


def wrap(method: Callable[..., Any], name: str) -> Callable[..., Any]:
    """`method` with a mark before and after it; functools.wraps keeps verl's @register data."""
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def run_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            close_interval(f"before_{name}")
            result = await method(self, *args, **kwargs)
            close_interval(name)
            return result

        run_async.soe_phase_memory = True  # type: ignore[attr-defined]
        return run_async

    @functools.wraps(method)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        close_interval(f"before_{name}")
        result = method(self, *args, **kwargs)
        close_interval(name)
        return result

    run.soe_phase_memory = True  # type: ignore[attr-defined]
    return run
