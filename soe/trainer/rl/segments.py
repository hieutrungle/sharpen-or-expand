"""Job segments of one RL run: provenance, training hours and merged metrics (M1.5).

A run is trained by a chain of Slurm jobs ("segments"), each resuming from the latest checkpoint.
main_rl appends a start and an end event per segment to `runs/<id>/segments.jsonl`. Per-segment
files (`metrics/segment-<job>.jsonl`, `rollout_tokens/segment-<job>.jsonl`) are ordered by job id;
for a step that two segments ran (the killed one, then its successor replaying it), the later
segment's records replace the earlier ones.

No verl or torch import, so this runs in the CPU tests.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

SEGMENTS = "segments.jsonl"
# Set by main_rl for the verl processes: when this job segment started and the step it resumed from.
SEGMENT_STARTED_ENV = "SOE_SEGMENT_STARTED"
RESUMED_FROM_ENV = "SOE_RESUMED_FROM_STEP"


class SegmentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["start", "end"]
    segment: str  # Slurm job id, or "local"
    at: datetime
    node: str
    hardware: str  # the run's hardware.txt line
    git_commit: str
    resumed_from_step: int | None = None  # start: the checkpoint resumed from (0 = fresh run)
    exit: Literal["done", "failed"] | None = None  # end only; a killed job writes no end event


def append_event(run_dir: Path, event: SegmentEvent) -> None:
    with open(run_dir / SEGMENTS, "a") as f:
        f.write(event.model_dump_json() + "\n")


def read_events(run_dir: Path) -> list[SegmentEvent]:
    path = run_dir / SEGMENTS
    if not path.exists():
        return []
    return [SegmentEvent.model_validate_json(x) for x in path.read_text().splitlines() if x.strip()]


def segment_order(path: Path) -> int:
    """Sort key of a `segment-<job>.jsonl` file: Slurm job ids increase with submission order."""
    job = path.stem.removeprefix("segment-")
    return int(job) if job.isdigit() else -1


def training_hours(run_dir: Path, now: datetime) -> float:
    """Wall-clock hours of every segment so far. A segment without an end event (killed, or the
    current one) ends at its last rollout-token record, or now for the current segment."""
    starts: dict[str, datetime] = {}
    ends: dict[str, datetime] = {}
    for event in read_events(run_dir):
        (starts if event.event == "start" else ends)[event.segment] = event.at
    hours = 0.0
    for segment, start in starts.items():
        end = ends.get(segment)
        if end is None:
            tokens = run_dir / "rollout_tokens" / f"segment-{segment}.jsonl"
            is_last = segment == max(starts, key=lambda s: starts[s])
            if is_last:
                end = now
            elif tokens.exists():
                end = datetime.fromtimestamp(tokens.stat().st_mtime, tz=UTC)
            else:
                end = start
        hours += max(0.0, (end - start).total_seconds() / 3600)
    return hours


def merged_metrics(metrics_dir: Path) -> dict[int, dict[str, Any]]:
    """verl's per-step records over all segments: within a segment the records of a step are
    combined (validation-before-training and the step's own record); a later segment's step
    replaces an earlier segment's."""
    merged: dict[int, dict[str, Any]] = {}
    for path in sorted(metrics_dir.glob("segment-*.jsonl"), key=segment_order):
        in_segment: dict[int, dict[str, Any]] = defaultdict(dict)
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                in_segment[int(record["step"])].update(record.get("data", {}))
        merged.update(in_segment)
    return dict(sorted(merged.items()))
