"""SFT resume bookkeeping (decision 028): which trainer checkpoint a job resumes from, and the
per-step log of trained example ids that shows the data order continues across a resume.

- The Trainer writes `trainer/checkpoint-<step>/` with the model, optimizer, scheduler and RNG
  state, then `trainer_state.json` last; a checkpoint without it was cut off and is skipped.
- Every training micro-batch appends `{"step", "examples"}` to
  `examples/segment-<job>-rank<r>.jsonl`; `step` is the optimizer step it belongs to and
  `examples` the train-set indices. A resumed job redoes the steps after its checkpoint, so a step
  logged by two segments must list the same examples on every rank (a killed job may have logged
  only the first micro-batches of its last step: then one list is a prefix of the other).

No torch import, so this runs in the CPU tests.
"""

import hashlib
import json
import re
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TRAINER_DIR = "trainer"
EXAMPLES_DIR = "examples"
STATE_FILE = "trainer_state.json"
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
_LOG = re.compile(r"^segment-(?P<segment>.+)-rank(?P<rank>\d+)\.jsonl$")


@dataclass(frozen=True, slots=True, kw_only=True)
class SavedCheckpoint:
    step: int
    path: Path


def last_complete_checkpoint(trainer_dir: Path) -> SavedCheckpoint | None:
    found = []
    for path in trainer_dir.glob("checkpoint-*") if trainer_dir.is_dir() else []:
        match = _CHECKPOINT.match(path.name)
        if match and (path / STATE_FILE).is_file():
            found.append(SavedCheckpoint(step=int(match.group(1)), path=path))
    return max(found, key=lambda c: c.step, default=None)


def example_key(input_ids: Sequence[int]) -> str:
    """Identifies a tokenized example (kept completions of a problem differ, so keys are unique)."""
    return hashlib.blake2b(array("q", input_ids).tobytes(), digest_size=16).hexdigest()


def log_path(run_dir: Path, segment: str, rank: int) -> Path:
    return run_dir / EXAMPLES_DIR / f"segment-{segment}-rank{rank}.jsonl"


def read_example_log(run_dir: Path) -> dict[str, dict[tuple[int, int], list[int]]]:
    """segment -> (step, rank) -> the example indices that step trained on, in micro-batch order."""
    out: dict[str, dict[tuple[int, int], list[int]]] = {}
    for path in sorted((run_dir / EXAMPLES_DIR).glob("segment-*.jsonl")):
        match = _LOG.match(path.name)
        if match is None:
            continue
        steps = out.setdefault(match.group("segment"), {})
        rank = int(match.group("rank"))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            steps.setdefault((row["step"], rank), []).extend(row["examples"])
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class Continuity:
    redone_steps: list[int]  # optimizer steps trained by more than one segment
    mismatched_steps: list[int]  # of those, steps whose examples differ (beyond a cut-off prefix)


def continuity(run_dir: Path) -> Continuity:
    """Across a resume, the redone steps must train on the same examples on every rank."""
    seen: dict[tuple[int, int], list[int]] = {}
    redone: set[int] = set()
    mismatched: set[int] = set()
    for steps in read_example_log(run_dir).values():
        for key, examples in steps.items():
            if key not in seen:
                seen[key] = examples
                continue
            redone.add(key[0])
            short, long = sorted((seen[key], examples), key=len)
            if long[: len(short)] != short:
                mismatched.add(key[0])
            seen[key] = long
    return Continuity(redone_steps=sorted(redone), mismatched_steps=sorted(mismatched))
