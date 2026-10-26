"""JSONL read/write for any Pydantic record (rollout samples, eval records, problems)."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ValidationError


def write_jsonl[M: BaseModel](path: Path, records: Iterable[M], *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a" if append else "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    records: list[M] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as e:
                raise ValueError(f"{path}:{line_number}: {e}") from e
    return records
