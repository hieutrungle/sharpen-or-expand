from pathlib import Path

import pytest

from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl, write_jsonl


def sample(i: int) -> SampleRecord:
    return SampleRecord(
        problem_id="p1",
        sample_index=i,
        checkpoint="base",
        completion="x",
        extracted_answer="1",
        correct=True,
        completion_tokens=3,
        finish_reason="length",
    )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    assert write_jsonl(path, [sample(0), sample(1)]) == 2
    write_jsonl(path, [sample(2)], append=True)
    assert read_jsonl(path, SampleRecord) == [sample(0), sample(1), sample(2)]


def test_jsonl_error_names_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write_jsonl(path, [sample(0)])
    with path.open("a") as f:
        f.write('{"problem_id": "p1"}\n')
    with pytest.raises(ValueError, match=r"s.jsonl:2"):
        read_jsonl(path, SampleRecord)
