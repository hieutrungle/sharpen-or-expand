import json
from pathlib import Path

import pytest

from soe.trainer.config.eval_suite import file_sha256
from soe.trainer.main_rl import checked_data_file, frozen_overrides


def test_data_file_must_match_its_hash(tmp_path: Path) -> None:
    (tmp_path / "train.parquet").write_bytes(b"rows")
    good = file_sha256(tmp_path / "train.parquet")
    assert checked_data_file(tmp_path, "train.parquet", good) == tmp_path / "train.parquet"
    with pytest.raises(ValueError, match="sha256"):
        checked_data_file(tmp_path, "train.parquet", "0" * 64)
    with pytest.raises(ValueError, match="missing"):
        checked_data_file(tmp_path, "nope.parquet", good)


def test_resumed_job_must_rebuild_the_same_overrides(tmp_path: Path) -> None:
    first = ["a=1", "b=2"]
    assert frozen_overrides(tmp_path, first) == first
    assert json.loads((tmp_path / "verl_overrides.json").read_text()) == first
    assert frozen_overrides(tmp_path, list(first)) == first
    with pytest.raises(ValueError, match="differ"):
        frozen_overrides(tmp_path, ["a=1", "b=3"])
