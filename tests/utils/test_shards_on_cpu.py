from pathlib import Path

from soe.utils.shards import claim_shard, mark_done, shard_path, unfinished_shards


def test_a_shard_is_claimed_once_and_finished_shards_are_tracked(tmp_path: Path) -> None:
    assert claim_shard(tmp_path, 0, "job 1") and not claim_shard(tmp_path, 0, "job 2")
    assert (tmp_path / "claims" / "shard-0" / "owner").read_text() == "job 1\n"
    assert unfinished_shards(tmp_path, 3) == [0, 1, 2]
    mark_done(tmp_path, 0)
    mark_done(tmp_path, 2)  # an explicit --shard run that never claimed
    assert unfinished_shards(tmp_path, 3) == [1]
    assert shard_path(tmp_path, 1, 3).name == "samples-1-of-3.jsonl"
