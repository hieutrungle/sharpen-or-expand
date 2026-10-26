"""Offline compute nodes need complete, pinned snapshots of every staged asset."""

from pathlib import Path

import yaml


def test_every_asset_is_pinned_and_staged_complete() -> None:
    assets = yaml.safe_load(Path("scripts/stage_assets.yaml").read_text())["assets"]
    assert assets
    for asset in assets:
        assert len(asset["revision"]) == 40, f"{asset['hf_id']}: pin a full commit sha"
        # A filtered download leaves the snapshot incomplete, and offline vLLM rejects it.
        assert "allow_patterns" not in asset, f"{asset['hf_id']}: stage the complete repo"
