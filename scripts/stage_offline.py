# /// script
# requires-python = ">=3.11"
# # socksio is needed only behind a SOCKS proxy (e.g. the agent sandbox); harmless elsewhere.
# dependencies = ["huggingface_hub>=0.34", "pydantic>=2.9,<3", "pyyaml>=6.0", "socksio"]
# ///
"""Download every model and dataset listed in an assets file into $HF_HOME, for offline jobs.

Run where there is internet: the FASTER login node (downloads are light work there) or your
machine followed by rsync. Uses PEP 723 inline deps, so it needs only uv:

    HF_HOME=$SOE_DATA/hf uv run scripts/stage_offline.py --assets scripts/stage_assets.yaml

Writes $HF_HOME/../manifest.json with the resolved commit sha and file list of each asset. LFS
files in the HF cache are stored under their sha256, which the manifest records as the checksum.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel, ConfigDict


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_id: str
    repo_type: Literal["model", "dataset"]
    revision: str
    note: str = ""


class AssetsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assets: list[Asset]


def file_entries(snapshot: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        blob = path.resolve()
        entries.append(
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": blob.stat().st_size,
                "blob": blob.name,  # sha256 for LFS files, git sha1 for small files
            }
        )
    return entries


def stage(asset: Asset, api: HfApi) -> dict[str, object]:
    sha = api.repo_info(asset.hf_id, repo_type=asset.repo_type, revision=asset.revision).sha
    snapshot = Path(
        snapshot_download(
            asset.hf_id,
            repo_type=asset.repo_type,
            revision=sha,
        )
    )
    files = file_entries(snapshot)
    total_gb = sum(int(str(f["bytes"])) for f in files) / 1e9
    print(
        f"staged {asset.repo_type} {asset.hf_id}@{sha[:12]}: {len(files)} files, {total_gb:.2f} GB"
    )
    return {
        "hf_id": asset.hf_id,
        "repo_type": asset.repo_type,
        "requested_revision": asset.revision,
        "commit_sha": sha,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="stage only these hf_ids")
    args = parser.parse_args()
    if "HF_HOME" not in os.environ:
        raise SystemExit(
            "set HF_HOME to $SOE_DATA/hf first (never stage into home: 10k-file quota)"
        )
    assets = AssetsFile.model_validate(yaml.safe_load(args.assets.read_text())).assets
    selected = [a for a in assets if not args.only or a.hf_id in args.only]
    api = HfApi()
    manifest_path = Path(os.environ["HF_HOME"]).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"assets": {}}
    for asset in selected:
        manifest["assets"][asset.hf_id] = stage(asset, api)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
