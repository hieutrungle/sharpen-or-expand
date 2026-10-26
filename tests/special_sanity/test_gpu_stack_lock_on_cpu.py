"""uv.lock must pin a consistent GPU stack; a mismatch would only show up on a GPU node."""

import tomllib
from pathlib import Path


def locked_packages() -> dict[str, dict[str, object]]:
    lock = tomllib.loads(Path("uv.lock").read_text())
    return {package["name"]: package for package in lock["package"]}


def test_flash_attn_build_matches_locked_torch() -> None:
    packages = locked_packages()
    torch_version = str(packages["torch"]["version"]).split("+")[0]  # e.g. "2.11.0"
    source = packages["flash-attn"]["source"]
    assert isinstance(source, dict) and "url" in source, "flash-attn must be pinned by direct URL"
    assert f"torch{torch_version}/" in str(source["url"]), (
        f"flash-attn wheel {source['url']} was not built for torch {torch_version}"
    )


def test_flash_attn_wheel_is_hash_pinned() -> None:
    wheels = locked_packages()["flash-attn"]["wheels"]
    assert isinstance(wheels, list) and wheels
    assert all(str(wheel.get("hash", "")).startswith("sha256:") for wheel in wheels)
