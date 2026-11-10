"""Frozen starts (decision 031): digest, launch check and scripts/freeze_checkpoint.py."""

import importlib.util
import json
import stat
from pathlib import Path
from types import ModuleType

import pytest

from soe.trainer.config.frozen_start import (
    FrozenStart,
    check_frozen_start,
    directory_digest,
    file_hashes,
)


def snapshot(tmp_path: Path) -> Path:
    """A Hub-like snapshot: files are symlinks into a blobs directory; one is a dotfile."""
    blobs, snap = tmp_path / "blobs", tmp_path / "snapshots" / "abc"
    blobs.mkdir()
    snap.mkdir(parents=True)
    for name, text in {
        "config.json": "{}",
        ".gitattributes": "*",
        "model.safetensors": "w",
    }.items():
        (blobs / f"b-{name}").write_text(text)
        (snap / name).symlink_to(blobs / f"b-{name}")
    return snap


def start_for(directory: Path) -> FrozenStart:
    hashes = file_hashes(directory)
    return FrozenStart(
        hf_id="org/model",
        revision="0" * 40,
        files=len(hashes),
        digest=directory_digest(hashes),
        report="docs/reports/x.json",
    )


def test_hashes_follow_symlinks_and_include_dotfiles(tmp_path: Path) -> None:
    hashes = file_hashes(snapshot(tmp_path))
    assert sorted(hashes) == [".gitattributes", "config.json", "model.safetensors"]
    assert directory_digest(hashes) == directory_digest(dict(reversed(list(hashes.items()))))


@pytest.mark.parametrize("change", ["edit", "extra", "missing"])
def test_launch_check_refuses_any_change(tmp_path: Path, change: str) -> None:
    snap = snapshot(tmp_path)
    start = start_for(snap)
    check_frozen_start(start, snap)  # unchanged: passes
    if change == "edit":
        (snap / "config.json").resolve().write_text('{"x": 1}')
    elif change == "extra":
        (snap / "generation_config.json").write_text("{}")
    else:
        (snap / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="does not match the frozen start"):
        check_frozen_start(start, snap)


def test_repo_frozen_start_file_validates() -> None:
    from soe.trainer.config.frozen_start import load_frozen_starts

    for start in load_frozen_starts():
        assert Path(start.report).is_file()
        report = json.loads(Path(start.report).read_text())
        assert (report["digest"], report["file_count"]) == (start.digest, start.files)
        hashes = {name: f["sha256"] for name, f in report["files"].items()}
        assert directory_digest(hashes) == start.digest


def writable(path: Path) -> bool:
    """Write bits of the target (mode bits, not os.access, which root always passes)."""
    return bool(path.resolve().stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def freeze_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fc", Path("scripts/freeze_checkpoint.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_copies_real_files_and_makes_them_read_only(tmp_path: Path) -> None:
    fc = freeze_script()
    snap, copy, report = snapshot(tmp_path), tmp_path / "frozen" / "hf", tmp_path / "r.json"
    fc.FreezeCheckpointCli(source=snap, copy_to=copy, read_only=True, report=report).cli_cmd()
    written = json.loads(report.read_text())
    assert written["digest"] == start_for(snap).digest and written["file_count"] == 3
    assert not (copy / "config.json").is_symlink()
    for path in [copy, *copy.rglob("*")]:
        assert not writable(path)
    assert writable(snap / "config.json")  # the source is untouched by a copy
    with pytest.raises(SystemExit, match="written once"):
        fc.FreezeCheckpointCli(source=snap, copy_to=copy, report=report).cli_cmd()
    for path in [copy, *copy.rglob("*")]:
        path.chmod(0o755)  # let pytest clean up tmp_path


def test_freeze_in_place_makes_the_blobs_read_only(tmp_path: Path) -> None:
    fc = freeze_script()
    snap, report = snapshot(tmp_path), tmp_path / "r.json"
    fc.FreezeCheckpointCli(source=snap, read_only=True, report=report).cli_cmd()
    assert not writable(snap / "config.json")
    assert (snap / "config.json").is_symlink()
    for blob in (tmp_path / "blobs").iterdir():
        blob.chmod(0o644)
