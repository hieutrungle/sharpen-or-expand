"""Every committed experiment config must load, so a broken YAML fails in CI, not on FASTER."""

from pathlib import Path

from soe.trainer.config.experiment import load_config


def test_all_example_configs_validate() -> None:
    paths = sorted(Path("examples").glob("*_trainer/*.yaml")) + sorted(
        Path("examples").glob("sft/*.yaml")
    )
    assert paths
    names = [load_config(path).name for path in paths]
    assert len(names) == len(set(names)), "config names must be unique (they name run directories)"
