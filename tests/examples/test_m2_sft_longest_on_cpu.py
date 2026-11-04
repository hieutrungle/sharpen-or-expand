"""examples/data_preprocess/m2_sft_longest.py: selection and the committed config (M2.8)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import yaml


def load() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_longest.py")
    spec = importlib.util.spec_from_file_location("m2_sft_longest", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


longest = load()


def test_longest_first_and_ties_in_file_order() -> None:
    assert longest.longest([5, 9, 7, 9, 1], 3) == [1, 3, 2]
    assert longest.longest([2, 1], 5) == [0, 1]


def test_config_reads_the_m2_5_train_file() -> None:
    raw = yaml.safe_load(Path("examples/data_preprocess/m2_sft_longest.yaml").read_text())
    config = longest.LongestConfig.model_validate(raw)
    probe = yaml.safe_load(Path("examples/sft/m2_probe_qwen35-4b.yaml").read_text())
    assert (config.input, config.input_sha256) == (
        probe["data"]["train_file"],
        probe["data"]["train_sha256"],
    )
    assert config.count == 256
