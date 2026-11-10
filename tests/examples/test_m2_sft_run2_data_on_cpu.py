"""examples/data_preprocess/m2_sft_run2_data.py: the nearest-rank cut and the committed config."""

import importlib.util
from pathlib import Path
from types import ModuleType

import yaml


def load() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_run2_data.py")
    spec = importlib.util.spec_from_file_location("m2_sft_run2_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step = load()


def test_nearest_rank_cut() -> None:
    lengths = list(range(1, 11))  # 1..10: rank ceil(0.9 * 10) = 9
    assert step.cut_length(lengths, 90) == 9
    assert step.cut_length([5, 1, 3], 90) == 5  # rank ceil(2.7) = 3
    assert step.cut_length(list(range(1, 101)), 90) == 90


def test_config_cuts_run_1s_train_file_at_the_90th_percentile() -> None:
    raw = yaml.safe_load(Path("examples/data_preprocess/m2_sft_run2_data.yaml").read_text())
    config = step.Run2DataConfig.model_validate(raw)
    run1 = yaml.safe_load(Path("examples/sft/m2_run1_qwen35-4b.yaml").read_text())
    assert (config.input, config.input_sha256) == (
        run1["data"]["train_file"],
        run1["data"]["train_sha256"],
    )
    assert config.percentile == 90 and config.output != config.input
