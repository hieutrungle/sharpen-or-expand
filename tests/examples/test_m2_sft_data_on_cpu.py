"""m2_sft_data: contiguous sampling runs, kept trajectories, problem-level split, length drops."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from soe.trainer.config.eval_suite import load_suite
from soe.utils.dataset.problems import Level, Problem
from soe.utils.dataset.records import SampleRecord


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_data.py")
    spec = importlib.util.spec_from_file_location("m2_sft_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(pid: str, index: int, completion: str, correct: bool = True) -> SampleRecord:
    return SampleRecord(
        problem_id=pid,
        checkpoint="m",
        sample_index=index,
        completion=completion,
        completion_tokens=len(completion),
        finish_reason="stop",
        extracted_answer="1",
        correct=correct,
    )


def test_runs_must_be_one_contiguous_permutation(tmp_path: Path) -> None:
    m = script()
    pilot = yaml.safe_load(Path("examples/data_preprocess/m2_sft_pilot.yaml").read_text())
    full = yaml.safe_load(Path("examples/data_preprocess/m2_sft_full.yaml").read_text())
    paths = []
    for name, cfg in [("a", pilot), ("b", full)]:
        paths.append(tmp_path / f"{name}.yaml")
        paths[-1].write_text(yaml.safe_dump(cfg))
    assert [r.draw_start for r in m.load_runs(paths)] == [0, 1000]
    paths[1].write_text(yaml.safe_dump(full | {"draw_start": 1200}))
    with pytest.raises(SystemExit, match="contiguous"):
        m.load_runs(paths)
    paths[1].write_text(yaml.safe_dump(full | {"draw_seed": 1}))
    with pytest.raises(SystemExit, match="draw_seed"):
        m.load_runs(paths)


def test_examples_split_by_problem_and_long_ones_are_dropped() -> None:
    m = script()
    prompt = load_suite(Path("soe/trainer/config/eval_suite.yaml")).prompt
    problems = {
        f"d/{i}": Problem(
            problem_id=f"d/{i}", source="d", level=Level.TRAIN, question=f"q{i}", answer="1"
        )
        for i in range(4)
    }
    samples = {
        "d/0": [record("d/0", 0, "a</think>x"), record("d/0", 1, "b</think>x")],
        "d/1": [record("d/1", 0, "c</think>x", correct=False)],
        "d/2": [record("d/2", 0, "long</think>x")],
        "d/3": [record("d/3", 0, "e</think>x")],
    }
    kept = m.kept_by_problem(samples, 2)
    assert [len(kept[p]) for p in sorted(kept)] == [2, 0, 1, 1]
    rows = m.examples(kept, {p: problems[p] for p in ("d/0", "d/2", "d/3")}, prompt)
    assert [r.problem_id for r in rows] == ["d/0", "d/0", "d/2", "d/3"]
    assert rows[0].user == prompt.render("q0")

    def measure(example: object) -> tuple[int, int]:
        n = len(example.completion)  # type: ignore[attr-defined]
        return 100 + n, n

    splits, sizes, too_long = m.split_and_measure(rows, {"d/0"}, measure, max_length=112)
    assert too_long == 1  # "long</think>x" is 113 tokens here
    assert [r.problem_id for r in splits["val"]] == ["d/0", "d/0"]
    assert [r.problem_id for r in splits["train"]] == ["d/3"]
    assert sizes["train"] == [(110, 10)]


def test_val_split_is_seeded_and_problem_level() -> None:
    m = script()
    ids = [f"d/{i}" for i in range(100)]
    assert m.val_ids(ids, 10, 3) == m.val_ids(list(reversed(ids)), 10, 3)
    assert len(m.val_ids(ids, 10, 3)) == 10


def test_committed_config_builds_from_the_pilot_and_full_runs() -> None:
    m = script()
    config = m.SftDataConfig.model_validate(
        yaml.safe_load(Path("examples/data_preprocess/m2_sft_data.yaml").read_text())
    )
    runs = m.load_runs(config.samples)
    assert (runs[0].draw_start, runs[-1].draw_stop) == (0, 16000)
    assert (config.val_problems, config.max_length) == (500, 18432)
