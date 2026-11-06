"""scripts/sft_diagnostic.py on synthetic evaluations (M2.11 diagnostic)."""

import csv
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from soe.utils.dataset.records import SampleRecord

LOOP = "the same words again " * 400  # compresses far below 15%


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sft_diagnostic", Path("scripts/sft_diagnostic.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = load()


def make_eval(
    directory: Path,
    n: int,
    correct: Callable[[int, int], bool],
    truncated: Callable[[int, int], bool],
    problems: int = 20,
) -> Path:
    directory.mkdir(parents=True)
    plan = {"suite_name": f"suite-{n}", "checkpoint": directory.name, "num_shards": 1}
    (directory / "plan.json").write_text(json.dumps(plan))
    lines = []
    for p in range(problems):
        for s in range(n):
            cut = truncated(p, s)
            record = SampleRecord(
                problem_id=f"q{p:02d}",
                sample_index=s,
                checkpoint=directory.name,
                completion=LOOP if cut else "short reasoning</think>\\boxed{1}",
                extracted_answer=None if cut else "1",
                correct=correct(p, s) and not cut,
                completion_tokens=16384 if cut else 50 + p,
                finish_reason="length" if cut else "stop",
            )
            lines.append(record.model_dump_json())
    (directory / "samples-0-of-1.jsonl").write_text("\n".join(lines) + "\n")
    return directory


def test_summary_decomposition_and_difficulty(tmp_path: Path) -> None:
    # base: stops 3 of 4, correct when stopped on even problems; SFT: stops 2 of 4, same accuracy
    base = make_eval(tmp_path / "b", 4, lambda p, s: p % 2 == 0, lambda p, s: s == 3)
    sft = make_eval(tmp_path / "s", 4, lambda p, s: p % 2 == 0, lambda p, s: s >= 2)
    _, _, b = diag.read_eval(base)
    _, _, s = diag.read_eval(sft)
    difficulty = {f"q{p:02d}": (4.0 if p < 10 else 8.0) for p in range(20)}
    out = diag.compare(b, s, difficulty)
    assert out["base"]["stop_rate"] == 0.75 and out["sft"]["stop_rate"] == 0.5
    assert out["base"]["accuracy_when_stopped"] == 0.5 == out["sft"]["accuracy_when_stopped"]
    assert out["base"]["pass@1"] == pytest.approx(0.375) and out["sft"]["pass@1"] == 0.25
    change = out["pass@1_change"]
    assert change["stopping"] + change["accuracy"] == pytest.approx(change["total"])
    assert change["accuracy"] == 0.0 and change["stopping"] == pytest.approx(-0.125)
    assert out["base"]["loops_among_truncated"] == 1.0 and out["sft"]["loop_rate"] == 0.5
    assert out["paired"]["truncated"]["sft_minus_base"] == pytest.approx(0.25)
    assert out["paired"]["truncated"]["se"] == 0.0  # the same change on every problem
    assert set(out["by_difficulty"]) == {"<=5", ">=7"}
    assert out["by_difficulty"]["<=5"]["problems"] == 10
    assert len(out["rows"]) == 20 and out["rows"][0]["n"] == 4


def test_cli_writes_both_splits_to_one_table(tmp_path: Path) -> None:
    pairs = []
    for split, n in (("rl_val", 16), ("sft_val", 4)):
        b = make_eval(tmp_path / split / "b", n, lambda p, s: s < 2, lambda p, s: False)
        s = make_eval(tmp_path / split / "s", n, lambda p, s: s < 1, lambda p, s: False)
        pairs.append(f"{split}:{b}:{s}")
    report, table = tmp_path / "r.json", tmp_path / "t.csv"
    diag.SftDiagnosticCli(pair=pairs, report=report, table=table).cli_cmd()
    out = json.loads(report.read_text())
    assert out["rl_val"]["base"]["pass@16"] == 1.0 and out["sft_val"]["sft"]["pass@4"] == 1.0
    rows = list(csv.DictReader(table.open()))
    assert len(rows) == 40 and {r["n"] for r in rows} == {"16", "4"}


def test_an_uneven_evaluation_is_refused(tmp_path: Path) -> None:
    d = make_eval(tmp_path / "b", 4, lambda p, s: True, lambda p, s: False)
    with (d / "samples-0-of-1.jsonl").open("a") as f:
        extra = SampleRecord(
            problem_id="q00",
            sample_index=4,
            checkpoint="b",
            completion="x",
            extracted_answer=None,
            correct=False,
            completion_tokens=1,
            finish_reason="stop",
        )
        f.write(extra.model_dump_json() + "\n")
    with pytest.raises(SystemExit, match="different sample counts"):
        diag.read_eval(d)
