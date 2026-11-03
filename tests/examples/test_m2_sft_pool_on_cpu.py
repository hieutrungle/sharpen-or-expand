"""m2_sft_pool keeps verifiable, non-guessable, English, unique problems with stable ids."""

import importlib.util
from pathlib import Path
from types import ModuleType

import yaml


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m2_sft_pool.py")
    spec = importlib.util.spec_from_file_location("m2_sft_pool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(question: str, answer: str, difficulty: float = 5.0) -> dict[str, object]:
    return {"question": question, "final_answer": answer, "difficulty": difficulty, "topic": "t"}


def test_drop_rules_and_duplicates() -> None:
    m = script()
    rows = [
        row("Compute 1+1.", "2"),
        row("Compute 1+1.", "2"),  # exact duplicate
        row("Is 7 prime?", "Yes"),
        row("Is 9 prime?", "\\text{No}"),
        row("Is the series convergent?", "True"),
        row("计算 1+1", "2"),
        row("Find x.", ""),
        row("Find the limit.", "\\frac{1}{2}", difficulty=6.5),
    ]
    pool = m.build_pool(rows, "deepmath")
    assert [p.question for p in pool.problems] == ["Compute 1+1.", "Find the limit."]
    assert pool.dropped == {
        "duplicate_question": 1,
        "binary_answer": 3,
        "non_english": 1,
        "empty": 1,
    }
    assert pool.rows == 8 and [m_.difficulty for m_ in pool.meta] == [5.0, 6.5]


def test_ids_are_content_hashes_independent_of_order() -> None:
    m = script()
    a = m.build_pool([row("P one.", "1"), row("P two.", "2")], "deepmath")
    b = m.build_pool([row("P two.", "2"), row("P one.", "1")], "deepmath")
    assert {p.problem_id for p in a.problems} == {p.problem_id for p in b.problems}
    assert all(p.problem_id.startswith("deepmath/") for p in a.problems)
    assert all(len(p.problem_id.split("/")[1]) == 16 for p in a.problems)


def test_a_gold_the_verifier_cannot_read_is_dropped() -> None:
    m = script()
    for gold in ("\\frac{1}{2}", "(-\\infty, 3]", "\\sin(x) + 1", "\\mathbb{R}", "e^{\\pi}"):
        assert m.drop_reason("Q?", gold) is None, gold
    assert m.drop_reason("Q?", "\\frac{") == "gold_unreadable"


def test_word_answers_are_dropped() -> None:
    m = script()
    for gold in ("None", "\\text{the empty set}", "Diverges", "\\text{Converges}"):
        assert m.drop_reason("Q?", gold) == "word_answer", gold


def test_config_pins_the_staged_revision() -> None:
    m = script()
    config = m.PoolConfig.model_validate(
        yaml.safe_load(Path("examples/data_preprocess/m2_sft_pool.yaml").read_text())
    )
    assets = yaml.safe_load(Path("scripts/stage_assets.yaml").read_text())["assets"]
    staged = {a["hf_id"]: a["revision"] for a in assets}
    assert staged[config.hf_id] == config.revision
