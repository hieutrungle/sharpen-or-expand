"""The audit's risk tags: they decide which verifier decisions a human looks at."""

import importlib.util
import random
from pathlib import Path
from types import ModuleType


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verifier_audit", Path("scripts/verifier_audit.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_boxed_handles_nesting_and_order() -> None:
    m = script()
    text = r"so \boxed{\frac{1}{2}} or rather \fbox{3} and finally \boxed{x^{2}}"
    assert m.all_boxed(text) == [r"\frac{1}{2}", "3", "x^{2}"]


def test_tags() -> None:
    m = script()
    tags = m.categories
    assert "false_positive_risk" in tags(r"\frac12", "0.5", True, ["0.5"])
    assert "false_positive_risk" not in tags(r"\dfrac{1}{2}", r"\frac{1}{2}", True, [])
    assert "false_negative_risk" in tags("(2,4)", "(4,2)", False, [])  # reordered tuple
    assert "false_negative_risk" in tags(r"30^\circ", "30", False, [])  # same digits
    assert "false_negative_risk" not in tags("7", "12", False, [])
    assert "multi_answer" in tags("69$,$84", "69,84", True, [])
    assert "units_text" in tags(r"5\text{ cm}", "5", True, [])
    assert "extraction" in tags("3", "3", True, ["2", "3"])
    assert tags("7", "12", False, [])[-1] == "control_incorrect"


def test_a_problem_is_drawn_once_even_with_many_answers() -> None:
    m = script()
    pool = [{"problem_id": "a", "extracted": str(i)} for i in range(5)]
    pool += [{"problem_id": "b", "extracted": "0"}, {"problem_id": "c", "extracted": "0"}]
    for seed in range(20):
        drawn = m.draw_problems(random.Random(seed), pool, {"c"}, 3)
        assert sorted(d["problem_id"] for d in drawn) == ["a", "b"]  # c used; a only once
