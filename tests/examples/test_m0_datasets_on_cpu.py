"""The v2 text repair (decision 007): DAPO-derived encoding debris, nothing else touched."""

import importlib.util
from pathlib import Path
from types import ModuleType


def script() -> ModuleType:
    path = Path("examples/data_preprocess/m0_repair_v2.py")
    spec = importlib.util.spec_from_file_location("m0_repair_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_form_feed_before_rac_was_a_frac() -> None:
    fix = script().repair_text
    assert fix("in the form \x0crac{m}{n}, find m+n") == "in the form \\frac{m}{n}, find m+n"


def test_stray_control_characters_are_removed() -> None:
    fix = script().repair_text
    assert fix("the other fi\x0cve; the \x0cfirst") == "the other five; the first"
    assert fix("m^2=\x0c\\frac{p}{q}") == "m^2=\\frac{p}{q}"
    assert fix("$x \\le\x14 2y$ \x88 $a \x01 \\cdot b$") == "$x \\le 2y$  $a  \\cdot b$"


def test_tabs_newlines_and_clean_text_are_untouched() -> None:
    fix = script().repair_text
    for text in ["a\tb\nc", "plain $\\frac{1}{2}$", "ünïcode ≤ ✓"]:
        assert fix(text) == text
