"""Pins the verifier's current accept/reject behaviour. The M0 hand audit (~200 items) reviews
these cases; any change to an expected value needs a decision note."""

import pytest

from soe.utils.reward_score.math_verify import answers_equivalent, extract_boxed, score_response


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"so the answer is \boxed{42}.", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"first \boxed{1}, final \boxed{2}", "2"),
        (r"\boxed{ \{1, 2\} }", r"\{1, 2\}"),
        (r"\fbox{7}", "7"),
        ("no box here: 42", None),
        (r"\boxed{unbalanced", None),
        (r"\boxed{}", ""),
    ],
)
def test_extract_boxed(text: str, expected: str | None) -> None:
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("0.5", r"\frac{1}{2}"),
        (r"\dfrac12", "1/2"),
        (r"\sqrt{8}", r"2\sqrt{2}"),
        ("(1,2]", "(1,2]"),
        ("3", "x=3"),
        ("B", r"\text{(B)}"),
    ],
)
def test_equivalent_forms_accepted(predicted: str, gold: str) -> None:
    assert answers_equivalent(predicted, gold)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("4", "3"),
        ("10", r"10\%"),  # percent vs bare number: currently rejected, flagged for the M0 audit
        (r"\frac{1}{3}", "0.33"),
    ],
)
def test_non_equivalent_rejected(predicted: str, gold: str) -> None:
    assert not answers_equivalent(predicted, gold)


def test_score_response_requires_a_box() -> None:
    assert score_response("The answer is 42", "42").correct is False
    assert score_response(r"The answer is \boxed{}", "42").extracted is None
    verdict = score_response(r"... \boxed{42}", "42")
    assert verdict.correct and verdict.extracted == "42"
