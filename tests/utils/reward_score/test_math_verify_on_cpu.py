"""Pins the verifier's accept/reject behaviour. Cases tagged V### (M0.8 hand audit) or G## (gold
screen) are real base-model answers whose verdict was settled by hand; any change to an expected
value needs a decision note."""

import logging

import pytest

from soe.utils.reward_score import math_verify as mv
from soe.utils.reward_score.math_verify import (
    all_boxed,
    answers_equivalent,
    extract_boxed,
    normalize,
    score_response,
)


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
        (r"3\text{ cm}", "3"),
    ],
)
def test_equivalent_forms_accepted(predicted: str, gold: str) -> None:
    assert answers_equivalent(predicted, gold)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("4", "3"),
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


GOLD_2501 = (
    r"x=\frac{1}{6} \pi+\pi k, \frac{1}{3} \pi+\pi k, \frac{2}{3} \pi+\pi k, \frac{5}{6} \pi+\pi k"
)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        # percent sign on either side (V107, V130, G09, G10), or as a fraction (V053)
        (r"60\%", "60"),
        (r"10\%", "10"),
        (r"8\%", r"8 \%"),
        (r"17.5\%", "17.5%"),
        (r"\frac{5}{8}", "62.5%"),
        # column vector written as a tuple (V113, V118, V119, V138, V152)
        (
            r"\left(\frac{16}{49},\frac{48}{49},\frac{24}{49}\right)",
            r"\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}",
        ),
        (
            r"\left(-\frac13,\;\frac23,\;\frac53\right)",
            r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}",
        ),
        (r"(-18,\,-49,\,96)", r"\begin{pmatrix} -18 \\ -49 \\ 96 \end{pmatrix}"),
        (r"(-7,\ 16,\ 5)", r"\begin{pmatrix} -7 \\ 16 \\ 5 \end{pmatrix}"),
        (r"\left(\frac{1}{5}, -\frac{18}{5}\right)", r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}"),
        # labelled answers (V108, V112, V046, V055)
        (r"w=16,\;d=3,\;a=1,\;b=1", "16,3,1,1"),
        (r"a = -2,\ b = -2", "-2,-2"),
        (r"\text{Minimum }1,\ \text{Maximum }5", "5,1"),
        (r"\text{Rayna: } 7,\ \text{Qing: } 14,\ \text{Paolo: } 18", "7,14,18"),
        # words and ordinals around the numbers (V122, V146, V123, V127)
        (r"7\text{ goats and }4\text{ helicopters}", "7,4"),
        (r"7 \text{ stuffed goats and } 4 \text{ toy helicopters}", "7,4"),
        (r"12^{\text{th}} grade", "12"),
        (r"12^{\mathrm{th}} \text{ grade}", "12"),
        # one-variable inequality for an interval (V136, G07)
        (r"-1\le k\le -\frac{1}{2}", r"[-1,-\frac{1}{2}]"),
        ("k < -5", r"(-\infty,-5)"),
        (r"x \geq 2", r"[2,\infty)"),
        # sets of terms listed flat (V012); families of solutions with another parameter (V031)
        (r"\{10,11,12,13,14\},\{-2,-1,0,1,2\}", "10,11,12,13,14,-2,-1,0,1,2"),
        (
            r"x = \dfrac{\pi}{6} + n\pi,\ \dfrac{\pi}{3} + n\pi,\ \dfrac{2\pi}{3} + n\pi,\ "
            r"\dfrac{5\pi}{6} + n\pi \quad \text{for } n \in \mathbb{Z}",
            GOLD_2501,
        ),
    ],
)
def test_audited_notations_accepted(predicted: str, gold: str) -> None:
    assert answers_equivalent(predicted, gold)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        # the rewrites must keep wrong answers wrong
        (r"7\text{ goats and }5\text{ helicopters}", "7,4"),
        (r"(-7,\ 16,\ 6)", r"\begin{pmatrix} -7 \\ 16 \\ 5 \end{pmatrix}"),
        (r"a = -2,\ b = 3", "-2,-2"),
        ("k < -5", r"(-\infty,-5]"),
        (r"k \le -5", r"(-\infty,-5)"),
        ("x > 3", r"(-\infty,3)"),
        (r"13^{\text{th}} grade", "12"),
        (r"60\%", "0.6%"),
        ("0.6", "60"),
        (r"\sin x", r"\cos x"),
        # V039: the same four families written as two with period pi/2; no general rule, kept
        # rejected
        (
            r"x = \frac{\pi}{6} + \frac{n\pi}{2} \text{ or } x = \frac{\pi}{3} + \frac{n\pi}{2}",
            GOLD_2501,
        ),
    ],
)
def test_audited_rejections_hold(predicted: str, gold: str) -> None:
    assert not answers_equivalent(predicted, gold)


def test_bare_lists_compare_as_sets() -> None:
    # Known limit (module docstring): order and repeats are ignored in bare comma lists.
    assert answers_equivalent("1,5", "5,1") and answers_equivalent("2,2", "2")
    assert not answers_equivalent("(1,5)", "(5,1)")


def test_normalize_keeps_function_names_and_matrices() -> None:
    assert normalize(r"\sin x + cos x", percent="strip") == r"\sin x + cos x"
    matrix = r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
    assert normalize(matrix, percent="strip") == matrix


def test_all_boxed_handles_nesting_and_order() -> None:
    text = r"so \boxed{\frac{1}{2}} or rather \fbox{3} and finally \boxed{x^{2}}"
    assert all_boxed(text) == [r"\frac{1}{2}", "3", "x^{2}"]


def test_timed_out_comparisons_are_recorded_with_their_pair() -> None:
    before = len(mv.timed_out_comparisons())
    mv._timeouts.pair = ("x", "y")
    logging.getLogger("math_verify.grader").warning("Timeout during comparison")
    mv._timeouts.pair = None
    logging.getLogger("math_verify.grader").warning("Timeout during comparison")  # no pair: ignored
    assert mv.timed_out_comparisons()[before:] == [("x", "y")]
