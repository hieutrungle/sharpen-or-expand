"""Final-answer verification for math problems.

A response is correct only if its last \\boxed{...} is equivalent to the gold answer. Equivalence
is delegated to math-verify (symbolic comparison via sympy). Where the M0.8 audit and gold screen
found it rejecting a correct answer written in another notation, both answers are rewritten to
one notation and compared again (`normalize`); a pair is accepted if either comparison accepts
it, so the rewrites can only add acceptances. The accept/reject behaviour on edge cases is
pinned in tests/utils/reward_score/test_math_verify_on_cpu.py.

Known limit kept on purpose: math-verify compares a bare comma list as a set, so order and
repeats are ignored ("1,5" equals "5,1", "2" equals "2,2").
"""

import re
from dataclasses import dataclass
from typing import Literal

from math_verify import parse, verify

_BOX_COMMANDS = ("\\boxed", "\\fbox")


@dataclass(frozen=True, slots=True, kw_only=True)
class Verdict:
    extracted: str | None
    correct: bool


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} (or \\fbox{...}), with nested braces."""
    start = max(text.rfind(command + "{") for command in _BOX_COMMANDS)
    if start == -1:
        return None
    open_brace = text.index("{", start)
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i].strip()
    return None  # unbalanced: treat as no answer


def _verify(predicted: str, gold: str) -> bool:
    # Both sides are wrapped in \\boxed{} so math-verify parses them with the same rules.
    gold_parsed = parse(f"\\boxed{{{gold}}}")
    predicted_parsed = parse(f"\\boxed{{{predicted}}}")
    return bool(verify(gold_parsed, predicted_parsed))


_SPACING = re.compile(r"(?<!\\)\\[,;:! ]|\\q?quad\b|\\displaystyle|\$")
_COLUMN_VECTOR = re.compile(r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}", re.S)
_ORDINAL = re.compile(
    r"\^\{?(?:\\(?:text|mathrm)\{)?(?:st|nd|rd|th)\}?\}?|(?<=\d)(?:st|nd|rd|th)\b"
)
_TEXT = re.compile(r"\\(?:text|textbf|textit|mathrm|mbox)\{([^{}]*)\}")
# Bare English words; command arguments ("{pmatrix}") and function names without a backslash
# are kept.
_WORD = re.compile(r"(?<![\\A-Za-z{])[A-Za-z]{3,}(?![A-Za-z])|(?<![\\A-Za-z{])or(?![A-Za-z])")
_FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "exp", "max", "min", "gcd", "lcm"}
_SET_BRACES = re.compile(r"\\[{}]")
_LABEL = re.compile(
    r"^(?:[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?|\\[A-Za-z]+)\s*(?:=|:|\\in(?![A-Za-z]))\s*"
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*\\?%")
_VAR = r"[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?"
_LESS = r"(<|\\lt(?![A-Za-z])|\\leqslant|\\leq|\\le(?![A-Za-z]))"
_GREATER = r"(>|\\gt(?![A-Za-z])|\\geqslant|\\geq|\\ge(?![A-Za-z]))"
_BETWEEN = re.compile(rf"^(.+?)\s*{_LESS}\s*{_VAR}\s*{_LESS}\s*(.+)$")
_BELOW = re.compile(rf"^{_VAR}\s*{_LESS}\s*(.+)$")
_ABOVE = re.compile(rf"^{_VAR}\s*{_GREATER}\s*(.+)$")
# "n \\in \\mathbb{Z}" after a family of solutions states the parameter's domain, not an answer.
_DOMAIN = re.compile(rf"^{_VAR}\s*\\in\s*\\mathbb\{{[A-Z]\}}$")
_PARAMETER = re.compile(r"(?<![A-Za-z\\])[nm](?![A-Za-z])")


def _column_vector_as_tuple(match: re.Match[str]) -> str:
    body = match.group(1)
    if "&" in body:  # a matrix, not a column vector
        return match.group(0)
    return "(" + ",".join(entry.strip() for entry in body.split("\\\\")) + ")"


def _text_as_separator(match: re.Match[str]) -> str:
    # Words between answers ("7 goats and 4 helicopters") separate them; text holding a number
    # keeps it.
    content = match.group(1)
    return content if any(c.isdigit() for c in content) else ","


def _drop_word(match: re.Match[str]) -> str:
    return match.group(0) if match.group(0) in _FUNCTIONS else ","


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return [*parts, text[start:]]


def _as_interval(element: str) -> str:
    """A one-variable inequality as an interval: "-1 \\le k \\le 2" -> "[-1,2]"."""
    compact = element.strip()
    if m := _BETWEEN.match(compact):
        low, op_low, op_high, high = m.groups()
        left = "(" if op_low in ("<", "\\lt") else "["
        right = ")" if op_high in ("<", "\\lt") else "]"
        return f"{left}{low},{high}{right}"
    if m := _BELOW.match(compact):
        op, bound = m.groups()
        return f"(-\\infty,{bound}" + (")" if op in ("<", "\\lt") else "]")
    if m := _ABOVE.match(compact):
        op, bound = m.groups()
        return ("(" if op in (">", "\\gt") else "[") + f"{bound},\\infty)"
    return element


def normalize(answer: str, *, percent: Literal["strip", "fraction"]) -> str:
    """One notation for answers the audit found written several ways: column vectors as tuples,
    percent signs dropped (or as /100), ordinals, words and labels removed, set braces flattened,
    one-variable inequalities as intervals."""
    text = _COLUMN_VECTOR.sub(_column_vector_as_tuple, answer)
    text = _SPACING.sub(" ", text)
    text = _PERCENT.sub(r"\1" if percent == "strip" else r"(\1/100)", text)
    text = _ORDINAL.sub("", text)
    text = _TEXT.sub(_text_as_separator, text)
    text = _WORD.sub(_drop_word, text)
    text = _SET_BRACES.sub("", text)
    parts = [part.strip() for part in _split_top_level(text)]
    elements = [_LABEL.sub("", part) for part in parts if not _DOMAIN.match(part)]
    return ",".join(_as_interval(e) for e in elements if e)


def answers_equivalent(predicted: str, gold: str) -> bool:
    if _verify(predicted, gold):
        return True
    for percent in ("strip", "fraction"):
        p, g = normalize(predicted, percent=percent), normalize(gold, percent=percent)
        if "\\pi" in p and "\\pi" in g:  # periodic families: the integer parameter's name
            p, g = _PARAMETER.sub("k", p), _PARAMETER.sub("k", g)
        if p and g and (p, g) != (predicted, gold) and _verify(p, g):
            return True
    return False


def score_response(response: str, gold: str) -> Verdict:
    extracted = extract_boxed(response)
    if not extracted:
        return Verdict(extracted=None, correct=False)
    return Verdict(extracted=extracted, correct=answers_equivalent(extracted, gold))
