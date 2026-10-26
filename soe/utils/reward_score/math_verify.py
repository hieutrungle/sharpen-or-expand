"""Final-answer verification for math problems.

A response is correct only if its last \\boxed{...} is equivalent to the gold answer. Equivalence
is delegated to math-verify (symbolic comparison via sympy). The exact accept/reject behaviour on
edge cases is audited by hand in M0 and pinned in
tests/utils/reward_score/test_math_verify_on_cpu.py.
"""

from dataclasses import dataclass

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


def answers_equivalent(predicted: str, gold: str) -> bool:
    # Both sides are wrapped in \\boxed{} so math-verify parses them with the same rules.
    gold_parsed = parse(f"\\boxed{{{gold}}}")
    predicted_parsed = parse(f"\\boxed{{{predicted}}}")
    return bool(verify(gold_parsed, predicted_parsed))


def score_response(response: str, gold: str) -> Verdict:
    extracted = extract_boxed(response)
    if not extracted:
        return Verdict(extracted=None, correct=False)
    return Verdict(extracted=extracted, correct=answers_equivalent(extracted, gold))
