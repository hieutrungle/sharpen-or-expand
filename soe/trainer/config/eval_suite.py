"""The frozen evaluation suite (decision 004): problem files pinned by sha256, plus the fixed
sampling, prompt and scoring rules every checkpoint is evaluated with.

Evaluation refuses to run if any problem file no longer matches its recorded hash, so the
yardstick cannot drift after M0.
"""

import hashlib
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from soe.utils.dataset.problems import Level

PROBLEM_PLACEHOLDER = "{problem}"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingSpec(FrozenModel):
    """Identical for base, SFT and every RL checkpoint."""

    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=-1)  # -1 disables top-k
    max_tokens: int = Field(gt=0)
    seed: int


class KSpec(FrozenModel):
    """Which k values a tier reports; the largest k is the number of samples per problem."""

    ks: list[int]
    transition_ks: list[int]  # k values at which unsolved->solved / solved->lost are counted

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.ks or self.ks != sorted(set(self.ks)) or self.ks[0] < 1:
            raise ValueError("ks must be strictly increasing positive integers")
        if not set(self.transition_ks) <= set(self.ks):
            raise ValueError("transition_ks must be a subset of ks")
        return self

    @property
    def samples_per_problem(self) -> int:
        return self.ks[-1]


class PromptSpec(FrozenModel):
    """Zero-shot; the same text for every model stage (no few-shot examples for base models)."""

    template: str
    use_chat_template: bool
    # Passed to the chat template. Qwen3.5's generation prompt ends with an empty
    # "<think>\n\n</think>\n\n" block unless enable_thinking is true (then an open "<think>\n").
    # None leaves the template's own default.
    enable_thinking: bool | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.template.count(PROBLEM_PLACEHOLDER) != 1:
            raise ValueError(f"template must contain {PROBLEM_PLACEHOLDER} exactly once")
        return self

    def render(self, problem: str) -> str:
        # Plain replacement, not str.format: the instruction itself contains \\boxed{}.
        return self.template.replace(PROBLEM_PLACEHOLDER, problem)

    def chat_template_kwargs(self) -> dict[str, bool]:
        return {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}


class ScoringSpec(FrozenModel):
    answer_extraction: Literal["last_boxed"]
    truncated_response: Literal["incorrect"]  # finish_reason == "length"
    maj_grouping: Literal["verifier_equivalence", "exact_string"]
    transition_solved_threshold: float = Field(gt=0, le=1)  # solved at k when pass@k >= this


class ProblemFile(FrozenModel):
    name: str
    path: str  # relative to the data root ($SOE_DATA)
    sha256: str = Field(min_length=64, max_length=64)


class ProblemSet(ProblemFile):
    level: Level


class HardSubset(ProblemFile):
    """Problems drawn from several sets and levels (decision 005); each keeps the level of the
    problem set it belongs to, so the file itself has none."""

    selection_report: str  # committed report: rule, counts, selected ids, this file's sha256


class EvalSuite(FrozenModel):
    name: str
    data_build_report: str  # committed report the problem-set hashes come from
    sampling: SamplingSpec
    prompt: PromptSpec
    scoring: ScoringSpec
    main: KSpec  # every problem set
    hard: KSpec  # the hard subset
    problem_sets: list[ProblemSet]
    hard_subset: HardSubset | None
    # problem_id -> reason. Excluded problems stay in the frozen files but are never sampled or
    # scored: gold that is wrong for the stated question, or an answer that is not unique.
    excluded_problems: dict[str, str] = {}

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [p.name for p in self.problem_sets]
        if len(names) != len(set(names)):
            raise ValueError("problem set names must be unique")
        if self.hard.samples_per_problem < self.main.samples_per_problem:
            raise ValueError("the hard subset needs at least as many samples as the main tier")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path) -> EvalSuite:
    return EvalSuite.model_validate(yaml.safe_load(path.read_text()))


def hash_mismatches(suite: EvalSuite, data_root: Path) -> list[str]:
    """Names of problem sets whose file is missing or no longer matches its frozen hash."""
    sets: list[ProblemFile] = [
        *suite.problem_sets,
        *([suite.hard_subset] if suite.hard_subset else []),
    ]
    bad = []
    for problem_set in sets:
        path = data_root / problem_set.path
        if not path.is_file() or file_sha256(path) != problem_set.sha256:
            bad.append(problem_set.name)
    return bad
