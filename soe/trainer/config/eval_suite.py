"""The frozen evaluation suite: problem files pinned by sha256, plus fixed sampling parameters.

Evaluation refuses to run if any problem file no longer matches its recorded hash, so the
yardstick cannot drift after M0.
"""

import hashlib
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from soe.utils.dataset.problems import Level


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingSpec(FrozenModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_tokens: int = Field(gt=0)
    ks: list[int]

    @model_validator(mode="after")
    def _check_ks(self) -> Self:
        if not self.ks or self.ks != sorted(set(self.ks)) or self.ks[0] < 1:
            raise ValueError("ks must be strictly increasing positive integers")
        return self

    @property
    def samples_per_problem(self) -> int:
        return self.ks[-1]


class ProblemSet(FrozenModel):
    name: str
    level: Level
    path: str  # relative to the data root ($SOE_DATA)
    sha256: str = Field(min_length=64, max_length=64)


class EvalSuite(FrozenModel):
    name: str
    sampling: SamplingSpec
    problem_sets: list[ProblemSet]
    hard_subset: ProblemSet | None
    hard_subset_sampling: SamplingSpec | None
    transition_solved_threshold: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _check_hard_subset(self) -> Self:
        if (self.hard_subset is None) != (self.hard_subset_sampling is None):
            raise ValueError("hard_subset and hard_subset_sampling go together")
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
    sets = [*suite.problem_sets, *([suite.hard_subset] if suite.hard_subset else [])]
    bad = []
    for problem_set in sets:
        path = data_root / problem_set.path
        if not path.is_file() or file_sha256(path) != problem_set.sha256:
            bad.append(problem_set.name)
    return bad
