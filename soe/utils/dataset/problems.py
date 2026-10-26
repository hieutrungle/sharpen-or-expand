"""The one problem record shared by RL pools and every evaluation level."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Level(StrEnum):
    A = "A"  # public benchmarks: MATH-500, AIME 2024/2025, an olympiad set
    B = "B"  # held-out split carved from the RL pool before any training
    C = "C"  # fresh / OOD; never used for any selection
    TRAIN = "train"


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_id: str
    source: str
    level: Level
    question: str
    answer: str  # gold final answer, compared by soe.utils.reward_score.math_verify
