"""The one problem record shared by RL pools and every evaluation level."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Level(StrEnum):
    A = "A"  # public benchmarks: MATH-500, AIME 2024/2025, OlympiadBench-EN (decision 002)
    B = "B"  # held-out split carved from the RL pool before any training (decision 003)
    C1 = "C1"  # competitions after the base model's release; never used for any selection
    C2 = "C2"  # 2025 competitions absent from the RL pool; never used for any selection
    TRAIN = "train"


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_id: str
    source: str
    level: Level
    question: str
    answer: str  # gold final answer, compared by soe.utils.reward_score.math_verify
