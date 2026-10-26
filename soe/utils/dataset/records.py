"""One JSONL line per sampled completion; evaluation and transition tables are built from these."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    problem_id: str
    sample_index: int = Field(ge=0)
    checkpoint: str  # run_id/step, or a model id for base checkpoints
    completion: str
    extracted_answer: str | None
    correct: bool
    completion_tokens: int = Field(ge=0)
    finish_reason: Literal["stop", "length"]
