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


class SamplingRunInfo(BaseModel):
    """Provenance of one sampling job, written next to its SampleRecord JSONL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checkpoint: str
    revision: str | None
    suite_name: str
    problem_sets: list[str]
    samples_per_problem: int = Field(ge=1)
    seed: int  # the vLLM per-request seed actually used (may differ from the suite's, e.g. M0.6)
    chat_template_kwargs: dict[str, bool]
    stop_token_ids: list[int]
    git_commit: str
    hostname: str
    slurm_job_id: str | None
    gpus: list[str]
    library_versions: dict[str, str | None]
