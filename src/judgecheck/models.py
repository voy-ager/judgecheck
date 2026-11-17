"""Pydantic models for judge records, labels and run metadata."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["pass", "fail"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class JudgeRecord(_Frozen):
    """One judge result. A failed judge call keeps score and verdict as None, never 0."""

    example_id: str
    framework: Literal["deepeval", "ragas"]
    metric: str
    score: float | None = None
    verdict: Verdict | None = None
    threshold: float | None = None
    reason: str | None = None
    raw_output: str | None = None
    status: Literal["ok", "failed"]
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _check_status(self) -> Self:
        if self.status == "failed":
            if self.score is not None or self.verdict is not None:
                raise ValueError("failed record must have score=None and verdict=None")
        elif self.score is None or not 0.0 <= self.score <= 1.0:
            raise ValueError("ok record must have a score in [0, 1]")
        return self


class HumanLabel(_Frozen):
    example_id: str
    label: Verdict
    labeller: str = "maintainer"
    note: str | None = None


class JoinedRecord(_Frozen):
    judge: JudgeRecord
    human: HumanLabel

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        if self.judge.example_id != self.human.example_id:
            raise ValueError("judge and human example_id must match")
        return self


class RunMeta(_Frozen):
    """Everything needed to reproduce a run."""

    judge_model: str
    judge_model_digest: str | None = None
    framework: Literal["deepeval", "ragas"]
    framework_version: str
    prompt_sha256: str | None = None
    temperature: float
    seed: int | None = None
    input_sha256: dict[str, str] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check_utc(self) -> Self:
        if self.created_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("created_at must be UTC")
        return self


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
