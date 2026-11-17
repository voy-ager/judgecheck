import hashlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from judgecheck.models import HumanLabel, JoinedRecord, JudgeRecord, RunMeta, sha256_file


def ok(**kw: Any) -> JudgeRecord:
    base: dict[str, Any] = {
        "example_id": "e1",
        "framework": "ragas",
        "metric": "faithfulness",
        "score": 0.8,
        "verdict": "pass",
        "status": "ok",
    }
    return JudgeRecord(**{**base, **kw})


def failed(**kw: Any) -> JudgeRecord:
    base: dict[str, Any] = {
        "example_id": "e1",
        "framework": "deepeval",
        "metric": "faithfulness",
        "status": "failed",
        "failure_reason": "parse error",
    }
    return JudgeRecord(**{**base, **kw})


def meta(**kw: Any) -> RunMeta:
    base: dict[str, Any] = {
        "judge_model": "llama3.2",
        "framework": "ragas",
        "framework_version": "0.2",
        "temperature": 0.0,
    }
    return RunMeta(**{**base, **kw})


def test_valid_ok_and_boundaries() -> None:
    assert ok(score=0.0).score == 0.0
    assert ok(score=1.0).score == 1.0


def test_valid_failed_keeps_none() -> None:
    rec = failed()
    assert rec.score is None and rec.verdict is None


@pytest.mark.parametrize("score", [None, -0.1, 1.1])
def test_ok_requires_score_in_range(score: float | None) -> None:
    with pytest.raises(ValidationError):
        ok(score=score)


def test_failed_rejects_score_or_verdict() -> None:
    with pytest.raises(ValidationError):
        failed(score=0.0)
    with pytest.raises(ValidationError):
        failed(verdict="fail")


def test_frozen_and_extra_forbidden() -> None:
    rec = ok()
    with pytest.raises(ValidationError):
        rec.score = 0.1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ok(bogus=1)


def test_bad_literals() -> None:
    with pytest.raises(ValidationError):
        ok(framework="other")
    with pytest.raises(ValidationError):
        HumanLabel(example_id="e1", label="maybe")  # type: ignore[arg-type]


def test_human_label_defaults() -> None:
    lab = HumanLabel(example_id="e1", label="pass")
    assert lab.labeller == "maintainer" and lab.note is None


def test_joined_matching_and_mismatch() -> None:
    lab = HumanLabel(example_id="e1", label="fail")
    assert JoinedRecord(judge=ok(), human=lab).human is lab
    with pytest.raises(ValidationError):
        JoinedRecord(judge=ok(example_id="e2"), human=lab)


def test_runmeta_defaults_utc_now() -> None:
    m = meta()
    assert m.created_at.utcoffset() == timedelta(0)
    assert m.input_sha256 == {} and m.seed is None


def test_runmeta_rejects_naive_and_non_utc() -> None:
    with pytest.raises(ValidationError):
        meta(created_at=datetime(2026, 1, 1))  # noqa: DTZ001
    with pytest.raises(ValidationError):
        meta(created_at=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2))))


def test_round_trip_json() -> None:
    m = meta(seed=1, input_sha256={"a.json": "ab"}, created_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert RunMeta.model_validate_json(m.model_dump_json()) == m
    j = JoinedRecord(judge=failed(raw_output="x"), human=HumanLabel(example_id="e1", label="pass"))
    assert JoinedRecord.model_validate_json(j.model_dump_json()) == j


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    data = b"hello" * 50000
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert sha256_file(str(p)) == sha256_file(p)


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_failed_requires_failure_reason(reason: str | None) -> None:
    with pytest.raises(ValidationError):
        failed(failure_reason=reason)


def test_ok_rejects_failure_reason() -> None:
    with pytest.raises(ValidationError):
        ok(failure_reason="boom")


@pytest.mark.parametrize("eid", ["", "   "])
def test_example_id_non_empty(eid: str) -> None:
    with pytest.raises(ValidationError):
        ok(example_id=eid)
    with pytest.raises(ValidationError):
        HumanLabel(example_id=eid, label="pass")


def test_example_id_stripped() -> None:
    assert ok(example_id=" e1 ").example_id == "e1"
