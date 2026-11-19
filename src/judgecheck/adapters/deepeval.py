"""DeepEval results adapter: native `test_results` JSON to JudgeRecord."""

import json
from pathlib import Path
from typing import Any

from judgecheck.adapters.base import (
    MISSING_REASON,
    load_raw_calls,
    match_raw_outputs,
    score_or_failure,
)
from judgecheck.models import JudgeRecord


def _failure(md: dict[str, Any]) -> tuple[float | None, str | None]:
    if md.get("error"):
        return None, str(md["error"])
    if "score" not in md:
        return None, MISSING_REASON
    if md["score"] is None:
        return None, "score is null"
    return score_or_failure(md["score"])


def _record(example_id: str, md: dict[str, Any]) -> JudgeRecord:
    score, reason = _failure(md)
    failed = reason is not None
    return JudgeRecord(
        example_id=example_id,
        framework="deepeval",
        metric=str(md.get("name", "faithfulness")).lower(),
        score=score,
        verdict=None if failed else ("pass" if md.get("success") else "fail"),
        threshold=md.get("threshold"),
        reason=md.get("reason"),
        status="failed" if failed else "ok",
        failure_reason=reason,
    )


def read_deepeval(path: Path, raw_calls: Path | None = None) -> list[JudgeRecord]:
    """Parse a DeepEval JSON export. example_id is the test case `name` (else its index).

    Test cases with no `metrics_data` (skipped rows) yield no record; `join` reports them.
    """
    data = json.loads(path.read_text("utf-8"))
    records: list[JudgeRecord] = []
    needles: dict[str, list[str]] = {}
    for i, tr in enumerate(data["test_results"]):
        eid = str(tr.get("name") or tr.get("index", i))
        needles[eid] = [tr.get("actual_output") or "", *(tr.get("retrieval_context") or [])]
        records.extend(_record(eid, md) for md in tr.get("metrics_data") or [])
    raw = match_raw_outputs(needles, load_raw_calls(raw_calls))
    return [r.model_copy(update={"raw_output": raw.get(r.example_id)}) for r in records]
