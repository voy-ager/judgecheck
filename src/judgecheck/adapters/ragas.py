"""Ragas results adapter: native CSV or JSON rows to JudgeRecord."""

import ast
import csv
import json
from pathlib import Path
from typing import Any

from judgecheck.adapters.base import (
    MISSING_REASON,
    load_raw_calls,
    match_raw_outputs,
    require_id,
    score_or_failure,
)
from judgecheck.models import JudgeRecord, Verdict


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return list(json.loads(path.read_text("utf-8")))
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _contexts(row: dict[str, Any]) -> list[str] | None:
    """Context strings, or None when the field parses to something that is not a list."""
    ctx = row.get("retrieved_contexts")
    if isinstance(ctx, str):
        try:
            ctx = ast.literal_eval(ctx)
        except (ValueError, SyntaxError):
            ctx = [ctx]
    if ctx is not None and not isinstance(ctx, list):
        return None
    return [str(c) for c in ctx or []]


def _record(eid: str, row: dict[str, Any], metric: str, threshold: float | None) -> JudgeRecord:
    score, reason = (None, MISSING_REASON) if metric not in row else score_or_failure(row[metric])
    verdict: Verdict | None = None
    if score is not None and threshold is not None:
        verdict = "pass" if score >= threshold else "fail"
    return JudgeRecord(
        example_id=eid,
        framework="ragas",
        metric=metric,
        score=score,
        verdict=verdict,
        threshold=threshold,
        status="ok" if reason is None else "failed",
        failure_reason=reason,
    )


def read_ragas(
    path: Path,
    raw_calls: Path | None = None,
    metric: str = "faithfulness",
    threshold: float | None = None,
    allow_index_ids: bool = False,
) -> list[JudgeRecord]:
    """Parse a Ragas CSV/JSON export. example_id is the `example_id` column.

    A row without one is an error unless `allow_index_ids` (then the row index is used).

    Ragas has no pass/fail threshold; verdict is set only when `threshold` is given.
    """
    rows = _load_rows(path)
    ids = [
        require_id(r.get("example_id"), path.name, allow_index_ids, i) for i, r in enumerate(rows)
    ]
    records = [_record(e, r, metric, threshold) for e, r in zip(ids, rows, strict=True)]
    needles: dict[str, list[str]] = {}
    for n, (e, r) in enumerate(zip(ids, rows, strict=True)):
        ctx = _contexts(r)
        if ctx is None:
            records[n] = JudgeRecord(
                example_id=e,
                framework="ragas",
                metric=metric,
                status="failed",
                failure_reason="retrieved_contexts is not a list",
            )
            ctx = []
        needles[e] = [str(r.get("response") or ""), *ctx]
    raw = match_raw_outputs(needles, load_raw_calls(raw_calls))
    return [r.model_copy(update={"raw_output": raw.get(r.example_id)}) for r in records]
