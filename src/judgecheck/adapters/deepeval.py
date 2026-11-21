"""DeepEval results adapter: native `test_results` JSON to JudgeRecord."""

import json
import re
from pathlib import Path
from typing import Any

from judgecheck.adapters.base import (
    FRAMEWORK_ERROR_FIELD,
    MISSING_REASON,
    MISSING_SUCCESS,
    NULL_SCORE,
    SKIPPED,
    ZERO_CLAIMS_DEFAULT,
    load_raw_calls,
    match_raw_outputs,
    require_id,
    score_or_failure,
    with_code,
)
from judgecheck.models import JudgeRecord

EMPTY_LIST_LOG = re.compile(r"(?m)^(Claims|Verdicts):\n\[\][ 	]*$")


def _failure(md: dict[str, Any]) -> tuple[float | None, str | None]:
    if md.get("error"):
        return None, with_code(FRAMEWORK_ERROR_FIELD, str(md["error"]))
    if "score" not in md:
        return None, MISSING_REASON
    if md["score"] is None:
        return None, with_code(NULL_SCORE, "score is null")
    score, reason = score_or_failure(md["score"])
    if reason is None and not isinstance(md.get("success"), bool):
        return None, with_code(MISSING_SUCCESS, "no boolean success field")
    logs = md.get("verboseLogs") or ""
    if reason is None and EMPTY_LIST_LOG.search(logs):
        return None, with_code(ZERO_CLAIMS_DEFAULT, "no claims extracted, framework default score")
    return score, reason


def _record(example_id: str, md: dict[str, Any]) -> JudgeRecord:
    score, reason = _failure(md)
    failed = reason is not None
    return JudgeRecord(
        example_id=example_id,
        framework="deepeval",
        metric=str(md.get("name", "faithfulness")).lower(),
        score=score,
        verdict=None if failed else ("pass" if md["success"] else "fail"),
        threshold=md.get("threshold"),
        reason=md.get("reason"),
        status="failed" if failed else "ok",
        failure_reason=reason,
    )


def _skipped(example_id: str, metric: str) -> JudgeRecord:
    return JudgeRecord(
        example_id=example_id,
        framework="deepeval",
        metric=metric.lower(),
        status="failed",
        failure_reason=with_code(SKIPPED, "test case has no metric result"),
    )


def read_deepeval(
    path: Path,
    raw_calls: Path | None = None,
    allow_index_ids: bool = False,
    metric: str = "faithfulness",
) -> list[JudgeRecord]:
    """Parse a DeepEval JSON export. example_id is the test case `name`.

    A test case without a name is an error unless `allow_index_ids` (then its index is used).

    Only `metric` entries are read, so a multi-metric export gives one record per id.
    Test cases with no `metrics_data` (skipped rows) yield a failed record (D11).
    """
    data = json.loads(path.read_text("utf-8"))
    records: list[JudgeRecord] = []
    needles: dict[str, list[str]] = {}
    for i, tr in enumerate(data["test_results"]):
        eid = require_id(tr.get("name"), path.name, allow_index_ids, tr.get("index", i))
        needles[eid] = [tr.get("actual_output") or "", *(tr.get("retrieval_context") or [])]
        mds = tr.get("metrics_data") or []
        if not mds:  # D11: a skipped test case is a failed record, not a missing one
            records.append(_skipped(eid, metric))
        records.extend(
            _record(eid, md)
            for md in mds
            if str(md.get("name", "faithfulness")).lower() == metric.lower()
        )
    raw = match_raw_outputs(needles, load_raw_calls(raw_calls))
    return [r.model_copy(update={"raw_output": raw.get(r.example_id)}) for r in records]
