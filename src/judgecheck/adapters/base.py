"""Helpers shared by the DeepEval and Ragas adapters."""

import json
import math
from pathlib import Path
from typing import Any

# D11: fixed reason codes for status="failed"; a reason is "<code>: <detail>".
ZERO_CLAIMS_DEFAULT = "zero_claims_default"  # no claims extracted, framework default score
FRAMEWORK_ERROR = "framework_error"  # NaN score, cause not visible
NO_STATEMENTS = "no_statements"  # Ragas found nothing to check
SCHEMA_ECHO = "schema_echo"  # judge echoed the JSON schema instead of answering
MISSING_SUCCESS = "missing_success"
SKIPPED = "skipped"  # test case has no metric result
MISSING_REASON = "missing_score: no score field"
NULL_SCORE = "null_score"
NON_NUMERIC_SCORE = "non_numeric_score"
OUT_OF_RANGE_SCORE = "out_of_range_score"
BAD_CONTEXTS = "bad_contexts"
FRAMEWORK_ERROR_FIELD = "deepeval_error"  # the framework's own error text, kept as detail
NAN_REASON = f"{FRAMEWORK_ERROR}: NaN score"


def with_code(code: str, detail: str) -> str:
    return f"{code}: {detail}"


def classify_nan(raw: str | None) -> str:
    """Refine a NaN failure reason from the raw judge output, when it is attached."""
    raw = raw or ""
    if any('"properties"' in part[:40] for part in raw.split("\n---\n")):
        return with_code(SCHEMA_ECHO, "judge echoed the JSON schema")
    try:  # the first call is the statement extraction; later calls are not about it
        if json.loads(raw.split("\n---\n")[0]).get("statements") == []:
            return with_code(NO_STATEMENTS, "no statements to check")
    except (ValueError, AttributeError):
        pass
    return NAN_REASON


def score_or_failure(raw: object) -> tuple[float | None, str | None]:
    """Return (score, None) for a usable score, else (None, failure_reason). Never 0."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "nan"}):
        return None, NAN_REASON
    if isinstance(raw, bool):  # float(True) == 1.0 would pass a bad export as a real score
        return None, with_code(NON_NUMERIC_SCORE, repr(raw))
    try:
        score = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, with_code(NON_NUMERIC_SCORE, repr(raw))
    if math.isnan(score):
        return None, NAN_REASON
    if not 0.0 <= score <= 1.0:
        return None, with_code(OUT_OF_RANGE_SCORE, str(score))
    return score, None


def _message_text(call: dict[str, Any]) -> str:
    messages = (call.get("request") or {}).get("messages", [])
    return "\n".join(str(m.get("content", "")) for m in messages)


def _response_text(call: dict[str, Any]) -> str:
    resp = call.get("response")
    try:
        if isinstance(resp, dict) and "message" in resp:  # native Ollama /api/chat
            return str(resp["message"]["content"])
        return str(call["response"]["choices"][0]["message"]["content"])  # OpenAI-compatible
    except (KeyError, IndexError, TypeError):
        return ""


RawCall = tuple[str, str, str | None]


def load_raw_calls(path: Path | None) -> list[RawCall]:
    """Read raw_calls.jsonl into (request_text, response_text, example_id or None)."""
    if path is None:
        return []
    calls = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [(_message_text(c), _response_text(c), c.get("example_id")) for c in calls]


def match_raw_outputs(needles: dict[str, list[str]], calls: list[RawCall]) -> dict[str, str]:
    """Map example_id to joined raw responses.

    A call's own `example_id` field wins. Only calls without one fall back to text search:
    the request must contain one of an example's needles, and a call matching several
    examples is ambiguous and attached to none.
    """
    out: dict[str, list[str]] = {}
    for request, response, call_id in calls:
        if call_id is not None:
            hits = [str(call_id)] if str(call_id) in needles else []
        else:
            hits = [k for k, ns in needles.items() if any(n and n in request for n in ns)]
        if len(hits) == 1:
            out.setdefault(hits[0], []).append(response)
    return {k: "\n---\n".join(v) for k, v in out.items()}


def require_id(value: object, where: str, allow_index: bool, index: int) -> str:
    """Return the row's id; an absent id is an error unless index ids were opted into."""
    text = str(value).strip() if value is not None else ""
    if text:
        return text
    if allow_index:
        return str(index)
    raise ValueError(
        f"{where}: row {index} has no example id (use allow_index_ids=True for row indexes)"
    )
