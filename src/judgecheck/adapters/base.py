"""Helpers shared by the DeepEval and Ragas adapters."""

import json
import math
from pathlib import Path
from typing import Any

NAN_REASON = "NaN score (framework error or no statements)"
MISSING_REASON = "missing score field"


def score_or_failure(raw: object) -> tuple[float | None, str | None]:
    """Return (score, None) for a usable score, else (None, failure_reason). Never 0."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "nan"}):
        return None, NAN_REASON
    if isinstance(raw, bool):  # float(True) == 1.0 would pass a bad export as a real score
        return None, f"non-numeric score: {raw!r}"
    try:
        score = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, f"non-numeric score: {raw!r}"
    if math.isnan(score):
        return None, NAN_REASON
    if not 0.0 <= score <= 1.0:
        return None, f"out-of-range score: {score}"
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
