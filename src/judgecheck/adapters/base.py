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
    messages = call.get("request", {}).get("messages", [])
    return "\n".join(str(m.get("content", "")) for m in messages)


def _response_text(call: dict[str, Any]) -> str:
    try:
        return str(call["response"]["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


def load_raw_calls(path: Path | None) -> list[tuple[str, str]]:
    """Read raw_calls.jsonl into (request_text, response_text) pairs."""
    if path is None:
        return []
    calls = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [(_message_text(c), _response_text(c)) for c in calls]


def match_raw_outputs(
    needles: dict[str, list[str]], calls: list[tuple[str, str]]
) -> dict[str, str]:
    """Map example_id to joined raw responses of calls whose request contains one of its needles.

    A call that matches more than one example is ambiguous and is attached to none.
    """
    out: dict[str, list[str]] = {}
    for request, response in calls:
        hits = [k for k, ns in needles.items() if any(n and n in request for n in ns)]
        if len(hits) == 1:
            out.setdefault(hits[0], []).append(response)
    return {k: "\n---\n".join(v) for k, v in out.items()}
