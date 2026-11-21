"""Silent-failure detection and the failure-rate gate (D11, D14, D16).

Judge failures come from records the adapters marked status="failed"; their failure_reason
is "<code>: <detail>" (codes in adapters/base.py). Data-quality findings are problems in the
input items themselves (blank answer, empty context) and never count toward the failure rate.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from judgecheck.models import JudgeRecord

DEFAULT_MAX_FAILURE_RATE = 0.10  # D14; becomes a setting in P3


@dataclass(frozen=True)
class Finding:
    code: str
    example_id: str
    detail: str


def detect(records: Iterable[JudgeRecord]) -> list[Finding]:
    """One finding per failed record, in input order. Decided by status, never by text."""
    out = []
    for r in records:
        if r.status != "failed":
            continue
        code, _, detail = (r.failure_reason or "").partition(": ")
        out.append(Finding(code=code, example_id=r.example_id, detail=detail))
    return out


def failure_rate(records: Sequence[JudgeRecord]) -> float:
    """Failed / all records; a count ratio, not 1 - ok share (float drift). 0.0 if empty."""
    if not records:
        return 0.0
    return sum(r.status == "failed" for r in records) / len(records)


def failure_gate(
    records: Sequence[JudgeRecord], max_rate: float = DEFAULT_MAX_FAILURE_RATE
) -> bool:
    """True = gate passes. An empty run fails: no evidence is not a pass (D16)."""
    if not records:
        return False
    return failure_rate(records) <= max_rate


def _blank(text: Any) -> bool:
    return not isinstance(text, str) or not text.strip()


def data_quality(items: Iterable[Mapping[str, Any]]) -> list[Finding]:
    """Problems in the golden items themselves: blank answer, missing or blank context."""
    out = []
    for item in items:
        eid = str(item.get("id", ""))
        if _blank(item.get("answer")):
            out.append(Finding("blank_answer", eid, "answer is empty or whitespace"))
        context = item.get("context")
        chunks = [context] if isinstance(context, str) else list(context or [])
        if not chunks or any(_blank(c) for c in chunks):
            out.append(Finding("empty_context", eid, "context is missing or has a blank chunk"))
    return out
