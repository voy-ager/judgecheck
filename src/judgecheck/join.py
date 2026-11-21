"""Join judge records and human labels by example ID. Nothing is dropped silently."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from judgecheck.models import HumanLabel, JoinedRecord, JudgeRecord


@dataclass(frozen=True)
class LabelLoad:
    labels: list[HumanLabel]
    unlabelled: list[str]


@dataclass(frozen=True)
class JoinResult:
    joined: list[JoinedRecord]  # ok judge records with a label: the agreement set
    failed: list[JoinedRecord]  # failed judge records with a label: reported, never scored
    missing_labels: list[str]
    missing_judgements: list[str]


def _reject_duplicates(ids: list[str], what: str) -> None:
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate example_id in {what}: {dupes}")


def load_labels(path: Path) -> LabelLoad:
    """Read `example_id,label,note`. A blank label means unlabelled: excluded and listed."""
    labels: list[HumanLabel] = []
    unlabelled: list[str] = []
    seen: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            eid = (row.get("example_id") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if eid:
                seen.append(eid)
            if not label:
                if eid:  # a fully blank row is skipped, not recorded as id ""
                    unlabelled.append(eid)
                continue
            data = {"example_id": eid, "label": label, "note": row.get("note") or None}
            try:
                labels.append(HumanLabel.model_validate(data))
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {n}: invalid label row") from exc
    _reject_duplicates(seen, "labels")
    return LabelLoad(labels, unlabelled)


def join(records: list[JudgeRecord], labels: list[HumanLabel]) -> JoinResult:
    """Pair records with labels; list every id present on only one side."""
    _reject_duplicates([r.example_id for r in records], "judge records")
    _reject_duplicates([lab.example_id for lab in labels], "labels")
    by_id = {lab.example_id: lab for lab in labels}
    judged = {r.example_id for r in records}
    pairs = [
        JoinedRecord(judge=r, human=by_id[r.example_id]) for r in records if r.example_id in by_id
    ]
    return JoinResult(
        joined=[p for p in pairs if p.judge.status == "ok"],
        failed=[p for p in pairs if p.judge.status == "failed"],
        missing_labels=[r.example_id for r in records if r.example_id not in by_id],
        missing_judgements=[lab.example_id for lab in labels if lab.example_id not in judged],
    )
