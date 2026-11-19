from pathlib import Path

import pytest

from judgecheck.join import join, load_labels
from judgecheck.models import HumanLabel, JudgeRecord


def rec(eid: str) -> JudgeRecord:
    return JudgeRecord(
        example_id=eid,
        framework="ragas",
        metric="faithfulness",
        score=1.0,
        verdict="pass",
        status="ok",
    )


def lab(eid: str) -> HumanLabel:
    return HumanLabel(example_id=eid, label="pass")


def test_join_reports_both_sides() -> None:
    res = join([rec("a"), rec("b")], [lab("b"), lab("c")])
    assert [j.judge.example_id for j in res.joined] == ["b"]
    assert res.missing_labels == ["a"] and res.missing_judgements == ["c"]


def test_join_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="judge records"):
        join([rec("a"), rec("a")], [])
    with pytest.raises(ValueError, match="labels"):
        join([], [lab("a"), lab("a")])


def test_load_labels_blank_label_is_unlabelled(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text("example_id,label,note\na,pass,ok\nb,,todo\nc, FAIL ,\n", "utf-8")
    out = load_labels(p)
    assert [(x.example_id, x.label, x.note) for x in out.labels] == [
        ("a", "pass", "ok"),
        ("c", "fail", None),
    ]
    assert out.unlabelled == ["b"]


def test_load_labels_invalid_and_duplicate(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text("example_id,label\na,maybe\n", "utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_labels(p)
    p.write_text("example_id,label\na,pass\na,fail\n", "utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_labels(p)
