"""Tests for judgecheck.checks.failures (maintainer implements).

Accepted API (D16; pure functions over lists of JudgeRecord / golden items):
    detect(records) -> list[Finding]        # one Finding per status="failed" record
    failure_rate(records) -> float          # failed / all records; 0.0 for an empty list
    failure_gate(records, max_rate=0.10) -> bool   # True = gate passes (D14: rate <= max_rate)
    data_quality(items) -> list[Finding]    # input problems, not judge failures
Finding has attributes `code` (str), `example_id` (str), `detail` (str).
`code` is the part of failure_reason before ": " (D11/D15 reason codes in adapters/base.py).

Decision (g005): a blank answer is a data-quality finding on the items, not a judge failure.
The judge did run and returned a score (DeepEval 0.33 on g005), so the record is status=ok;
flagging it in `detect` would hide that the golden item itself is bad.
Accepted by the maintainer (D16).
"""

import json
from pathlib import Path
from typing import Literal

import pytest

from judgecheck.adapters import base
from judgecheck.checks import failures as fl
from judgecheck.models import JudgeRecord

_NAMES = ("detect", "failure_rate", "failure_gate", "data_quality")
pytestmark = pytest.mark.xfail(
    not all(hasattr(fl, n) for n in _NAMES),
    strict=False,
    reason="awaiting maintainer implementation, failures.py",
)


def failed(
    eid: str, code: str, detail: str, framework: Literal["deepeval", "ragas"] = "ragas"
) -> JudgeRecord:
    return JudgeRecord(
        example_id=eid,
        framework=framework,
        metric="faithfulness",
        status="failed",
        failure_reason=base.with_code(code, detail),
    )


def ok(eid: str, score: float = 1.0) -> JudgeRecord:
    return JudgeRecord(
        example_id=eid,
        framework="ragas",
        metric="faithfulness",
        score=score,
        verdict="pass" if score == 1.0 else "fail",
        status="ok",
    )


# One fixture per failure type: (code, detail). Details quote real spike output where possible.
FIXTURES = {
    base.FRAMEWORK_ERROR: "NaN score",
    base.SCHEMA_ECHO: "judge echoed the JSON schema",  # Ragas llama3.2, all 30 items
    base.NO_STATEMENTS: "no statements to check",
    base.ZERO_CLAIMS_DEFAULT: "no claims extracted, framework default score",  # DeepEval 7b g028
    base.MISSING_SUCCESS: "no boolean success field",
    base.SKIPPED: "test case has no metric result",
    base.NULL_SCORE: "score is null",
    base.NON_NUMERIC_SCORE: "'high'",
    base.OUT_OF_RANGE_SCORE: "1.5",
    base.BAD_CONTEXTS: "retrieved_contexts is not a list",
    base.FRAMEWORK_ERROR_FIELD: "Connection refused",
}


@pytest.mark.parametrize(("code", "detail"), FIXTURES.items())
def test_each_failure_type_is_detected_with_its_code(code: str, detail: str) -> None:
    (f,) = fl.detect([ok("g001"), failed("g028", code, detail), ok("g002")])
    assert (f.code, f.example_id) == (code, "g028")
    assert detail in f.detail


def test_ok_records_give_no_findings() -> None:
    assert fl.detect([ok("a"), ok("b", 0.5)]) == []


def test_every_failed_record_is_reported_once_in_order() -> None:
    recs = [failed(f"g{i:03d}", base.SCHEMA_ECHO, "x") for i in (3, 1, 2)]
    assert [f.example_id for f in fl.detect(recs)] == ["g003", "g001", "g002"]


def test_code_is_prefix_only_detail_may_contain_colons() -> None:
    (f,) = fl.detect([failed("a", base.BAD_CONTEXTS, "expected list: got str")])
    assert f.code == base.BAD_CONTEXTS


def test_answer_text_quoting_a_code_does_not_misfire() -> None:
    # An ok record whose judge reason quotes a code must not be reported.
    rec = ok("a").model_copy(update={"reason": "schema_echo: the answer said Claims: []"})
    assert fl.detect([rec]) == []


def test_ragas_llama_all_schema_echo_has_rate_one() -> None:
    recs = [
        failed(f"g{i:03d}", base.SCHEMA_ECHO, "judge echoed the JSON schema") for i in range(30)
    ]
    assert len(fl.detect(recs)) == 30 and fl.failure_rate(recs) == 1.0
    assert not fl.failure_gate(recs)


def test_failure_rate_counts_all_failed_codes_over_all_records() -> None:
    recs = [ok("a"), ok("b"), failed("c", base.SKIPPED, "x"), failed("d", base.NULL_SCORE, "x")]
    assert fl.failure_rate(recs) == 0.5


def test_failure_rate_empty_is_zero() -> None:
    assert fl.failure_rate([]) == 0.0


def _run(n_failed: int, n: int = 30) -> list[JudgeRecord]:
    bad = [failed(f"f{i}", base.ZERO_CLAIMS_DEFAULT, "x") for i in range(n_failed)]
    return bad + [ok(f"o{i}") for i in range(n - n_failed)]


def test_d14_three_of_thirty_is_ten_percent_and_passes() -> None:
    assert fl.failure_rate(_run(3)) == pytest.approx(0.10)
    assert fl.failure_gate(_run(3)) is True  # exactly 10% passes: gate fails only when > 10%


def test_d14_four_of_thirty_fails() -> None:
    assert fl.failure_gate(_run(4)) is False


def test_d14_zero_failures_passes() -> None:
    assert fl.failure_gate(_run(0)) is True


def test_d14_threshold_is_a_parameter() -> None:
    assert fl.failure_gate(_run(4), max_rate=0.20) is True
    assert fl.failure_gate(_run(3), max_rate=0.05) is False


def test_d14_empty_run_is_not_a_pass() -> None:
    # No results at all is a broken run, not a healthy one; gate must not pass vacuously.
    assert fl.failure_gate([]) is False


# Data quality (items), not judge failures.
def item(eid: str, answer: str = "Yes.", context: list[str] | None = None) -> dict[str, object]:
    ctx = ["The Atlas room seats 12."] if context is None else context
    return {"id": eid, "question": "q?", "context": ctx, "answer": answer}


def test_blank_answer_g005_is_data_quality_not_judge_failure() -> None:
    items = [item("g001"), item("g005", answer="   \n ")]  # g005 as in data/golden/v0
    (f,) = fl.data_quality(items)
    assert (f.code, f.example_id) == ("blank_answer", "g005")
    # the judge's own score for g005 (status ok) is not a failure
    assert fl.detect([ok("g005", 0.33)]) == []


def test_empty_context_is_data_quality() -> None:
    (f,) = fl.data_quality([item("g1", context=[])])
    assert (f.code, f.example_id) == ("empty_context", "g1")


def test_whitespace_only_context_chunk_is_empty_context() -> None:
    (f,) = fl.data_quality([item("g2", context=["  \n"])])
    assert f.code == "empty_context"


def test_clean_items_have_no_data_quality_findings() -> None:
    assert (
        fl.data_quality([item("a"), item("b", answer="I'm not sure, please check the docs.")]) == []
    )


def test_missing_score_reason_is_detected() -> None:
    # MISSING_REASON is a full reason ("missing_score: no score field"), emitted by both adapters.
    rec = JudgeRecord(
        example_id="g1",
        framework="deepeval",
        metric="faithfulness",
        status="failed",
        failure_reason=base.MISSING_REASON,
    )
    (f,) = fl.detect([ok("g2"), rec])
    assert (f.code, f.example_id) == ("missing_score", "g1")
    assert "no score field" in f.detail


def test_data_quality_on_real_golden_v0_finds_g005_and_g027() -> None:
    # g005 has a blank answer; g027 has an empty context list (answer supported by nothing).
    path = Path(__file__).parents[1] / "data" / "golden" / "v0" / "items.json"
    items = json.loads(path.read_text("utf-8"))
    items = items if isinstance(items, list) else items["items"]
    assert len(items) == 30
    found = [(f.code, f.example_id) for f in fl.data_quality(items)]
    assert found == [("blank_answer", "g005"), ("empty_context", "g027")]


def test_detail_quoting_another_code_keeps_its_own_code() -> None:
    (f,) = fl.detect([failed("a", base.SKIPPED, "like schema_echo: not really")])
    assert f.code == base.SKIPPED


def test_blank_answer_item_leaves_failure_rate_unchanged() -> None:
    recs = _run(3)
    before = fl.failure_rate(recs)
    assert [f.code for f in fl.data_quality([item("g005", answer="   ")])] == ["blank_answer"]
    assert fl.failure_rate(recs) == before and fl.failure_gate(recs) is True


def test_d14_boundary_one_of_ten_passes_two_of_ten_fails() -> None:
    assert fl.failure_gate(_run(1, 10)) is True
    assert fl.failure_gate(_run(2, 10)) is False
