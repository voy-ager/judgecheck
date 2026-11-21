import json
from pathlib import Path
from typing import Any

import pytest

from judgecheck.adapters.base import load_raw_calls, score_or_failure
from judgecheck.adapters.deepeval import read_deepeval
from judgecheck.adapters.ragas import read_ragas

FX = Path(__file__).parent / "fixtures" / "pilot"
NAN = "framework_error: NaN score"


def test_pilot_deepeval_with_raw() -> None:
    recs = read_deepeval(FX / "deepeval.json", FX / "deepeval_raw_calls.jsonl")
    assert [r.example_id for r in recs] == ["test_case_0", "test_case_1", "test_case_2"]
    assert [r.verdict for r in recs] == ["fail", "pass", "pass"]
    assert recs[0].score == 0.0 and recs[0].status == "ok"
    assert recs[0].metric == "faithfulness" and recs[0].threshold == 0.5
    assert recs[0].raw_output and "claims" in recs[0].raw_output


def test_pilot_deepeval_without_raw() -> None:
    assert all(r.raw_output is None for r in read_deepeval(FX / "deepeval.json"))


@pytest.mark.parametrize("name", ["ragas.csv", "ragas.json"])
def test_pilot_ragas(name: str) -> None:
    recs = read_ragas(FX / name, FX / "ragas_raw_calls.jsonl", threshold=0.5, allow_index_ids=True)
    assert [r.example_id for r in recs] == ["0", "1", "2"]
    assert all(r.score == 1.0 and r.verdict == "pass" for r in recs)
    assert all(r.raw_output for r in recs)


def test_ragas_no_threshold_means_no_verdict() -> None:
    assert all(r.verdict is None for r in read_ragas(FX / "ragas.csv", allow_index_ids=True))


def write(tmp_path: Path, name: str, obj: Any) -> Path:
    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj), "utf-8")
    return p


def test_ragas_nan_csv_and_json(tmp_path: Path) -> None:
    csv_p = write(tmp_path, "r.csv", "example_id,response,faithfulness\na,x,nan\nb,y,\nc,z,0.5\n")
    recs = read_ragas(csv_p, threshold=0.6)
    assert [r.status for r in recs] == ["failed", "failed", "ok"]
    assert recs[0].failure_reason == NAN and recs[0].score is None and recs[0].verdict is None
    assert recs[2].verdict == "fail" and recs[2].example_id == "c"
    json_p = write(
        tmp_path, "r.json", '[{"response": "x", "faithfulness": NaN}, {"faithfulness": null}]'
    )
    assert [r.failure_reason for r in read_ragas(json_p, allow_index_ids=True)] == [NAN, NAN]


def test_ragas_missing_field_and_bad_values(tmp_path: Path) -> None:
    p = write(tmp_path, "r.json", [{"response": "x"}, {"faithfulness": "abc"}, {"faithfulness": 7}])
    recs = read_ragas(p, allow_index_ids=True)
    assert [r.status for r in recs] == ["failed"] * 3
    assert recs[0].failure_reason == "missing score field"
    assert "non-numeric" in (recs[1].failure_reason or "")
    assert "out-of-range" in (recs[2].failure_reason or "")


def test_ragas_context_string_unparseable(tmp_path: Path) -> None:
    p = write(tmp_path, "r.csv", "response,retrieved_contexts,faithfulness\nx,not a list,1.0\n")
    assert read_ragas(p, allow_index_ids=True)[0].status == "ok"


def de_row(md: dict[str, Any], name: str = "t") -> dict[str, Any]:
    return {"name": name, "actual_output": "a", "retrieval_context": ["c"], "metrics_data": [md]}


def test_deepeval_error_row_is_failed_not_fail(tmp_path: Path) -> None:
    md = {"name": "Faithfulness", "success": False, "score": None, "error": "ValidationError: bad"}
    rec = read_deepeval(write(tmp_path, "d.json", {"test_results": [de_row(md)]}))[0]
    assert rec.status == "failed" and rec.failure_reason == "ValidationError: bad"
    assert rec.verdict is None and rec.score is None


def test_deepeval_error_with_score_still_failed(tmp_path: Path) -> None:
    md = {"name": "Faithfulness", "success": False, "score": 0.0, "error": "boom"}
    rec = read_deepeval(write(tmp_path, "d.json", {"test_results": [de_row(md)]}))[0]
    assert rec.status == "failed" and rec.score is None


def test_deepeval_missing_and_null_score(tmp_path: Path) -> None:
    rows = [
        de_row({"name": "Faithfulness", "success": True}, "a"),
        de_row({"name": "Faithfulness", "success": True, "score": None}, "b"),
        de_row({"name": "Faithfulness", "success": True, "score": 2.0}, "c"),
    ]
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": rows}))
    assert [r.failure_reason for r in recs][:2] == ["missing score field", "score is null"]
    assert recs[2].status == "failed" and "out-of-range" in (recs[2].failure_reason or "")


def test_deepeval_missing_or_null_success_is_failed(tmp_path: Path) -> None:
    rows = [
        de_row({"name": "Faithfulness", "score": 1.0}, "a"),
        de_row({"name": "Faithfulness", "score": 1.0, "success": None}, "b"),
    ]
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": rows}))
    assert all(r.status == "failed" and r.verdict is None for r in recs)
    assert all("success" in (r.failure_reason or "") for r in recs)


def test_deepeval_skipped_row_and_index_id(tmp_path: Path) -> None:
    rows = [
        {"index": 4, "metrics_data": [{"score": 1.0, "success": True}]},
        {"name": "skipped", "metrics_data": []},
    ]
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": rows}), allow_index_ids=True)
    assert [r.example_id for r in recs] == ["4", "skipped"] and recs[0].metric == "faithfulness"


def test_ambiguous_raw_call_attached_to_none(tmp_path: Path) -> None:
    rows = [
        de_row({"name": "Faithfulness", "score": 1.0, "success": True}, "a"),
        de_row({"name": "Faithfulness", "score": 1.0, "success": True}, "b"),
    ]
    call = {
        "request": {"messages": [{"content": "has a and c"}]},
        "response": {"choices": [{"message": {"content": "R"}}]},
    }
    raw = write(
        tmp_path, "raw.jsonl", json.dumps(call) + "\n\n" + json.dumps({"request": {}}) + "\n"
    )
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": rows}), raw)
    assert all(r.raw_output is None for r in recs)


def test_base_helpers(tmp_path: Path) -> None:
    assert load_raw_calls(None) == []
    assert score_or_failure(0.25) == (0.25, None)
    assert score_or_failure(float("nan"))[1] == NAN
    assert score_or_failure(True) == (None, "non-numeric score: True")


def test_missing_id_is_error_by_default(tmp_path: Path) -> None:
    rows = [{"metrics_data": [{"score": 1.0, "success": True}]}]
    with pytest.raises(ValueError, match=r"d\.json: row 0 has no example id"):
        read_deepeval(write(tmp_path, "d.json", {"test_results": rows}))
    with pytest.raises(ValueError, match=r"r\.csv: row 1"):
        read_ragas(write(tmp_path, "r.csv", "example_id,faithfulness\na,1.0\n,1.0\n"))


def test_raw_call_example_id_field_wins(tmp_path: Path) -> None:
    def call(eid: str | None, text: str, resp: str) -> str:
        c: dict[str, Any] = {
            "request": {"messages": [{"content": text}]},
            "response": {"choices": [{"message": {"content": resp}}]},
        }
        if eid is not None:
            c["example_id"] = eid
        return json.dumps(c)

    md = {"name": "Faithfulness", "score": 1.0, "success": True}
    rows = [de_row(md, "a"), de_row(md, "b")]
    rows[0]["actual_output"], rows[1]["actual_output"] = "ans-a", "ans-b"
    lines = [
        call("b", "mentions a only", "R-b"),
        call("zzz", "x", "R-unknown"),
        call(None, "text has ans-a", "R-a"),
    ]
    raw = write(tmp_path, "raw.jsonl", "\n".join(lines))
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": rows}), raw)
    assert recs[0].raw_output == "R-a" and recs[1].raw_output == "R-b"


def test_raw_response_shapes_null_request_and_int_ids(tmp_path: Path) -> None:
    calls = [
        {"example_id": 1, "request": None, "response": {"message": {"content": "native"}}},
        {"example_id": 1, "response": {"choices": [{"message": {"content": "openai"}}]}},
        {"example_id": 1, "response": "not json body"},
    ]
    raw = write(tmp_path, "raw.jsonl", "\n".join(json.dumps(c) for c in calls))
    md = {"name": "Faithfulness", "score": 1.0, "success": True}
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": [de_row(md, "1")]}), raw)
    assert recs[0].raw_output == "native\n---\nopenai\n---\n"


def test_deepeval_metric_filter(tmp_path: Path) -> None:
    other = {"name": "Relevancy", "score": 0.1, "success": False}
    good = {"name": "Faithfulness", "score": 1.0, "success": True}
    row = de_row(good, "a")
    row["metrics_data"].append(other)
    p = write(tmp_path, "d.json", {"test_results": [row]})
    assert [r.metric for r in read_deepeval(p)] == ["faithfulness"]
    assert [r.metric for r in read_deepeval(p, metric="relevancy")] == ["relevancy"]


def test_ragas_contexts_not_a_list_is_failed(tmp_path: Path) -> None:
    rows = [
        {"response": "x", "retrieved_contexts": "5", "faithfulness": 1.0},
        {"response": "y", "retrieved_contexts": "['c']", "faithfulness": 1.0},
    ]
    recs = read_ragas(write(tmp_path, "r.json", rows), allow_index_ids=True)
    assert recs[0].status == "failed" and "retrieved_contexts" in (recs[0].failure_reason or "")
    assert recs[1].status == "ok"


def _de_row(name: str, logs: str) -> dict[str, Any]:
    md = {"score": 1.0, "success": True, "threshold": 0.5, "verboseLogs": logs}
    return {"name": name, "metrics_data": [md]}


@pytest.mark.parametrize(
    "logs", ["Claims:\n[]\n\nVerdicts:\n[]", 'Claims:\n["a"]\n\nVerdicts:\n[]']
)
def test_deepeval_zero_claims_is_failed(tmp_path: Path, logs: str) -> None:
    (rec,) = read_deepeval(write(tmp_path, "d.json", {"test_results": [_de_row("g", logs)]}))
    assert rec.status == "failed" and rec.score is None and rec.verdict is None
    assert (rec.failure_reason or "").startswith("zero_claims_default: ")


def test_deepeval_nonempty_claims_stay_ok(tmp_path: Path) -> None:
    logs = 'Claims:\n["a"]\n\nVerdicts:\n[{"verdict": "yes"}]'
    (rec,) = read_deepeval(write(tmp_path, "d.json", {"test_results": [_de_row("g", logs)]}))
    assert rec.status == "ok" and rec.verdict == "pass"


def test_deepeval_skipped_row_is_failed_record(tmp_path: Path) -> None:
    rows = [{"name": "s", "metrics_data": []}, {"name": "t"}]
    recs = read_deepeval(write(tmp_path, "d.json", {"test_results": rows}))
    assert [r.example_id for r in recs] == ["s", "t"]
    assert all(
        r.status == "failed" and (r.failure_reason or "").startswith("skipped") for r in recs
    )


def test_ragas_nan_reason_codes(tmp_path: Path) -> None:
    p = write(tmp_path, "r.json", [{"example_id": e, "faithfulness": None} for e in "abc"])

    def call(eid: str, text: str) -> str:
        msg = {"message": {"content": text}}
        return json.dumps({"example_id": eid, "request": {"messages": []}, "response": msg})

    lines = [call("a", '{\n  "properties": {}}'), call("b", '{"statements": []}'), call("c", "x")]
    raw = write(tmp_path, "raw.jsonl", "\n".join(lines))
    codes = [(r.failure_reason or "").split(":")[0] for r in read_ragas(p, raw)]
    assert codes == ["schema_echo", "no_statements", "framework_error"]
