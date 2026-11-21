"""Edge branches of bootstrap_ci not covered by the test-first suite."""

import math

import pytest

from judgecheck.checks import agreement as ag
from judgecheck.models import HumanLabel, JoinedRecord, JudgeRecord


def _failed(eid: str) -> JoinedRecord:
    judge = JudgeRecord(
        example_id=eid,
        framework="ragas",
        metric="faithfulness",
        status="failed",
        failure_reason="schema_echo: echoed schema",
    )
    return JoinedRecord(judge=judge, human=HumanLabel(example_id=eid, label="fail"))


@pytest.mark.parametrize("kwargs", [{"stat": "f1"}, {"alpha": 0.0}, {"alpha": 1.0}])
def test_bootstrap_rejects_bad_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ag.bootstrap_ci([], **kwargs)  # type: ignore[arg-type]


def test_bootstrap_empty_is_nan() -> None:
    lo, hi = ag.bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_kappa_all_failed_is_nan() -> None:
    lo, hi = ag.bootstrap_ci([], [_failed("a"), _failed("b")], stat="kappa", n_boot=20)
    assert math.isnan(lo) and math.isnan(hi)
