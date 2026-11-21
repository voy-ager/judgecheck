"""Tests for judgecheck.checks.agreement (maintainer implements, #41).

Proposed API (pure functions, no numpy needed by the tests):
    percent_agreement(joined, failed=()) -> float
    cohens_kappa(joined) -> float                      # NaN when undefined
    confusion_matrix(joined, failed=()) -> dict[(human, judge), int]
    bootstrap_ci(joined, failed=(), *, stat="agreement", n_boot=1000, seed=0, alpha=0.05)
        -> (lo, hi)
`joined` / `failed` are the lists from JoinResult. Failed records count in the denominator
as wrong (P1 Validator trap); they never enter kappa (no judge verdict to compare).
"""

import math

import pytest

from judgecheck.checks import agreement as ag
from judgecheck.models import HumanLabel, JoinedRecord, JudgeRecord

# Stays green until the maintainer implements the module; XPASS is allowed (strict=False).
pytestmark = pytest.mark.xfail(
    strict=False, reason="awaiting maintainer implementation, #41", raises=AttributeError
)


def pair(i: int, judge: str, human: str) -> JoinedRecord:
    eid = f"e{i:04d}"
    return JoinedRecord(
        judge=JudgeRecord(
            example_id=eid,
            framework="ragas",
            metric="faithfulness",
            score=1.0 if judge == "pass" else 0.0,
            verdict=judge,
            status="ok",
        ),
        human=HumanLabel(example_id=eid, label=human),
    )


def fail(i: int, human: str = "pass") -> JoinedRecord:
    eid = f"f{i:04d}"
    return JoinedRecord(
        judge=JudgeRecord(
            example_id=eid,
            framework="ragas",
            metric="faithfulness",
            status="failed",
            failure_reason="NaN",
        ),
        human=HumanLabel(example_id=eid, label=human),
    )


def make(cells: dict[tuple[str, str], int]) -> list[JoinedRecord]:
    """cells[(judge, human)] = count."""
    out: list[JoinedRecord] = []
    for (j, h), n in cells.items():
        out += [pair(len(out) + k, j, h) for k in range(n)]
    return out


# Textbook 2x2, n=50: both pass 20, judge pass/human fail 5, judge fail/human pass 10, both fail 15
TEXTBOOK = {("pass", "pass"): 20, ("pass", "fail"): 5, ("fail", "pass"): 10, ("fail", "fail"): 15}


def test_perfect_agreement():
    d = make({("pass", "pass"): 3, ("fail", "fail"): 2})
    assert ag.percent_agreement(d) == 1.0  # 5/5
    assert ag.cohens_kappa(d) == pytest.approx(1.0)  # po=1, pe=.6*.6+.4*.4=.52 -> (1-.52)/.48


def test_total_disagreement():
    d = make({("pass", "fail"): 2, ("fail", "pass"): 2})
    assert ag.percent_agreement(d) == 0.0  # 0/4
    # po=0, pe=.5*.5+.5*.5=.5 -> (0-.5)/(1-.5) = -1
    assert ag.cohens_kappa(d) == pytest.approx(-1.0)


def test_textbook_kappa():
    d = make(TEXTBOOK)
    assert ag.percent_agreement(d) == pytest.approx(0.7)  # (20+15)/50
    # judge pass=25, human pass=30; pe=(25*30+25*20)/50^2=1250/2500=.5; (.7-.5)/(1-.5)=.4
    assert ag.cohens_kappa(d) == pytest.approx(0.4)


def test_kappa_one_constant_rater_is_zero():
    d = make({("pass", "pass"): 2, ("pass", "fail"): 2})  # judge always pass
    # po=.5; pe=1*.5+0*.5=.5 -> kappa=0 (agreement no better than chance)
    assert ag.cohens_kappa(d) == pytest.approx(0.0)


def test_kappa_both_constant_is_undefined():
    # DECISION for maintainer: pe=1 gives 0/0. Test assumes NaN (alternatives: 1.0 or ValueError).
    d = make({("pass", "pass"): 4})
    assert math.isnan(ag.cohens_kappa(d))


def test_empty_input():
    # DECISION for maintainer: assumes NaN, not ZeroDivisionError.
    assert math.isnan(ag.percent_agreement([]))
    assert math.isnan(ag.cohens_kappa([]))


def test_all_failed_input():
    f = [fail(i) for i in range(3)]
    assert ag.percent_agreement([], f) == 0.0  # 0 right / 3 attempted
    assert math.isnan(ag.cohens_kappa([]))  # no judged pairs


def test_failures_count_as_wrong_in_denominator():
    d = make({("pass", "pass"): 3})
    assert ag.percent_agreement(d, [fail(0)]) == pytest.approx(0.75)  # 3/4, not 3/3
    assert ag.percent_agreement(d, [fail(0), fail(1)]) == pytest.approx(0.6)  # 3/5
    assert ag.percent_agreement(d) == 1.0  # guard: dropping failures would give 1.0


def test_confusion_matrix_counts_and_failures():
    cm = ag.confusion_matrix(make(TEXTBOOK), [fail(0, "pass"), fail(1, "fail")])
    assert cm[("pass", "pass")] == 20
    assert cm[("fail", "pass")] == 5  # human fail, judge pass
    assert cm[("pass", "fail")] == 10  # human pass, judge fail
    assert cm[("fail", "fail")] == 15
    assert cm[("pass", "failed")] == 1
    assert cm[("fail", "failed")] == 1
    # key order is (human, judge); total equals all attempted records
    assert sum(cm.values()) == 52


def _ci(d, **kw):
    return ag.bootstrap_ci(d, **kw)


def test_bootstrap_deterministic_and_contains_point():
    d = make({("pass", "pass"): 16, ("pass", "fail"): 4})  # 0.8
    lo, hi = _ci(d, seed=7)
    assert (lo, hi) == _ci(d, seed=7)  # same seed, same answer
    assert 0.0 <= lo <= ag.percent_agreement(d) <= hi <= 1.0
    assert lo < hi


def test_bootstrap_width_shrinks_with_n():
    small = make({("pass", "pass"): 16, ("pass", "fail"): 4})
    big = make({("pass", "pass"): 160, ("pass", "fail"): 40})  # same 0.8, 10x n
    w = lambda d: (lambda ci: ci[1] - ci[0])(_ci(d, seed=1))  # noqa: E731
    assert w(big) < w(small)


def test_bootstrap_counts_failures_as_wrong():
    d = make({("pass", "pass"): 5})
    assert _ci(d, seed=1) == (1.0, 1.0)  # no variance without failures
    lo, hi = _ci(d, failed=[fail(i) for i in range(5)], seed=1)
    assert lo < 1.0 and hi < 1.0  # 5 right of 10 attempted; dropping failures would stay 1.0
    assert _ci([], failed=[fail(0)], seed=1) == (0.0, 0.0)


def test_bootstrap_kappa_stat():
    lo, hi = _ci(make(TEXTBOOK), stat="kappa", seed=3)
    assert lo <= ag.cohens_kappa(make(TEXTBOOK)) <= hi


# sklearn cross-checks (dev dependency).
def test_kappa_matches_sklearn():
    from sklearn import metrics

    d = make({("pass", "pass"): 7, ("pass", "fail"): 3, ("fail", "pass"): 2, ("fail", "fail"): 8})
    j = [r.judge.verdict for r in d]
    h = [r.human.label for r in d]
    assert ag.cohens_kappa(d) == pytest.approx(metrics.cohen_kappa_score(h, j))


def test_confusion_matches_sklearn():
    from sklearn import metrics

    d = make(TEXTBOOK)
    j = [r.judge.verdict for r in d]
    h = [r.human.label for r in d]
    m = metrics.confusion_matrix(h, j, labels=["pass", "fail"])  # rows human, cols judge
    cm = ag.confusion_matrix(d)
    assert [[cm[(a, b)] for b in ("pass", "fail")] for a in ("pass", "fail")] == m.tolist()
