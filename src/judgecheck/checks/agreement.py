"""Judge-vs-human agreement statistics. Pure Python, no numpy at runtime (D13).

`joined` = ok judge records with a human label; `failed` = failed judge records with a label
(both from `JoinResult`). A failed record has no verdict, so it counts as a disagreement in
percent agreement and its bootstrap, and is left out of kappa (D13). Undefined -> NaN.
"""

import math
import random
from collections.abc import Sequence

from judgecheck.models import JoinedRecord

LABELS = ("pass", "fail")
FAILED = "failed"  # judge-side cell for records the judge never scored


def _verdict(r: JoinedRecord) -> str:
    return r.judge.verdict or FAILED  # ok records always carry a verdict


def percent_agreement(joined: Sequence[JoinedRecord], failed: Sequence[JoinedRecord] = ()) -> float:
    """Share of all labelled items where the judge verdict equals the human label."""
    n = len(joined) + len(failed)
    if n == 0:
        return math.nan
    return sum(_verdict(r) == r.human.label for r in joined) / n


def cohens_kappa(joined: Sequence[JoinedRecord]) -> float:
    """Cohen's kappa on scored records: (po - pe) / (1 - pe). NaN when pe == 1 or empty."""
    n = len(joined)
    if n == 0:
        return math.nan
    po = sum(_verdict(r) == r.human.label for r in joined) / n
    pe = sum(
        (sum(r.human.label == lab for r in joined) / n)
        * (sum(_verdict(r) == lab for r in joined) / n)
        for lab in LABELS
    )
    if pe == 1.0:  # both raters constant on the same label: chance agreement is total
        return math.nan
    return (po - pe) / (1 - pe)


def confusion_matrix(
    joined: Sequence[JoinedRecord], failed: Sequence[JoinedRecord] = ()
) -> dict[tuple[str, str], int]:
    """Counts keyed by (human label, judge verdict); judge side includes "failed"."""
    cells = {(h, j): 0 for h in LABELS for j in (*LABELS, FAILED)}
    for r in (*joined, *failed):
        cells[(r.human.label, _verdict(r))] += 1
    return cells


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation quantile (same rule as numpy's default)."""
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(
    joined: Sequence[JoinedRecord],
    failed: Sequence[JoinedRecord] = (),
    *,
    stat: str = "agreement",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling labelled items (ok and failed together).

    For kappa, failed items drawn into a resample are dropped before computing kappa, and
    resamples where kappa is undefined are skipped. (nan, nan) if nothing is computable.
    """
    if stat not in ("agreement", "kappa"):
        raise ValueError(f"unknown stat {stat!r}; use 'agreement' or 'kappa'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    pool = [*joined, *failed]
    if not pool:
        return (math.nan, math.nan)
    rng = random.Random(seed)  # noqa: S311 - statistical resampling, not security
    values: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(pool, k=len(pool))
        ok = [r for r in sample if r.judge.status == "ok"]
        if stat == "agreement":
            v = percent_agreement(ok, [r for r in sample if r.judge.status != "ok"])
        else:
            v = cohens_kappa(ok)
        if not math.isnan(v):
            values.append(v)
    if not values:
        return (math.nan, math.nan)
    values.sort()
    return (_quantile(values, alpha / 2), _quantile(values, 1 - alpha / 2))
