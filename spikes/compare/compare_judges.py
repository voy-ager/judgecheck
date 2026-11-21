"""Spike: compare 6 golden-set judge runs (3 models x DeepEval/Ragas) to the human labels.

Simple counts for a decision, NOT the official statistics (the maintainer writes
checks/agreement.py in P2). With 30 items, no difference here is statistically strong.

Pass threshold: DeepEval's own default (0.5, read from each result's `threshold`).
Ragas has no pass threshold, so 0.5 is used to match DeepEval (our assumption).
Wall time: meta.json has no duration field, so it is meta timestamp (run end) minus
the first raw call timestamp (approximate; excludes the first call's own latency).

Caveats: "unjudged" = DeepEval extracted zero claims and skipped the verdict call; the adapter
marks these failed (zero_claims_default, D11), so they are counted wrong, not
excluded. Cross-framework comparison mixes judge quality with format enforcement
(DeepEval enforces the output schema, Ragas does not);
the within-framework comparison across models is the fair one.

Threshold 1.0 matches the labelling rule (pass = every claim supported) but was chosen after
seeing these data; confirm on fresh labelled items (P5).

Run: uv run python spikes/compare/compare_judges.py
"""

import json
from datetime import datetime
from pathlib import Path

from judgecheck.adapters.deepeval import read_deepeval
from judgecheck.adapters.ragas import read_ragas
from judgecheck.join import load_labels
from judgecheck.models import JudgeRecord

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spikes" / "generate" / "out"
GOLD = ROOT / "data" / "golden" / "v0"
RAGAS_THRESHOLD = 0.5
ZERO = "zero_claims_default"
MODELS = ["llama3.2_latest", "qwen2.5-coder_3b", "qwen2.5_7b"]


def wall_seconds(run: Path) -> float:
    end = datetime.fromisoformat(json.loads((run / "meta.json").read_text("utf-8"))["timestamp"])
    first = json.loads((run / "raw_calls.jsonl").read_text("utf-8").splitlines()[0])["ts"]
    return (end - datetime.fromisoformat(first)).total_seconds()


def load(fw: str, run: Path) -> list[JudgeRecord]:
    raw = run / "raw_calls.jsonl"
    if fw == "deepeval":
        return read_deepeval(run / "native_deepeval.json", raw)
    return read_ragas(run / "native_ragas.json", raw, threshold=RAGAS_THRESHOLD)


def echoed_schema(run: Path) -> tuple[int, int]:
    """(raw calls whose reply is the JSON schema echoed back, total raw calls)."""
    calls = [json.loads(x) for x in (run / "raw_calls.jsonl").read_text("utf-8").splitlines()]
    texts = [c["response"]["choices"][0]["message"]["content"] for c in calls]
    echo = sum('"properties"' in t[:30] for t in texts)
    return echo, len(calls)


def main() -> None:
    human = {lab.example_id: lab.label for lab in load_labels(GOLD / "human_labels.csv").labels}
    print("NOTE: cross-framework comparison mixes judge quality with format enforcement")
    print("(DeepEval enforces the output schema, Ragas does not); compare within a framework.")
    print("NOTE: 30 items; no difference is statistically strong. Pass threshold 0.5 (DeepEval")
    print("default; assumed for Ragas). The main figures use 0.5; see the t= lines for 0.7, 1.0.")
    print("NOTE: t=1.0 matches the labelling rule (pass = every claim supported), so it is")
    print("principled, but it was chosen after seeing these data: confirm on fresh labelled")
    print("items (P5) before relying on it.")
    print()
    for fw in ("deepeval", "ragas"):
        for model in MODELS:
            run = OUT / f"{fw}-{model}"
            recs = {r.example_id: r for r in load(fw, run)}
            scored = [i for i in human if i in recs and recs[i].verdict is not None]
            failed = {
                i: (recs[i].failure_reason if i in recs else "no record")
                for i in human
                if i not in scored
            }
            wrong = [i for i in scored if recs[i].verdict != human[i]]
            fp = [i for i in wrong if recs[i].verdict == "pass"]
            ff = [i for i in wrong if recs[i].verdict == "fail"]
            n, s = len(human), len(scored)
            print(f"== {fw} / {model}  wall={wall_seconds(run):.0f}s")
            print(f"scored {s}/{n}, failed/NaN {len(failed)}")
            for why in sorted(set(failed.values()), key=str):
                ids = [i for i, w in failed.items() if w == why]
                shown = ids if len(ids) < 6 else f"{ids[0]}..{ids[-1]}"
                print(f"  failed {len(ids)}x ({why}): {shown}")
            print(
                f"agreement on scored: {(s - len(wrong)) / s:.0%} ({s - len(wrong)}/{s})"
                if s
                else "agreement on scored: n/a"
            )
            print(f"agreement, failed=wrong: {(s - len(wrong)) / n:.0%} ({s - len(wrong)}/{n})")
            if fw == "deepeval":
                uj = sorted(i for i, r in recs.items() if (r.failure_reason or "").startswith(ZERO))
                print(f"unjudged (zero claims, default score): {len(uj)} {uj}")
            if failed and fw == "ragas":
                e, t = echoed_schema(run)
                print(f"  real cause: judge echoed the JSON schema in {e}/{t} calls (retried)")
            note = f" (all {n} failed)" if not scored else ""
            print("  by pass threshold (score >= t; failed counted wrong):")
            for th in (0.5, 0.7, 1.0):
                v = {i: "pass" if recs[i].score >= th else "fail" for i in scored}  # type: ignore[operator]
                g = sum(v[i] == human[i] for i in scored)
                tfp = [i for i in scored if v[i] == "pass" and human[i] == "fail"]
                tff = [i for i in scored if v[i] == "fail" and human[i] == "pass"]
                print(
                    f"    t={th}: agree {g / n:.0%} ({g}/{n}), "
                    f"false pass {len(tfp)}, false fail {len(tff)}{note}"
                )
            print(f"false pass (judge pass, human fail): {len(fp)} {fp}{note}")
            print(f"false fail (judge fail, human pass): {len(ff)} {ff}{note}")
            for i in ("g005", "g029"):
                r = recs.get(i)
                what = (
                    "no record"
                    if r is None
                    else (
                        f"FAILED ({r.failure_reason})"
                        if r.verdict is None
                        else f"score={r.score} verdict={r.verdict}"
                    )
                )
                print(f"  {i} (human={human[i]}): {what}")
            print()
    intended = {
        k: v["label"]
        for k, v in json.loads((GOLD / "intended_labels.json").read_text("utf-8")).items()
    }
    diff = [i for i in human if human[i] != intended.get(i)]
    print(
        f"== sanity: human vs intended labels: {len(human) - len(diff)}/{len(human)} agree; "
        f"differ: {diff}"
    )


if __name__ == "__main__":
    main()
