"""Local judge smoke test (issue #5): can llama3.2 act as a RAG faithfulness judge on CPU?

Engineering smoke test only; 20 items are too few for any statistical claim.
Usage: uv run python scripts/judge_smoke.py [--limit N] [--runs N] [--model M]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

URL = "http://localhost:11434/api/generate"
ITEMS = Path(__file__).with_name("judge_smoke_items.json")
PROMPT = """Check an ANSWER against a CONTEXT.
Rule: verdict "pass" if everything the ANSWER says appears in the CONTEXT (the ANSWER does not need to
repeat all of the CONTEXT). Verdict "fail" only if the ANSWER contains a fact that is absent from or
contradicts the CONTEXT.
Reply with JSON only: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}"""  # noqa: E501


def call(model: str, item: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": PROMPT.format(**item),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "seed": 7},
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        raw = json.load(resp)["response"]
    latency = time.perf_counter() - start
    parsed, verdict = False, None
    try:
        obj = json.loads(raw)
        parsed = isinstance(obj, dict)
        verdict = obj.get("verdict") if parsed else None
    except json.JSONDecodeError:
        pass
    valid = verdict in ("pass", "fail")
    return {
        "id": item["id"],
        "raw": raw,
        "latency_s": latency,
        "parsed": parsed,
        "valid": valid,
        "verdict": verdict if valid else None,
        "match": valid and verdict == item["expected"],
    }


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--model", default="llama3.2:latest")
    args = ap.parse_args()
    items = json.loads(ITEMS.read_text())
    if args.limit:
        items = items[: args.limit]
    runs = [[call(args.model, it) for it in items] for _ in range(args.runs)]
    flat = [r for run in runs for r in run]
    exp = {it["id"]: it["expected"] for it in items}
    n = len(flat)
    lat = [r["latency_s"] for r in flat]
    false_pass = sum(1 for r in flat if r["verdict"] == "pass" and exp[r["id"]] == "fail")
    false_fail = sum(1 for r in flat if r["verdict"] == "fail" and exp[r["id"]] == "pass")
    flips = 0
    if len(runs) > 1:
        flips = sum(
            1 for a, b in zip(runs[0], runs[1], strict=True) if a["verdict"] != b["verdict"]
        )
    print(f"model={args.model} items={len(items)} runs={args.runs} calls={n}")
    print(f"latency median={statistics.median(lat):.1f}s p90={pct(lat, 0.9):.1f}s")
    print(f"parse failures={sum(not r['parsed'] for r in flat)}/{n}")
    print(f"invalid verdicts={sum(not r['valid'] for r in flat)}/{n}")
    print(f"agreement={sum(r['match'] for r in flat)}/{n}")
    print(f"false passes={false_pass} false fails={false_fail}")
    print(f"flips between runs={flips}/{len(items)}")
    for r in runs[0]:
        print(f"  #{r['id']} exp={exp[r['id']]} got={r['verdict']} {r['latency_s']:.1f}s")


if __name__ == "__main__":
    main()
