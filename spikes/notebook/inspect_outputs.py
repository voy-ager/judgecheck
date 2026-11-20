"""Spike: load committed DeepEval/Ragas outputs via the adapters, join by example_id, report.

Plain script instead of a notebook (no new deps). Run: uv run python spikes/notebook/inspect_outputs.py
Does not touch human labels.
"""

import json
from pathlib import Path

from judgecheck.adapters.deepeval import read_deepeval
from judgecheck.adapters.ragas import read_ragas

OUT = Path(__file__).resolve().parents[1] / "generate" / "out"
GOLDEN = Path(__file__).resolve().parents[2] / "data" / "golden" / "v0" / "items.json"

de_dir, rg_dir = OUT / "deepeval-qwen2.5-coder_3b", OUT / "ragas-qwen2.5-coder_3b"
records = {
    "deepeval": read_deepeval(de_dir / "native_deepeval.json", de_dir / "raw_calls.jsonl"),
    "ragas": read_ragas(rg_dir / "native_ragas.json", rg_dir / "raw_calls.jsonl"),
}

print("== Record counts ==")
for fw, recs in records.items():
    ok = sum(r.status == "ok" for r in recs)
    print(f"{fw}: {len(recs)} records, {ok} ok, {len(recs) - ok} failed")

print("\n== Failed rows ==")
failed = [r for recs in records.values() for r in recs if r.status == "failed"]
for r in failed:
    print(f"{r.framework} {r.example_id}: {r.failure_reason}")
if not failed:
    print("none")

print("\n== Join by example_id ==")
ids = {fw: {r.example_id for r in recs} for fw, recs in records.items()}
print("deepeval only:", sorted(ids["deepeval"] - ids["ragas"]) or "none")
print("ragas only:   ", sorted(ids["ragas"] - ids["deepeval"]) or "none")
print("both:", sorted(ids["deepeval"] & ids["ragas"]))
golden = {i["id"] for i in json.loads(GOLDEN.read_text("utf-8"))}
for fw, s in ids.items():
    print(f"{fw} ids not in golden set: {sorted(s - golden) or 'none'}")
for fw, s in ids.items():
    print(f"golden items with no {fw} record: {len(golden - s)} of {len(golden)}")

print("\n== Side by side (score, status) ==")
by = {fw: {r.example_id: r for r in recs} for fw, recs in records.items()}
for eid in sorted(ids["deepeval"] & ids["ragas"]):
    d, g = by["deepeval"][eid], by["ragas"][eid]
    print(f"{eid}: deepeval={d.score} [{d.status}]  ragas={g.score} [{g.status}]")
