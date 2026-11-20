"""Spike: load every committed DeepEval/Ragas run via the adapters and report data health.

Plain script instead of a notebook (no new deps). Run:
uv run python spikes/notebook/inspect_outputs.py
Counts only: no agreement numbers (those live in spikes/compare).
"""

import json
from pathlib import Path

from judgecheck.adapters.deepeval import read_deepeval
from judgecheck.adapters.ragas import read_ragas
from judgecheck.join import join, load_labels

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spikes" / "generate" / "out"
GOLDEN = ROOT / "data" / "golden" / "v0"

golden = {i["id"] for i in json.loads((GOLDEN / "items.json").read_text("utf-8"))}
load = load_labels(GOLDEN / "human_labels.csv")
print(
    f"golden items: {len(golden)}; labels: {len(load.labels)}; unlabelled: {len(load.unlabelled)}"
)

for run in sorted(p for p in OUT.iterdir() if p.is_dir()):
    fw = run.name.split("-")[0]
    raw = run / "raw_calls.jsonl"
    if fw == "deepeval":
        recs = read_deepeval(run / "native_deepeval.json", raw)
    else:
        recs = read_ragas(run / "native_ragas.json", raw)
    ids = {r.example_id for r in recs}
    failed = [r for r in recs if r.status == "failed"]
    j = join(recs, load.labels)
    print(f"\n== {run.name} ==")
    print(f"records: {len(recs)} ({len(recs) - len(failed)} ok, {len(failed)} failed)")
    for r in failed:
        print(f"  failed {r.example_id}: {r.failure_reason}")
    print(
        f"ids not in golden items: {len(ids - golden)}; golden items without record: "
        f"{len(golden - ids)}"
    )
    print(
        f"join vs human labels: matched {len(j.joined)}, records without label "
        f"{len(j.missing_labels)}, labels without record {len(j.missing_judgements)}"
    )
