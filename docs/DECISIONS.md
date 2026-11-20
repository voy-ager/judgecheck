# Decisions log

| # | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-11-17 | Develop on native Windows + uv for P0–P4; CI on Ubuntu; revisit WSL/Docker in P5 | 16 GB RAM, no GPU: a WSL VM would compete with Ollama; CI already catches Linux issues |
| D2 | 2026-11-17 | Maintainer writes stats and failure detectors; agents write tests first and review | Maintainer's learning goal; independent tests catch errors in either side |
| D3 | 2026-11-17 | Zero paid spend: local Ollama judge only, no API judge, demo report on GitHub Pages (not Vercel) | Maintainer requirement: no extra costs; Pages is free and needs no extra account |
| D4 | 2026-11-17 | "Position swap" probe becomes **context-order sensitivity** (shuffle retrieved passages, re-score) | Faithfulness scores one answer; there is no answer order to swap |
| D5 | 2026-11-17 | Probes call DeepEval/Ragas metric functions directly (optional extras), not copied prompts | Probing must use the judge's real prompt to be valid |
| D6 | 2026-11-17 | CI gate on kappa uses the bootstrap CI **lower bound** plus a minimum label count | 30 labels give intervals of about ±0.25; a point estimate would flap |
| D7 | 2026-11-17 | No default judge yet. P0 smoke test (#5, 20 items, CPU, temp 0): `llama3.2` 3B said "fail" on all 40 calls (50% agreement, 20 false fails; stable, 0 parse errors, ~5 s/call); `qwen2.5-coder:3b` scored 14/20 with 6 false passes (misses subtle hallucinations). Decide in the P1 spike using real Ragas/DeepEval faithfulness (D5) on both models; a larger local model needs maintainer approval for the download | Both 3B judges fail in opposite directions (too harsh vs too lenient); false passes are the dangerous kind for a release gate. 20 items is a smoke test, not evidence |
| D8 | 2026-11-17 | One board (GitHub Project); ruflo for swarm, memory and claims only | Two boards double the token cost |
| D9 | 2026-11-17 | Spike dependencies live in `spikes/generate/` (own `pyproject.toml` + `uv.lock`, `package = false`), not in the core lockfile | DeepEval and Ragas pull about 130 packages; the core project must stay small and its lockfile auditable |
| D10 | 2026-11-17 | Known vulnerability PYSEC-2026-2447 in diskcache 5.6.3 (pulled by ragas, no fixed version; found with `pip-audit` 2.10.1 on the spike lock) is accepted only inside the local spike env, which never ships | No fix exists; the env runs locally on our own data, and the dependency must not enter the core lockfile (D9). Revisit if Ragas becomes an optional extra |
| D11 | 2026-11-20 | Dismissed Dependabot alert #1 (diskcache, unsafe pickle, medium) | Only in the isolated spike env, which never ships; cache files are local and written by our own runs |
