# Decisions log

| # | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-11-17 | Develop on native Windows + uv for P0–P4; CI on Ubuntu; revisit WSL/Docker in P5 | 16 GB RAM, no GPU: a WSL VM would compete with Ollama; CI already catches Linux issues |
| D2 | 2026-11-17 | Maintainer writes stats and failure detectors; agents write tests first and review | Maintainer's learning goal; independent tests catch errors in either side |
| D3 | 2026-11-17 | Zero paid spend: local Ollama judge only, no API judge, demo report on GitHub Pages (not Vercel) | Maintainer requirement: no extra costs; Pages is free and needs no extra account |
| D4 | 2026-11-17 | "Position swap" probe becomes **context-order sensitivity** (shuffle retrieved passages, re-score) | Faithfulness scores one answer; there is no answer order to swap |
| D5 | 2026-11-17 | Probes call DeepEval/Ragas metric functions directly (optional extras), not copied prompts | Probing must use the judge's real prompt to be valid |
| D6 | 2026-11-17 | CI gate on kappa uses the bootstrap CI **lower bound** plus a minimum label count | 30 labels give intervals of about ±0.25; a point estimate would flap |
| D7 | 2026-11-17 | Default judge `llama3.2` (3B) for the spike; larger local model only if the P0 smoke test shows it's usable on CPU | No GPU; repeat-run probes need many calls |
| D8 | 2026-11-17 | One board (GitHub Project); ruflo for swarm, memory and claims only | Two boards double the token cost |
