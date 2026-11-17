# JudgeCheck — Engineering Plan (v1)

Rules of work: `CLAUDE.md` and `docs/CHARTER.md`. Decisions: `docs/DECISIONS.md`.
Status: **Approved 2026-11-17. P0 in progress.**

## 0. Scope (v1)

An open-source Python library + CLI that tells an AI engineer whether their LLM-judge
eval scores can be trusted before they gate a release on them. v1 audits DeepEval and
Ragas **RAG faithfulness** scores (pass/fail + score).

**Must do**
1. Read DeepEval/Ragas results (JSON/CSV) and a human-label CSV; join by example ID.
2. Flag every silent judge failure (NaN, parse error, empty context, out-of-range score)
   with the example, raw judge output (when available) and reason. Never average failures into scores.
3. Judge-human agreement: percent agreement, Cohen's kappa, confusion matrix, 95% bootstrap intervals.
4. Bias probes on a sample: context-order sensitivity, verbosity padding, repeat-run stability.
5. Exit code 1 when any threshold in `judgecheck.yaml` is breached (CI gate).
6. Markdown + HTML report; demo report published as a static page.
7. Record judge model, version, prompt and temperature with every run.

**Quality bar:** statistics verified against scikit-learn or hand-computed cases; ~85% line
coverage on core modules with a fixture per failure type; $0 local judge; one command
(`uv run judgecheck audit ...`) reproduces any reported number; mypy + Ruff clean;
README readable in 60 seconds.

**Later (v1.x):** promptfoo / Inspect AI / Langfuse adapters, more metrics, multi-turn and
agent judges, multi-labeler agreement, self-preference probe, dashboard, MCP server.
**Out of scope:** training judges, hosted SaaS, a labeling UI, "best judge" claims.

## 1. Architecture

```
 DeepEval JSON/CSV ─┐                      ┌─> checks/failures.py   (NaN, parse error, empty ctx, out-of-range)
 Ragas JSON/CSV ────┼─> adapters/ ─> join ─┼─> checks/agreement.py  (% agree, kappa, confusion, bootstrap CI)
 human_labels.csv ──┘   (normalise to      └─> probes/              (context-order, verbosity pad, repeat stability)
                         JudgeRecord)                │  calls DeepEval/Ragas metrics with a local Ollama judge
                                                     v
 judgecheck.yaml ─> config.py ─────────────> gate.py ──> exit 0 / 1 (CI)
                                                     └─> report/ (Markdown + HTML, Jinja2) ─> GitHub Pages static page
 every run ─> RunMeta (judge model, version, prompt hash, temperature, seed, input hashes) stored in the report
```

Package layout (`src/` layout, uv-managed):

```
src/judgecheck/
  models.py        Pydantic v2: JudgeRecord, HumanLabel, JoinedRecord, FailureFinding, RunMeta
  config.py        judgecheck.yaml schema + thresholds
  adapters/        base.py, deepeval.py, ragas.py   -> list[JudgeRecord]
  join.py          join by example_id; reports unmatched IDs (never drops silently)
  checks/          failures.py, agreement.py        pure functions, numpy only
  judges/          base.py (protocol), ollama.py
  probes/          context_order.py, verbosity.py, stability.py  (async, sampled)
  gate.py          thresholds -> verdict + exit code
  report/          markdown.py, html.py, templates/
  cli.py           Typer: `judgecheck audit|probe|report`
tests/             unit + fixtures per failure type; stats checked against scikit-learn
examples/rag_app/  small RAG app (LlamaIndex or plain Python) + Dockerfile
action.yml         GitHub Action wrapper
```

Design rules: core checks are pure functions (easy to test, no I/O); failed examples are
excluded from every average and listed in the report; bootstrap uses a fixed seed recorded
in RunMeta; scikit-learn is a **test-only** dependency (runtime stays light).

## 2. Phases and timeline (10 weeks + 1 holiday buffer)

| Phase | Dates | Deliverables | Exit gate |
|---|---|---|---|
| **P0 Setup** | Nov 17–23 | git + GitHub repo, branch protection on main, uv skeleton, CI (Ruff, mypy, pytest+cov, gitleaks, bandit, pip-audit), pre-commit, charter, Kanban board, ruflo swarm, guard hook, GitHub secret scanning + Dependabot | CI green on an empty PR |
| **P1 Spike + data layer** | W1: Nov 24–30 | Read DeepEval/Ragas parsing code; generate real results with Ollama; adapters + models + join; spike notebook; first 30 human labels | Adapters parse both tools' real output |
| **P2 Detectors + stats** | W2–3: Dec 1–14 | Silent-failure detectors (fixture per type); % agreement, kappa, confusion matrix, bootstrap CI | Stats match sklearn/hand cases; ≥85% cov |
| **P3 CLI, gate, report** | W4: Dec 15–21 | Typer CLI, `judgecheck.yaml`, exit codes, RunMeta, Markdown + HTML report | `uv run judgecheck audit …` reproduces numbers |
| buffer | Dec 22–Jan 4 | Holidays / catch-up | — |
| **P4 Probes + Action** | W5–6: Jan 5–18 | Async judge client, context-order/verbosity/stability probes, GitHub Action | Action fails a PR on breached threshold |
| **P5 Demo app** | W7–8: Jan 19–Feb 1 | Sample RAG app, Docker, 100–200 labels (local judge only, $0) | End-to-end demo audit from committed data |
| **P6 Ship** | W9–10: Feb 2–15 | Hardening, README (60-second scan), GitHub Pages demo report, v1.0 tag | Maintainer sign-off, release |

Each phase = one GitHub milestone. Maintainer approves each phase before the next starts.

## 3. Workflow

- **One issue → one branch (`p<phase>/<issue#>-slug`) → one PR.** Never commit/merge to main; maintainer squash-merges.
- Commits: conventional commits, GitHub noreply email (repo-local config), no attribution lines.
- Reviews per PR: peer (Dev) + QA. Research-critical PRs (stats, detectors, probes) also Senior AI Eng + Validator.
- Board: GitHub Projects v2 with fields Status, Owner (role), Phase, Progress (one line).

## 4. Team → agents → models

| Role | Agent type | Model | Typical work |
|---|---|---|---|
| Orchestrator | main session | opus | plan, route, explain to maintainer |
| Principal Architect | Plan | opus | design decisions, phase gates |
| Dev-1..3 | ruflo-core:coder | sonnet | implementation |
| Senior AI Eng | ruflo-core:researcher | opus | stats/probe correctness, DeepEval/Ragas internals |
| QA-1/2 | ruflo-core:reviewer | sonnet | tests, coverage, `/code-review` |
| Validator | ruflo-core:reviewer | opus | independent re-computation of reported numbers |
| Client Reviewer | general-purpose | sonnet | README/report from the user's viewpoint |
| Board keeper | general-purpose | haiku | board updates, labels, listings |

Ruflo: claims board (`claims_*`) for who-owns-what, `task_*` mirrored to the GitHub board,
`memory_store` for decisions, swarm for agent coordination (no second task board, no paid spend),
`aidefence_has_pii` scan before any push.

## 5. Tooling

- **Python stack:** uv, Python 3.12, Pydantic v2, Typer, numpy/pandas, Jinja2, httpx/asyncio, pytest (+pytest-cov, pytest-asyncio), Ruff, mypy, pre-commit.
- **Eval/AI:** DeepEval, Ragas, Ollama (local judge, $0), no API judge ($0 rule), LlamaIndex for the demo app.
- **CI/security:** GitHub Actions, gitleaks, bandit, pip-audit, Dependabot.
- **Ship:** Docker, GitHub Pages (static report), GitHub Action marketplace listing.
- **Claude Code skills:** ponytail (smallest change), code-review / security-review / simplify on every PR,
  ponytail-audit at each phase gate, dataviz + ui-ux-pro-max for the HTML report, claude-in-chrome to check the published report page.

## 6. Decisions

All resolved 2026-11-17, see `docs/DECISIONS.md` (D1–D8).
