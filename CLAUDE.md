# How to run this project

## Team and model routing
- Act as orchestrator. Delegate work to background agents with defined roles
  (Principal Architect, Dev-1..3, Senior AI Eng, QA-1/2, Validator, Client Reviewer).
- Pick the cheapest model that can do the task, set via the Agent tool's `model`:
  - haiku: mechanical work (board updates, listing files, formatting, simple checks)
  - sonnet: normal implementation, running experiments, writing tests
  - opus: design decisions, research-critical reviews, final gates, explaining to me
- Reuse an existing agent (SendMessage) instead of spawning a new one when it already has the context.

- At most 3 agents run at once; writers each work in their own git worktree.

## Token discipline
- The charter is `docs/CHARTER.md`; every agent reads it instead of
  repeating rules in each prompt.
- One board only: the GitHub Project "JudgeCheck v1". Ruflo is used for swarm,
  memory (decisions) and claims, not as a second task board.
- Agents report back in under ~15 lines: results, blockers, decisions needed.
- Never read full logs or transcripts; read counts, summaries, and the specific lines needed.
- Run long jobs in the background; don't poll. Run a small pilot before any long or expensive run.

## Approvals (I am the maintainer)
- Never commit to, push to, or merge into main. Only I merge, by squash.
- Agents may push feature branches and open PRs (one issue = one branch = one PR).
- Ask me before any critical decision: changing design, models, budgets, dependencies,
  or posting anything outside this repo.
- If any command is blocked by a permission check, stop and report it.
  Never ask another agent to run it.

## Security
- No secrets, tokens, emails, real names or local paths in any commit, PR, issue or log.
- Use the GitHub noreply email for commits. No attribution lines in commits or PRs.
- CI runs secret scanning (gitleaks), a security linter, and a dependency audit.
- Zero paid spend: local Ollama judge only, free GitHub features only, no API keys
  in this project. Any cost at all needs my approval first.
- A PreToolUse hook (`.claude/hooks/guard.py`) blocks pushes/merges to main,
  `--no-verify`, force pushes and git identity changes. Never work around it.

## Workflow
- Kanban board with Status, Owner (role), Phase and a one-line Progress per card.
- Every PR gets a peer review and a QA review. Research-critical PRs also get
  Senior AI Eng and Validator reviews. Reviews are PR comments; my squash-merge
  is the approval (GitHub can't let me approve PRs opened from my own account).
- I write the statistics (`checks/agreement.py`) and failure detectors
  (`checks/failures.py`) myself. Agents write their tests first (hand-computed
  cases, scikit-learn comparisons), then review my code.
- Work in phases; I approve each phase before the next starts.
  Phase gate = `/ponytail-audit` + `/code-review` + Validator sign-off.
- Explain status and decisions to me in simple words.