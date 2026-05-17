# CLAUDE.md — repository-level guidance for AI coding assistants

This file is read by Claude Code (and compatible agents) when working inside this repository.
It supplements the user's global guidance with project-specific rules.

## Project at a glance

- **Goal**: First OSS that runs Native Sparse Attention on Apple Silicon (MPS), plus a unified
  long-context evaluation suite. See `README.md` for the v0.1 acceptance criteria.
- **Maintainer dev environment**: WSL2 Linux x86_64. MPS validation requires a Mac contributor.
- **Reference design**: `~/.claude/projects/-home-runza/memory/nsa-final-design-2026-05-17.md`
  (private to the maintainer; do not commit).

## Hard rules (do not violate)

1. **No deletion of failed experiments.** Move them to `experiments/_wip/<short-name>/` with a
   README that explains what was tried and why it failed. Re-running a dead-end approach is
   the most expensive class of mistake in this repo.
2. **No "permanent" or "fully automated" language** in README, CHANGELOG, or papers. Use
   "minimum human intervention through v0.1" instead. Past incidents have shown these phrases
   become hostages to the next regression.
3. **No internal rule numbers (R-series) in commit messages or PR descriptions.** They are
   maintainer-internal and rot quickly.
4. **No arXiv / Hugging Face submissions from CI.** Submissions are gated on a human approval
   (gate B / gate A' in the design memo).
5. **No leaderboard public launch before Phase 3.** Leaderboard workflow is gated on
   `vars.LEADERBOARD_ENABLED`; do not flip without a Phase 3 readiness review.

## Soft rules (prefer, ask before deviating)

- Prefer editing existing files over creating new ones.
- Prefer reproducing a benchmark cell on CPU before bringing in CUDA / MPS device fallback
  logic. Numeric equivalence on CPU is the contract; device-specific paths optimise it.
- Prefer adding a `tests/test_attention/` numeric regression check when touching any file in
  `src/nsa_eval/attention/`.
- Prefer the existing 6-decision scope (MPS-first, 3-month v0.1, NeurIPS D&B Track) when
  proposing new features. Scope changes need gate D.

## Code style

- Python 3.10+. Type hints required for public APIs.
- `ruff` for lint and format (config in `pyproject.toml`).
- One short docstring per public symbol; do not write multi-paragraph docstrings.
- Comments only when the *why* is non-obvious. The code's *what* should be self-documenting.

## When in doubt

Open an issue. The maintainer's bandwidth is limited; structured questions are cheaper than
PR back-and-forth.
