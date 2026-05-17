---
name: oss-init-nsa
description: Day-0 / week-1 init runbook for nsa-eval. Run once on a fresh clone to set up the environment, register the auto-cron, and confirm the CI green.
---

# oss-init-nsa

Single-purpose skill: take a fresh clone of `hinanohart/nsa-eval`, confirm the v0.1 contract is intact, and bring the repository to the state where the autonomous weekly Claude Code Routine can take over.

## Pre-flight

1. `gh auth status` — must show `repo` and `workflow` scopes.
2. `python3 --version` — 3.10 or newer.
3. `which uv` — install via `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing.
4. `uname -m` — `arm64` = Mac path, `x86_64` Linux = WSL2/Kaggle path. The skill picks the right notebook template based on this.

## Init steps

1. `uv venv && source .venv/bin/activate && uv pip install -e ".[dev,bench]"`
2. `ruff check .` and `pytest -m "not slow and not cuda and not mps"` — both must pass.
3. If `gh repo view` returns 404, `gh repo create hinanohart/nsa-eval --public --source=. --remote=origin --push`.
4. Trigger `gh workflow run ci.yml` and wait until the run is green.
5. Open the v0.1 issue board (issues 1-85). If the board is empty, run the bulk-create script (`scripts/seed-issues.sh`).
6. Confirm `vars.HAVE_MPS_RUNNER` and `vars.LEADERBOARD_ENABLED` exist (set to `false` initially).

## Human gates (do not bypass)

- A: `gh repo create` to a public repository.
- A': Adding `HF_TOKEN` / `KAGGLE_KEY` to repo secrets. Only via the GitHub UI; never paste tokens into the chat.
- B: arXiv submission (Phase 2). Endorser must be confirmed before submission.
- C: License change away from MIT.
- D: Re-architecting away from the v0.1 6-decision set.

## Output

The skill writes its run log to `~/.claude/projects/-home-runza/memory/oss-init-nsa-<date>.md` so the next session can pick up state.
