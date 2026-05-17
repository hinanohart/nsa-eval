# oss-init-nsa runbook (companion to SKILL.md)

This document expands SKILL.md with the per-step procedure for a Day 0 init. It is intended to
be read by both human and AI agent operators. SKILL.md remains the entry point; this file
holds the detail that would otherwise bloat it.

## Step order (matches design memo §H)

### 1. Read context

Before any tool call, read in this order:

- `CLAUDE.md` at repo root — the hard rules.
- `README.md` — public-facing scope and DoD.
- `~/.claude/projects/-home-runza/memory/nsa-final-design-2026-05-17.md` — private design memo,
  not on the repo.

### 2. Declare understanding (R16-equivalent)

In one sentence, restate what is about to happen. Example: "I will initialise the public
GitHub repository, seed the 85-issue board, and trigger the first CI run; everything else is
deferred to the weekly Claude Code Routine."

### 3. Pre-flight checks

Run all in parallel:

```bash
gh auth status                     # scope must include repo + workflow
python3 --version                  # >= 3.10
which uv                           # install if missing
which kaggle                       # required for Kaggle dataset push
which huggingface-cli || true      # optional at Day 0; gated by HF_TOKEN
uname -m                           # arm64 -> Mac path, x86_64 -> Linux/Kaggle path
```

Fail loudly if any required tool is missing — do not silently fall back.

### 4. Gate A — `gh repo create` (human approval required)

If `gh repo view hinanohart/nsa-eval` returns 404:

```bash
gh repo create hinanohart/nsa-eval \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "MPS-native Native Sparse Attention reference + unified long-context eval suite"
```

This is a one-way action (the repo becomes public immediately). Do not run automatically;
require explicit human go-ahead in the conversation.

### 5. Initial scaffold commit + submodule

Already present on disk before this step. Add the third-party submodule:

```bash
git submodule add https://github.com/fla-org/native-sparse-attention.git \
  third_party/native-sparse-attention
git commit -m "vendor fla-org native-sparse-attention submodule"
git push
```

### 6. Hugging Face dataset / space create (deferred)

These steps need `HF_TOKEN` provisioned through gate A'. Until then, the workflows skip the HF
push step and write the artifact locally only. The actual create commands are:

```bash
huggingface-cli repo create nsa-eval-bench --type dataset --organization hinanohart
huggingface-cli repo create nsa-eval-demo --type space --space_sdk gradio --organization hinanohart
```

### 7. Kaggle dataset create

```bash
mkdir -p .kaggle-meta && \
kaggle datasets init -p .kaggle-meta && \
# edit .kaggle-meta/dataset-metadata.json to set id "hinanohart/nsa-eval-bench" and title
kaggle datasets create -p .kaggle-meta
```

### 8. Gate A' — repo secrets (human approval, browser only)

Set these via the GitHub web UI under Settings → Secrets:

- `HF_TOKEN` — Hugging Face write token, scoped to `hinanohart` org datasets and spaces.
- `KAGGLE_USERNAME` and `KAGGLE_KEY` — from `~/.kaggle/kaggle.json`.

Do not paste tokens into the chat. The conversation history is privileged.

### 9. Trigger CI + seed issues

```bash
gh workflow run ci.yml --ref main
bash scripts/seed-issues.sh
```

`scripts/seed-issues.sh` is idempotent: re-running it skips issues whose titles already exist.

### 10. Hand off to the weekly Claude Code Routine

Confirm the routine is scheduled. The routine runs Mon 09:00 UTC and:

- triages new issues
- opens PRs for week-N tickets that match the threshold criteria written in the issue body
- updates `benchmarks/results/` from any successful bench-dispatch / bench-mps runs

Write a one-line status log to `~/.claude/projects/-home-runza/memory/oss-init-nsa-<date>.md`.

## Failure-handling

- If a step fails, do not retry blindly. Move the failed artifact to
  `experiments/_wip/<short-name>/` per the hard rule in CLAUDE.md, and open an issue.
- If gate A or A' is declined, stop. Do not work around it.

## What is intentionally not automated

- arXiv submission (gate B, Phase 2).
- License change (gate C).
- Scope re-architecture away from the 6-decision set (gate D).
- Marketing / launch posts.
