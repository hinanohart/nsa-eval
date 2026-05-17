#!/usr/bin/env bash
# scripts/finish-setup.sh — one-shot human-only setup the Claude session cannot perform.
#
# Why this script exists
#   Day 0 init + the round-3 audit fix loop wired the entire OSS scaffold end-to-end except
#   for items that require credentials, GitHub web UI clicks, or third-party human approval.
#   Those items are listed below and bundled into one idempotent run so the maintainer
#   doesn't have to keep a checklist in their head.
#
# Idempotent: every step skips itself if already done. Re-run safely.
#
# Usage:
#   bash scripts/finish-setup.sh                # interactive, asks before each section
#   bash scripts/finish-setup.sh --yes          # non-interactive, skip prompts
#   bash scripts/finish-setup.sh --check        # report status only, change nothing
#
# Requires: gh CLI logged in (`gh auth status`); kaggle / huggingface CLIs optional (the
# script can install them on demand).

set -euo pipefail

REPO="${REPO:-hinanohart/nsa-eval}"
YES=0
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --check)  CHECK=1 ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

confirm() {
  local prompt="$1"
  if [ "$YES" -eq 1 ] || [ "$CHECK" -eq 1 ]; then return 0; fi
  read -r -p "$prompt [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

section() { echo ; echo "=== $1 ===" ; }

# --- 0. preflight --------------------------------------------------------------------------
section "0. preflight"
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not installed. https://cli.github.com/"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh not logged in. Run: gh auth login"
  exit 1
fi
echo "OK: gh CLI ready (user: $(gh api user --jq .login))"

# --- 1. Provision repo secrets (gate A') ---------------------------------------------------
# These three secrets unlock the bench-kaggle workflow and the HF Datasets push. The values
# never touch this script's stdout/stderr — the maintainer types them into gh secret set
# stdin. The script just provides the prompts and idempotency guard.
section "1. repo secrets (gate A')"

set_secret_interactive() {
  local secret_name="$1"
  local description="$2"
  local existing
  existing=$(gh secret list --repo "$REPO" --json name --jq ".[] | select(.name==\"$secret_name\") | .name" || true)
  if [ -n "$existing" ]; then
    echo "OK: $secret_name already set; skipping."
    return 0
  fi
  if [ "$CHECK" -eq 1 ]; then
    echo "MISSING: $secret_name — $description"
    return 0
  fi
  if confirm "Set $secret_name now? ($description)"; then
    echo "Paste the value for $secret_name (input hidden via gh secret set stdin):"
    gh secret set "$secret_name" --repo "$REPO"
    echo "OK: $secret_name set."
  else
    echo "SKIP: $secret_name — re-run when ready."
  fi
}

set_secret_interactive HF_TOKEN \
  "Hugging Face write token for the static benchmark dataset push (DoD #5)."
set_secret_interactive KAGGLE_USERNAME \
  "Kaggle username for bench-kaggle workflow."
set_secret_interactive KAGGLE_KEY \
  "Kaggle API key for bench-kaggle workflow."

# --- 2. Flip repo variables (gate A' / gate Phase-3) ---------------------------------------
# Variables (not secrets) toggle which scheduled workflows actually run. Each gate has its
# own variable so flipping one cannot accidentally enable another.
section "2. repo variables"

set_var_interactive() {
  local var_name="$1"
  local intended_value="$2"
  local description="$3"
  local existing
  existing=$(gh variable list --repo "$REPO" --json name,value \
    --jq ".[] | select(.name==\"$var_name\") | .value" 2>/dev/null || true)
  if [ -n "$existing" ]; then
    echo "OK: $var_name=$existing (already set)"
    return 0
  fi
  if [ "$CHECK" -eq 1 ]; then
    echo "MISSING: $var_name — $description"
    return 0
  fi
  if confirm "Set $var_name=$intended_value? ($description)"; then
    gh variable set "$var_name" --body "$intended_value" --repo "$REPO"
    echo "OK: $var_name=$intended_value"
  else
    echo "SKIP: $var_name"
  fi
}

set_var_interactive KAGGLE_ENABLED true \
  "Enable bench-kaggle weekly push (needs HF/Kaggle secrets above and notebooks/kernel-metadata.json)."
set_var_interactive HAVE_MPS_RUNNER false \
  "Set to true after a Mac contributor provisions a self-hosted runner with the 'mps' label."
set_var_interactive LEADERBOARD_ENABLED false \
  "Stays false until Phase 3 (month 7-9). The leaderboard workflow refuses to no-op."

# --- 3. arXiv endorser (gate B) ------------------------------------------------------------
# arxiv.org/cs.LG endorsement is required to submit the v0.1 preprint. The endorser flow is
# entirely off-platform; this section just nags the maintainer to start the conversation.
section "3. arXiv endorser (gate B)"
if [ "$CHECK" -eq 1 ]; then
  echo "REMINDER: confirm a cs.LG endorser before month 6."
else
  if confirm "Open the arXiv endorser request page now?"; then
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "https://arxiv.org/auth/need-endorsement"
    else
      echo "Open: https://arxiv.org/auth/need-endorsement"
    fi
  fi
  echo "OK: tracked in tech_report_v0_1.md §7."
fi

# --- 4. Mac contributor recruitment --------------------------------------------------------
# Issue P1-01 is the open call; this section makes sure it has the right labels and is pinned
# so it shows up on the repo landing page.
section "4. Mac contributor (issue P1-01)"
P1_01=$(gh issue list --repo "$REPO" --state open --search "in:title \"[P1-01]\"" \
  --json number --jq '.[0].number // empty')
if [ -z "$P1_01" ]; then
  echo "WARN: P1-01 not found; re-run scripts/seed-issues.sh first."
else
  pinned=$(gh issue view "$P1_01" --repo "$REPO" --json isPinned --jq '.isPinned' 2>/dev/null || echo false)
  if [ "$pinned" = "true" ]; then
    echo "OK: P1-01 (#$P1_01) already pinned."
  elif [ "$CHECK" -eq 1 ]; then
    echo "PENDING: P1-01 (#$P1_01) is not pinned."
  elif confirm "Pin issue #$P1_01 to the repo landing page?"; then
    gh issue pin "$P1_01" --repo "$REPO" && echo "OK: pinned #$P1_01"
  fi
fi

# --- 5. Discussions / repo polish ----------------------------------------------------------
section "5. repo polish (optional)"
if [ "$CHECK" -eq 0 ] && confirm "Enable GitHub Discussions on the repo (recruits Mac contributors faster)?"; then
  gh api -X PATCH "repos/$REPO" -F has_discussions=true >/dev/null \
    && echo "OK: Discussions enabled."
fi

# --- 6. final status -----------------------------------------------------------------------
section "6. status"
gh api "repos/$REPO/milestones" --jq '.[] | "milestone \(.title): \(.open_issues) open / \(.closed_issues) closed"'
echo
gh run list --repo "$REPO" --limit 3 \
  --json conclusion,workflowName,status,createdAt \
  --jq '.[] | "run: \(.status) \(.conclusion // "running") \(.workflowName)"'
echo
echo "done. Re-run with --check at any time to see what's still pending."
