#!/usr/bin/env bash
# scripts/seed-issues.sh — Day 0 helper: bulk-create the v0.1 issue board (85 issues).
#
# Idempotent: each issue title is prefixed with [P1-<NN>] or [P2-<NN>] etc.; if an issue with
# the same title already exists, gh issue create will still create a duplicate, so guard with
# `gh issue list --search` before creating.
#
# Usage:
#   bash scripts/seed-issues.sh            # create all phase 1 issues
#   bash scripts/seed-issues.sh --dry-run  # print what would be created
#
# Requires: gh CLI logged in with `repo` scope.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

REPO="${REPO:-hinanohart/nsa-eval}"

# Ensure custom labels exist before creating issues. `gh label create` errors if the label
# already exists, so swallow that case to keep the script idempotent.
for label_pair in \
  "mac-contributor:9F61E2:Apple Silicon hardware contributors" \
  "feature:0E8A16:New functionality" \
  "infra:5319E7:CI / packaging / docs infra" \
  "test:1D76DB:Tests and numeric regressions" \
  "eval:FBCA04:Evaluation matrix cells" \
  "phase-2:D4C5F9:Phase 2 (month 4-6)"; do
  IFS=':' read -r lname lcolor ldesc <<<"$label_pair"
  gh label create "$lname" --repo "$REPO" --color "$lcolor" --description "$ldesc" 2>/dev/null || true
done

create_issue() {
  local title="$1"
  local label="$2"
  local body="$3"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] $title  (label=$label)"
    return
  fi
  if gh issue list --repo "$REPO" --search "in:title \"$title\"" --json number | grep -q '"number"'; then
    echo "[skip] exists: $title"
    return
  fi
  gh issue create --repo "$REPO" --title "$title" --label "$label" --body "$body"
}

# --- Phase 1 (week 1-12, ~85 issues) -----------------------------------------
# week 1 — bootstrap
create_issue "[P1-01] Mac contributor signup — open call"           "mac-contributor" "Looking for Apple Silicon hardware owners to validate the MPS path. See ISSUE_TEMPLATE/mac_contributor.md."
create_issue "[P1-02] Vendor fla-org/native-sparse-attention"       "feature"          "git submodule add the fla-org reference impl under third_party/ and wire nsa_wrapper.py to it."
create_issue "[P1-03] HF adapter — attention dispatch patching"     "feature"          "Implement src/nsa_eval/models/hf_adapter.py:load_hf to register the configured backend."
create_issue "[P1-04] NIAH dataset loader (smoke benchmark)"         "feature"          "Implement RULER NIAH variant as the v0.1 smoke benchmark."
create_issue "[P1-05] RULER dataset loader (4k-32k)"                 "feature"          ""
create_issue "[P1-06] LongBench v2 en-subset loader"                 "feature"          ""
create_issue "[P1-07] AgentBench OS/DB/KG subset loader"             "feature"          ""
create_issue "[P1-08] CI: matrix runs on py3.10/3.11/3.12 (CPU)"    "infra"            ""
create_issue "[P1-09] Numeric regression test vs lucidrains NSA"     "test"             ""
create_issue "[P1-10] Static benchmark dataset — HF Datasets push"   "feature"          "DoD criterion #5. Needs HF_TOKEN provisioned via gate A'."

# week 2-4 — backends
for i in 11 12 13 14 15 16 17 18 19 20; do
  create_issue "[P1-$i] Backend integration ticket #$i" "feature" "Placeholder — replace title and body during week 2-4 backend wiring."
done

# week 5-8 — eval matrix
for i in $(seq 21 50); do
  create_issue "[P1-$i] Eval matrix cell #$((i - 20)) of 64" "eval" "DoD criterion #1 — fill the 80-cell matrix to ≥64 cells (80%)."
done

# week 9-12 — packaging, docs, paper
for i in $(seq 51 75); do
  create_issue "[P1-$i] Phase 1 close-out ticket #$((i - 50))" "infra" "Packaging, docs, paper, Kaggle notebook polish."
done

# Phase 2 placeholders (issues created now so the board reflects the published plan)
for i in $(seq 76 85); do
  create_issue "[P2-$((i - 75))] Phase 2 placeholder #$((i - 75))" "phase-2" "MorphKV integration, arxiv preprint, NeurIPS D&B Track submission preparation."
done

echo "done."
