# Contributing to nsa-eval

Thanks for your interest. This project ships as a reference implementation, so PRs need to keep the existing reference numbers reproducible.

## Issue triage

- `bug`: include a reproducer (config snippet + command line), the device, and the relevant log. If the failure is on Mac/MPS, attach `system_profiler SPHardwareDataType | grep "Chip"` so we can attribute it to a specific Apple Silicon generation.
- `feature`: scope must be NSA, sparse attention, or the evaluation harness. New benchmark types are welcome but must come with a dataset license check and a reference number.
- `mac-contributor`: open one of these if you have access to Apple Silicon hardware and can validate MPS runs. We track Mac coverage explicitly because the maintainer cannot.

## PR checklist

- Branch from `main`. One PR, one topic.
- `pytest` and `ruff check` must be green locally. The CI workflow re-runs them.
- If you touch `src/nsa_eval/attention/`, add or update a numeric regression check in `tests/test_attention/`.
- If you touch a benchmark adapter, attach the run manifest (`benchmarks/results/<date>/<model>_<bench>_<device>.json`) to the PR description.
- Failed experiments are not deleted. Move them to `experiments/_wip/<short-name>/` with a `README.md` explaining what failed and why. This is intentional - kept on purpose to avoid retrying dead-end approaches.

## Dependency changes

If you edit `pyproject.toml` dependencies, call this out in the PR description with the upstream version, the reason, and the security review (license + maintenance + last-release recency).

## Style

`ruff` settings live in `pyproject.toml`. No bikeshedding in PR review - run `ruff format` before pushing.
