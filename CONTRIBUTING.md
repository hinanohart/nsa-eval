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

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) so that `release-please` can compute the next version and generate `CHANGELOG.md` automatically. The first line must follow:

```
<type>(<optional scope>): <short summary>
```

Allowed `<type>` values:

| type       | when to use                                                          | bumps |
|------------|----------------------------------------------------------------------|-------|
| `feat`     | user-visible new capability                                          | minor |
| `fix`      | user-visible bug fix                                                 | patch |
| `perf`     | performance improvement with no behavioural change                   | patch |
| `deps`     | dependency upgrade                                                   | patch |
| `revert`   | revert of a prior commit                                             | patch |
| `docs`     | README, papers, CHANGELOG-only changes                               | none  |
| `refactor` | code restructuring with no behavioural change                        | none  |
| `test`     | tests-only changes                                                   | none  |
| `build`    | build system / packaging                                             | none  |
| `ci`       | CI workflow changes                                                  | none  |
| `chore`    | repo housekeeping that doesn't fit elsewhere                         | none  |

Breaking changes (any version pre-1.0.0): append `!` after the type/scope, e.g. `feat(api)!: drop is_causal kwarg`. After `1.0.0`, breaking changes bump major.

Squash-merge into `main`. The squash commit subject is what `release-please` parses, so write it carefully — do not rely on the body.
