# Security Policy

## Reporting a Vulnerability

If you discover a security issue in `nsa-eval`, please **do not open a public
issue**. Use the private channel:

- **GitHub Security Advisories**: <https://github.com/hinanohart/nsa-eval/security/advisories/new>

(GitHub will forward the advisory to the maintainer and keeps the report
private until a fix and CVE — if applicable — are coordinated.)

We aim to acknowledge new reports within 7 days and to publish a fixed
release plus advisory within 30 days of confirmation, whichever class of
vulnerability allows. For research-time-only or evaluation-only impact
(e.g. only affects benchmark output, not user data), the timeline is
relaxed and discussed on the advisory thread.

## Scope

In scope:

- The `nsa_eval` Python package (`src/nsa_eval/`).
- CI/CD workflows under `.github/workflows/` when they grant capability
  beyond what an unauthenticated viewer should have.
- Supply-chain concerns introduced by our pinned dependencies or by the
  vendored `third_party/native-sparse-attention` submodule.

Out of scope:

- Issues only reproducible against a fork of this repo.
- DoS via large model downloads or unbounded benchmark configs (these are
  evaluation tools meant to be invoked at user discretion).
- Vulnerabilities in upstream PyTorch / Hugging Face / fla-org code — please
  report those to the respective upstream projects (we will mirror the
  advisory once they publish a fix).

## Supported Versions

Only the most recent `0.0.x` line is supported during pre-release. Once
`0.1.0` ships we will document an LTS policy here.
