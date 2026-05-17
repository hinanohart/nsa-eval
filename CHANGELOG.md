# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial scaffold: package skeleton, evaluation runner protocol, attention backend protocol, MPS stub, CUDA wrapper around `fla-org/native-sparse-attention`, MorphKV decode adapter, H2O / SnapKV baselines, RULER / LongBench / NIAH / AgentBench benchmark stubs.
- CI: 7 GitHub Actions workflows (ci, bench-dispatch, bench-mps, leaderboard-update, preprint-rebuild, release, nightly-deps).
- Notebook templates for Kaggle (T4) and Apple Silicon (MPS).
- Tech report v0.1 outline (`papers/tech_report_v0_1.md`).
- `experiments/_wip/` archive convention for failed experiments.
