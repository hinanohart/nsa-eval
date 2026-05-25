# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3](https://github.com/hinanohart/nsa-eval/compare/v0.0.2...v0.0.3) (2026-05-25)


### Bug Fixes

* register attention backends/benchmarks and fix scheduled CI workflows ([#100](https://github.com/hinanohart/nsa-eval/issues/100)) ([f6ebc95](https://github.com/hinanohart/nsa-eval/commit/f6ebc95c0217432cdd590e3217d32a4ff1926c8d))

## [0.0.2](https://github.com/hinanohart/nsa-eval/compare/v0.0.1...v0.0.2) (2026-05-19)


### Documentation

* add CI + license badges (DEEP-FIX-2) ([#95](https://github.com/hinanohart/nsa-eval/issues/95)) ([4338924](https://github.com/hinanohart/nsa-eval/commit/433892422da2c5df9c84555ef0e59c851c55aacf))
* portfolio cosmetic sweep ([#98](https://github.com/hinanohart/nsa-eval/issues/98)) ([373970a](https://github.com/hinanohart/nsa-eval/commit/373970a55f4207661e960a9d1e803acf374f8539))

## [Unreleased]

### Added

- Initial scaffold: package skeleton, evaluation runner protocol, attention backend protocol, MPS stub, CUDA wrapper around `fla-org/native-sparse-attention`, MorphKV decode adapter, H2O / SnapKV baselines, RULER / LongBench / NIAH / AgentBench benchmark stubs.
- CI: 7 GitHub Actions workflows (ci, bench-dispatch, bench-mps, leaderboard-update, preprint-rebuild, release, nightly-deps).
- Notebook templates for Kaggle (T4) and Apple Silicon (MPS).
- Tech report v0.1 outline (`papers/tech_report_v0_1.md`).
- `experiments/_wip/` archive convention for failed experiments.
