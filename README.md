# nsa-eval

Apple Silicon (MPS) native Native Sparse Attention (NSA) reference implementation + unified long-context evaluation suite.

A vanilla-PyTorch NSA reference written to dispatch onto Metal kernels on Apple Silicon, alongside a cross-device (CUDA / MPS / CPU) benchmark runner with reproducible Kaggle notebooks. To the maintainer's knowledge there is no other OSS NSA implementation that targets MPS as a first-class device — please open an issue if you find a counterexample. The project is run with minimum human intervention through v0.1: a weekly Claude Code Routine drives the issue board, and a Mac contributor is needed only for MPS validation.

## Why

- DeepSeek NSA ([arXiv 2502.11089](https://arxiv.org/abs/2502.11089), ACL 2025 Best Paper) ships as a CUDA/Triton implementation. There is no published path for Apple Silicon users to run it locally.
- Sparse attention research papers use inconsistent benchmark setups, which makes head-to-head comparison and reproduction hard.
- This repo provides three things:
  1. An MPS-native NSA reference implementation, with a Linux/CUDA path that mirrors it cell-for-cell.
  2. A single runner for RULER / LongBench / Needle-in-a-Haystack / AgentBench against a fixed set of small open LLMs.
  3. Kaggle notebooks (T4 / P100 / TPU v3-8) and Apple Silicon scripts that reproduce every reported number end-to-end.

## v0.1 (3-month) acceptance criteria

| # | criterion |
|---|---|
| 1 | Evaluation matrix: 64 of 80 cells (80%) completed |
| 2 | MPS NSA reference implementation runs Qwen2.5-1.5B + RULER 16k end-to-end on Mac M1/M2 (needs a Mac contributor) |
| 3 | CUDA NSA path reproduces the same numbers on Kaggle T4 |
| 4 | Tech report v0.1 published as a GitHub release (Markdown) |
| 5 | Static benchmark dataset + reference script published on Hugging Face Datasets |
| 6 | Kaggle notebook public and reproducible end-to-end |
| 7 | README documents the MPS path honestly: "minimum human intervention through v0.1" + maintainer-known absence of another MPS NSA OSS (counterexamples invited) |
| 8 | `experiments/_wip/` contains the failed cells, not deleted |

## Quick start (Kaggle T4)

Open `notebooks/_template_kaggle.ipynb` on Kaggle and Run All. The notebook self-installs everything, downloads the configured model, and writes results to `benchmarks/results/<date>/`.

## Quick start (Mac, Apple Silicon)

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
nsa-eval run --attention nsa_mps --model qwen25_0_5b --benchmark niah --device mps
```

## Mac contributor wanted

The maintainer's development machine is WSL2 Linux x86_64, so MPS execution cannot be validated locally. The MPS backend ships as a complete interface + Linux-runnable stub, and we need contributors with Apple Silicon hardware (M1 / M2 / M3) to validate, profile, and report regressions. See issue #1.

## Layout

```
src/nsa_eval/
  attention/    # protocol, NSA wrapper, MPS backend stub, MorphKV decode, baselines
  eval/         # eval spec, runner, benchmarks (RULER / LongBench / NIAH / AgentBench)
  models/       # Qwen2.5, Llama-3.2, HF adapter
  config/       # pydantic schema
  utils/        # seed, device select, run manifest
  cli.py
configs/        # YAML configs: model / benchmark / attention / device
notebooks/      # Kaggle + MPS templates
papers/         # tech report v0.1 source
experiments/_wip/   # failed-experiment archive (rm-forbidden)
.github/workflows/  # ci / bench-dispatch / bench-mps / leaderboard-update / preprint-rebuild / release / nightly-deps
```

## License

MIT.
