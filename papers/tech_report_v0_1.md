# nsa-eval v0.1 Technical Report

**Status**: outline. Targeted for GitHub release at the end of month 3.

## 1. Introduction

Native Sparse Attention (NSA), introduced by DeepSeek in [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) and recognised as the ACL 2025 Best Paper, achieves long-context efficiency through three branches (compression, selection, sliding window) combined by a learnable sigmoid gate. The reference implementation is CUDA + Triton, which makes the method inaccessible to two large constituencies: (i) Apple Silicon users who develop and validate models locally on Macs, and (ii) researchers who want a fixed, comparable evaluation harness across sparse-attention papers rather than re-running each paper's bespoke setup. This report describes nsa-eval v0.1, which addresses both gaps with a single repository.

## 2. Related work

- DeepSeek NSA (arXiv 2502.11089)
- DSA in DeepSeek V3.2 (arXiv 2512.02556)
- MorphKV (arXiv 2503.00979, ICML 2025)
- Flash-Sparse-Attention / FSA (arXiv 2508.18224)
- VideoNSA (arXiv 2510.02295)
- H2O (arXiv 2306.14048), SnapKV (arXiv 2404.14469)
- RULER (arXiv 2404.06654), LongBench v2 (arXiv 2412.15204)

## 3. MPS-native NSA reference

### 3.1 Branch implementations

- Compression: strided pooling of K and V across the sequence, dense attention against the compressed KV. Implemented with `torch.stack` over slice means; MPS dispatches to Metal kernels for the dense step.
- Selection: block-wise importance via `einsum`, top-n blocks via `topk`, dense attention on the gathered slice. The sparse gather is the load-bearing op for MPS performance; numbers depend on Mac contributor validation.
- Sliding window: banded causal mask, dense attention on the slice. MPS lacks a fast banded kernel today, so v0.1 uses full mask + slice and accepts the overhead; replaced in Phase 2 if a Metal-native banded kernel ships.

### 3.2 Limitations

- No MPS-fused kernel; every branch is a sequence of PyTorch ops.
- `bfloat16` on M1 falls back to `float16` (M2+ has native `bfloat16`); the manifest records which path ran.
- Numbers in this report are produced on a Linux CPU reference. Mac M1/M2 numbers land via the bench-mps workflow once a contributor provisions a self-hosted runner.

## 4. Unified evaluation suite

### 4.1 Design

`EvalSpec(model, benchmark, attention, device, ...)` → registry lookup → JSON result on disk. Backends and benchmarks are loaded by name, so plugins can register their own without patching the runner.

### 4.2 Coverage in v0.1

| Model         | Benchmark   | Attention                                  | Device        |
|---------------|-------------|--------------------------------------------|---------------|
| Qwen2.5-0.5B  | NIAH        | full / nsa_cuda / nsa_mps / h2o / snapkv / morphkv | cuda / mps / cpu |
| Qwen2.5-1.5B  | RULER 16k   | nsa_cuda / nsa_mps                         | cuda / mps    |
| Llama-3.2-1B  | LongBench   | nsa_cuda / full                            | cuda          |

Matrix size: 4 models × 4 benchmarks × 5 attentions × ~1-3 devices = 80 cells targeted, 64 (80%) required for v0.1.

## 5. Reproducibility

- All runs go through the `BenchmarkRunner`, which writes a manifest with torch / CUDA / MPS / platform / seed.
- A Kaggle notebook (`notebooks/_template_kaggle.ipynb`) reproduces the CUDA-T4 subset end-to-end.
- A Mac notebook (`notebooks/_template_mps.ipynb`) reproduces the MPS subset (Mac contributor required for actual numbers).
- Failed cells are kept under `experiments/_wip/<short-name>/` with a README.

## 6. Discussion

The MPS reference is intentionally written against vanilla PyTorch ops, not a custom Metal kernel. Hand-tuned Metal kernels for sparse attention belong in Phase 3 once the eval suite has stabilised and a Mac contributor has validated the reference numbers. Releasing the eval suite and reference together — rather than waiting for the kernel — lets the research community start comparing against a fixed harness immediately.

## 7. Future work

- Phase 2 (month 4-6): MorphKV decode integration on top of NSA; arxiv preprint submission via endorsement.
- Phase 3 (month 7-9): NeurIPS Datasets & Benchmarks Track submission; HF Spaces leaderboard; Mac contributor regression dashboard.
- Phase 4+: Metal-native fused kernel; CUDA Triton tuning; ROCm port.

## Acknowledgements

DeepSeek for the NSA design. fla-org for the CUDA reference implementation. Future Mac contributors for closing the MPS validation gap.
