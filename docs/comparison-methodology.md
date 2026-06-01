# Comparison methodology

The paper compares this framework against the fused-kernel ABFT reference
of Wu et al., *Anatomy of High-Performance GEMM with Online Fault Tolerance
on GPUs* (ICS 2023), whose source is public at
<https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs>. This
document records how the head-to-head is set up so the measurement isolates
the fault-tolerance layer rather than incidental harness differences. The
sweeps are driven by
[`scripts/run_comparison_felix.sbatch`](../scripts/run_comparison_felix.sbatch)
and
[`scripts/run_comparison_pacca.sbatch`](../scripts/run_comparison_pacca.sbatch);
the resulting CSVs are described in [data.md](data.md).

## What is measured

Each panel of the comparison reports, per shape, six curves:

| `who` | What it is |
| ----- | ---------- |
| `ours_baseline` | our framework's cuBLAS GEMM with verification disabled |
| `ours_online` | our online ABFT, no active fault (detection only) |
| `ours_online_inj` | our online ABFT with an always-on additive fault |
| `theirs_cublas` | the reference's bare `cublasSgemm`, timed in isolation |
| `theirs_fused` | the reference's fused FT-SGEMM kernel, no active fault |
| `theirs_fused_inj` | the reference's fused FT-SGEMM with an always-on fault |

## Alignment choices

Three things are deliberately matched so the comparison is fair:

1. **Same iteration count and warm-up.** Both sides time the same number of
   iterations per shape after a matching warm-up window. The reference
   driver originally omitted any warm-up; the comparison scripts inject one
   so both sides start at boosted GPU clocks.
2. **Same shapes.** Both consume the same `(M, K, N)`. The reference driver
   is square-only, so it is patched to read `FT_M` / `FT_N` / `FT_K`
   environment variables and feed non-square dimensions to its kernels.
3. **Same timing protocol.** Fixed random operand pair, steady-state clocks,
   arithmetic-mean throughput (GFLOPS), with time derived as
   `t = 2·M·N·K / GFLOPS`. This matches the protocol our own binary uses
   (see [architecture.md](architecture.md) §4).

The fault model is matched as a single large additive corruption to one
element of the output tile, on both sides.

## Per-shape kernel selection

The reference is not a single kernel but a family selected by input shape.
Its driver registers the following (index → kernel), where the `abft_*`
entries are the fused fault-tolerant kernels:

| idx | kernel | role |
| --- | ------ | ---- |
| 0 | `cublas` | reference cuBLAS (our `theirs_cublas`) |
| 1–6 | `kernel_sgemm_{small,medium,large,tall,wide,huge}` | their hand-tuned plain SGEMM |
| 7 | `abft_baseline` | their non-fused ABFT baseline |
| 8 | `abft_kernel_small` | fused FT-SGEMM, small tiles |
| 9 | `abft_kernel_medium` | fused FT-SGEMM, medium tiles |
| 10 | `abft_kernel_large` | fused FT-SGEMM, large tiles |
| 11 | `abft_kernel_tall` | fused FT-SGEMM, tall-and-skinny (fixed-K elongated) |
| 12 | `abft_kernel_wide` | fused FT-SGEMM, wide |
| 13 | `abft_kernel_huge` | fused FT-SGEMM, 128×128 tiles |

For a defensible head-to-head, each shape is dispatched to the
shape-appropriate fused kernel rather than a single fixed variant, mirroring
the reference's own codegen selection criteria (Wu et al., Table 1):

| Input shape | Their best fused kernel |
| ----------- | ----------------------- |
| `max(M,N,K) ≤ 128` (tiny square) | `abft_kernel_small` |
| `max(M,N,K) ≤ 256` (medium square) | `abft_kernel_medium` |
| `max(M,N,K) ≤ 512` (large square) | `abft_kernel_large` |
| non-square with fixed `K` (`small_ns`, `big_ns`) | `abft_kernel_tall` |
| square, `max > 512` (`big_sq`) | `abft_kernel_huge` |

Selecting per shape this way lets the reference run at the variant its
authors intended for that shape, so the reported gap reflects the
asynchronous-overlap contribution rather than a mismatched kernel choice.

## Reproducing

The comparison scripts clone the reference on demand, provision its
`cuda-samples` headers, apply the two minimal patches above (non-square
dimension override and warm-up window), build it with and without an
always-on fault, build our `abft_gemm`, sweep the four shape regimes, and
emit `compare_all.csv`. See the script headers for the full knob list and
the top-level [README](../README.md) for invocation.
