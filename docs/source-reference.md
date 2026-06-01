# Source reference

A file-by-file guide to [`../src`](../src). The source is header-only
(`.cuh`) plus a single translation unit (`main.cu`); everything is compiled
into one binary, `abft_gemm`. The conceptual background for the algorithms
referenced here is in [architecture.md](architecture.md).

```
src/
├── main.cu                     orchestrator: argv → grid → buffers → trial loop → metrics
├── core/
│   ├── common.cuh              includes, error-check macros, math + timing utilities
│   ├── types.cuh               plain-data structs (Grid2D, ExperimentConfig, results)
│   └── cli.cuh                 --help text and the argv parser
├── distribution/
│   └── grid.cuh                2-D process grid + A/B scatter + row/col broadcast
├── kernels/
│   ├── gemm_cublas.cuh         cuBLAS SGEMM wrapper + warm-up
│   ├── abft_stepwise.cuh       all ABFT kernels + per-stage launchers
│   └── swifi.cuh               fault injection (single bit-flip and additive)
├── metrics/
│   └── metrics.cuh             confusion matrix, derived metrics, MPI reduction, CSV
└── pipeline/
    ├── buffers.cuh             PipelineBuffers: streams, events, device scratch
    └── passes.cuh              pass_baseline / pass_calibrate / pass_online_loop
```

---

## `main.cu` — orchestrator

The single translation unit. Its `main()` performs, in order:

1. **MPI + grid setup.** Reads `world_rank`/`world_size`, picks (or accepts
   via `--grid`) a near-square `Pr × Pc`, and builds the row/column
   sub-communicators used for the A/B broadcasts.
2. **Device binding.** Maps the rank to a local GPU via `SLURM_LOCALID` or
   `OMPI_COMM_WORLD_LOCAL_RANK`.
3. **Block dimensions.** Splits `M` across `Pr` and `N` across `Pc` to get
   the per-rank `M_b`, `N_b`, and the fragment count `F`.
4. **Size-aware `F`.** Collapses `F = 1` for small problems
   (`max(M,K,N) < 1024`); optionally raises `F` when `--frag-cap` is set
   (see [architecture.md](architecture.md) §6).
5. **Buffer allocation.** `dA`, `dB`, and **two** `dC` buffers for double
   buffering; plus the `PipelineBuffers` scratch.
6. **Input generation.** `regen_inputs()` fills `A,B` from a seed,
   scatters them across ranks, uploads to the device, and recomputes the
   per-fragment thresholds from the new `B`. Called once in fixed-seed
   mode, or once per trial under `--reseed-per-trial`.
7. **Trial loop.** Each trial times a `repeats`-iteration baseline pass and
   (unless `--baseline`) a `repeats`-iteration protected pass, both under
   the identical whole-loop timing scope. The first `--warmups` trials are
   discarded so cuBLAS picks its heuristic algorithm and the GPU clocks
   settle at the real problem shape.
8. **Aggregation + emit.** Reduces timings (max over ranks) and the
   confusion matrix (sum over ranks) onto rank 0, which prints a summary
   and appends one row to the metrics CSV.

Only two modes exist: unprotected `--baseline`, and online ABFT (default).
A `--calibrate` mode runs a separate noise-collection pass instead of the
trial loop.

---

## `core/common.cuh`

Common includes plus the building blocks used everywhere:

- **Error-check macros** `CUDA_CHECK`, `MPI_CHECK`, `CUBLAS_CHECK` — each
  reports file/line and calls `MPI_Abort` on failure.
- **`fill_random`** — fills a vector with values in `[-1, 1]` from a seeded
  Mersenne-Twister.
- **`matrix_inf_norm`** — `‖·‖_∞` of a row-major matrix (max absolute row
  sum), used in the threshold formula.
- **`compute_threshold_formula`** — the theoretical worst-case rounding
  bound `τ = γ_K · ‖A‖_∞ · ‖B‖_∞`, `γ_K = K·ε/(1 − K·ε)`. In practice this
  can be far above the natural noise floor; `--calibrate` yields an
  empirically tuned threshold instead.
- **`TimingStats` + `stats_of`** — min/median/**mean**/max over a vector of
  per-trial times. The mean is the figure compared against the reference
  (which reports a mean), so timings stay apples-to-apples.

## `core/types.cuh`

Plain-data structs, no logic:

- **`Grid2D`** — the process grid (`Pr,Pc,pr,pc`) and the two MPI
  sub-communicators (`row_comm` shares an A-stripe, `col_comm` shares a
  B-stripe).
- **`ExperimentConfig`** — every CLI-settable knob with its default: matrix
  dims, grid, `frags_per_rank`, `repeats`, `warmups`, `frag_cap` (the
  resilience knob), `num_samples`, `reseed_per_trial`, injection mode and
  zone, threshold override, calibration safety factor, RNG seeds, and
  output paths.
- **`ABFTResult` / `InjectionInfo`** — bookkeeping records for a single
  verification and a single injected fault.

## `core/cli.cuh`

`print_help()` (the full flag reference, also reproduced in the top-level
[README](../README.md)) and `parse_args()`, which walks `argv`, fills an
`ExperimentConfig`, accepts the positional `N` (square) or `M K N`
(rectangular) form, and clamps out-of-range values.

---

## `distribution/grid.cuh`

The 2-D decomposition (see [architecture.md](architecture.md) §2):

- **`choose_grid`** — near-square factorization `Pr·Pc = world`.
- **`split_dim`** — even/uneven block split of a dimension into `parts`
  contiguous `(count, offset)` ranges.
- **`distribute_A`** — rank 0 ships each `M_b × K` row-stripe to the
  `(pr, 0)` rank, then each row-communicator broadcasts it; every rank in a
  grid row ends with the same A-stripe.
- **`distribute_B`** — rank 0 packs each `K × N_b` column-block into a
  contiguous buffer (column blocks are strided in the full row-major `B`),
  ships it to `(0, pc)`, then each column-communicator broadcasts it.

`K` is never split, so no rank ever holds a partial product.

---

## `kernels/gemm_cublas.cuh`

- **`gemm_cublas`** — the SGEMM wrapper. The matrices are row-major but
  cuBLAS is column-major, so the call computes (column-major)
  `C^T(N×M) = B^T(N×K)·A^T(K×M)`, which is the same memory as (row-major)
  `C(M×N) = A(M×K)·B(K×N)` — "swap A/B, swap m/n", no transposes, no
  copies. Delegating to cuBLAS removes the GEMM implementation as a
  confounding variable in the ABFT study.
- **`gemm_warmup`** — forces cuBLAS module load, JIT, and heuristic-cache
  population *before* any timed measurement by submitting a few throwaway
  64×64 SGEMMs and synchronizing. The real warm-up still comes from the
  discarded trials, but this removes the first-call JIT outlier.

## `kernels/abft_stepwise.cuh`

All ABFT math, as small CUDA kernels plus thin per-stage launch helpers.
The clean-path / fault-path split is what makes the verification cheap.

Per-fragment data flow (one rank, one `C`-fragment of `M_b × N_frag`):

```
colSumA[k]     = Σ_i A_stripe[i,k]                   one-shot, A only
expectedRow[j] = Σ_k colSumA[k]·B_frag[k,j]          A,B only
actualRow[j]   = Σ_i C_frag[i,j]                      needs C, every iter
detect         : |actualRow[j] − expectedRow[j]| > τ
```

On detection (rare path):

```
rowSumB[k]     = Σ_j B_frag[k,j]
expectedCol[i] = Σ_k A_stripe[i,k]·rowSumB[k]
actualCol[i]   = Σ_j C_frag[i,j]
locate row i*, then correct C_frag[i*,j*] in place
```

Key kernels:

- **`k_col_checksum_A` / `k_expected_row`** — the precomputed, input-only
  quantities.
- **`k_actual_row`** — the only reduction on the per-iteration critical
  path.
- **`k_detect_row`** — single-block reduction that finds the worst
  offending column, writes `{err_col, row_diff}` for the localization
  stage, and folds the fragment's `TP/TN/FP/FN` outcome into the device
  confusion matrix.
- **The `_g` (gated) kernels and `k_locate_correct`** — the localization
  and correction work. Each is device-gated on the detection record, so it
  is a launch-latency no-op on the clean path and only does real work
  (`O(M_b·K)` / `O(M_b·N_frag)`) when a fault is present.
- **`launch_*` helpers** — wrap grid/block configuration and stream
  placement for each stage so `passes.cuh` reads as a sequence of named
  steps.

## `kernels/swifi.cuh`

Software-implemented fault injection (see [architecture.md](architecture.md)
§7):

- **`inject_add_constant`** (`--inject add`) — a large additive
  perturbation to one random element, always above `τ`. Guarantees
  detection and exercises the full correction path; used for the overhead
  study and to mirror the reference's additive fault.
- **`inject_single_bitflip`** (`--inject swifi`) — flips one random bit of
  one element, optionally restricted to an IEEE-754 region
  (`--swifi-zone any|sign|exponent|sig_high|sig_low`) via
  `swifi_zone_range`. Used for detection-accuracy characterization, not for
  the throughput numbers.

## `metrics/metrics.cuh`

- **`ConfusionMatrix`** (`TP/TN/FP/FN`) and **`ExperimentMetrics`** (the
  full per-run record: matrices, timings, dims, grid, threshold, and
  calibration fields).
- **MPI reducers** — `mpi_reduce_cm` (sum), `mpi_reduce_int_sum`,
  `mpi_reduce_double_max`, `mpi_reduce_timing_max` (worst-rank time, the
  conservative choice for a distributed run).
- **Derived metrics** — `compute_recall`, `compute_precision`,
  `compute_correction_precision`, `compute_runtime_overhead`
  (`100·(t_protected − t_baseline)/t_baseline`), and `compute_gflops`
  (`2·M·N·K / (t·1e6)`, the whole-problem worst-rank throughput, matching
  the reference's formula).
- **`finalize_metrics` / `print_metrics` / `log_metrics_csv`** — assemble
  the derived fields, print a human-readable summary on rank 0, and append
  one row to the metrics CSV (the column list is documented in
  [data.md](data.md)).

## `pipeline/buffers.cuh`

- **`PipelineBuffers`** — every persistent handle, allocated once for the
  whole run: the cuBLAS handle, the `compute_stream` and `verify_stream`,
  the per-fragment and buffer-reuse CUDA events, the per-fragment
  `col_counts`/`col_offsets`, and all device scratch for the checksums and
  the fault record. Because detect/localize/correct are fully
  device-resident and the verify stream is in-order, the localization
  scratch is single-instance: fragment `k`'s locate+correct completes
  before fragment `k+1` starts.
- **`buffers_init` / `buffers_free`** — allocate and release the above.

## `pipeline/passes.cuh`

The three timed passes, all sharing one whole-loop timing scope so their
times are directly comparable:

- **`pass_baseline`** — `F` cuBLAS calls per iteration, no verification.
  The overhead reference.
- **`pass_calibrate`** — clean GEMM plus the row checksums but no detection;
  records every `|actualRow − expectedRow|` so `--calibrate` can report the
  observed noise floor and a suggested `τ`.
- **`pass_online_loop`** — the pipelined online ABFT (see
  [architecture.md](architecture.md) §4–5). Precomputes `colSumA` and the
  per-fragment `expectedRow` once; then, per iteration, per fragment, on
  the verify stream gated behind `compute_done`, runs `actualRow →
  k_detect_row` and (when injection is enabled) the device-gated
  localize+correct chain. Two `dC` buffers alternate by parity; iteration
  `k+2` waits on `buf_verify_done[k%2]` before overwriting that buffer.
  Inside the timed loop there is zero host round-trip; the host reads the
  confusion-matrix counters and the restored-fault count once, after the
  loop.
