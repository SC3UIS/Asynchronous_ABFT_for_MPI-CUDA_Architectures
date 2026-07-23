# Architecture

This document explains the design of the asynchronous ABFT framework end
to end. It is the conceptual companion to the comment-free source in
[`../src`](../src); the per-file mechanics are in
[source-reference.md](source-reference.md).

## 1. System and fault model

The target is single-precision dense GEMM, `C ← A·B`, on a cluster of MPI
ranks, each bound to one CUDA device. The multiplication is delegated to
cuBLAS and treated as a black box, so the contribution is the
fault-tolerance layer and the protected throughput is always reported
against the strongest possible (vendor) baseline.

Faults in `A`, `B`, or `C` all manifest as a deviation in the output
checksum and are covered by the same mechanism, under the standard ABFT
working assumption of **at most one corrupted element per protected
fragment per verification interval**. The size-aware fragmentation policy
(Section 6) keeps that assumption valid as the problem grows.

The framework targets three properties:

1. provable single-error coverage,
2. *asynchrony* of verification with respect to the GEMM, and
3. no host round-trip on the critical path.

## 2. Distributed data decomposition

MPI ranks are arranged in a near-square 2-D grid `Pr × Pc`, chosen as the
most balanced factorization of the world size. Operand `A` is row-striped
into `Pr` blocks and broadcast along each grid row; operand `B` is
column-striped into `Pc` blocks and broadcast along each grid column.
Rank `(pr, pc)` owns an `M_b × K` stripe of `A` and a `K × N_b` stripe of
`B` and computes the complete output block

```
C_(pr,pc) = A_(pr,·) · B_(·,pc)        C_(pr,pc) ∈ R^(M_b × N_b)
```

Crucially, the contraction dimension `K` is never partitioned, so every
block is a complete inner product and therefore **final**: there is no
cross-rank reduction and no partial-sum all-reduce. The global result is
the disjoint tiling of the `Pr × Pc` blocks. This keeps the
fault-tolerance layer purely intra-rank — each block is self-contained and
can be verified locally — which is what makes the host-free design of
Section 5 possible. The single-GPU configuration is the degenerate case
`Pr = Pc = 1`.

Implemented in [`distribution/grid.cuh`](../src/distribution/grid.cuh).

## 3. The Huang–Abraham checksum

The detector is a variant of the Huang–Abraham checksum encoding,
decomposed so that the parts that depend only on the (fault-free) inputs
are computed once and reused, while only the `C`-dependent part is
recomputed every iteration.

### Detection

Define the column-sum of `A`, `s_A ∈ R^K`, and the expected row-checksum
of `C`, `e ∈ R^(N_b)`:

```
s_A[k]  = Σ_i A[i,k]                              (A only, once per problem)
e[j]    = Σ_k s_A[k]·B[k,j] = Σ_i C[i,j]          (A,B only, once per problem)
a[j]    = Σ_i C[i,j]                              (needs C, every iteration)
detect  : |a[j] - e[j]| > τ   ⇒ fault in column j
```

The right-hand identity `Σ_k s_A[k]·B[k,j] = Σ_i C[i,j]` holds for the
fault-free product. Both `s_A` and `e` depend only on `A` and `B` and are
precomputed once; only `a` is recomputed per iteration, which is the key
to keeping verification cheap.

### Localization and correction

Detection alone yields the corrupted *column*. To recover the *row*, the
symmetric construction with the row-sums of `B`, `r_B[k] = Σ_j B[k,j]`, is
used:

```
ê[i] = Σ_k A[i,k]·r_B[k] = Σ_j C[i,j]   vs.   â[i] = Σ_j C[i,j]
locate : |â[i] - ê[i]| > τ   ⇒ row i*  ⇒ fault at (i*, j*)
```

For a single additive corruption `C[i*,j*] ← C[i*,j*] + δ`, the
row-checksum discrepancy is exactly the error magnitude
`δ = a[j*] - e[j*]`, and the element is restored in place by subtracting
it:

```
correct : C[i*,j*] ← C[i*,j*] - (a[j*] - e[j*])
```

Correction therefore requires no recomputation of the tile, only the
already-available checksum discrepancy.

### Numerical note

All checksum reductions accumulate in `double` even though the operands
are single precision, which lowers the rounding floor of the invariant and
widens the gap between noise and a genuine fault. The threshold `τ` is a
small constant above that floor; the theoretical worst-case bound is
`τ = γ_K · ‖A‖_∞ · ‖B‖_∞` with `γ_K = K·ε / (1 − K·ε)`. In the
performance study a large additive fault drives detection, so the precise
value of `τ` is not a sensitive parameter. The `--calibrate` mode exists
to tune `τ` empirically from the observed noise floor.

Implemented in [`kernels/abft_stepwise.cuh`](../src/kernels/abft_stepwise.cuh)
(math) and [`core/common.cuh`](../src/core/common.cuh) (`compute_threshold_formula`).

## 4. The asynchronous two-stream pipeline

The central idea is to run the GEMM and its verification *concurrently* on
two CUDA streams so the protection cost is hidden behind the dominant
matrix multiplication.

Each rank's `M_b × N_b` block is split into `F` contiguous column
**fragments**. Fragment `f`'s verification can begin as soon as its GEMM
has completed, while fragment `f+1` is still multiplying. Fragments are
distinct from the MPI stripes of Section 2: stripes partition data across
GPUs, fragments pipeline work within a GPU and exist even on a single
device.

- A `compute_stream` issues the per-fragment cuBLAS SGEMM calls.
- A `verify_stream` issues the ABFT kernels.
- Per-fragment CUDA events (`compute_done_f`) gate the verify stream behind
  the exact fragment it depends on, with no host involvement.

To benchmark a representative steady state the loop is repeated `R` times
with the output **double-buffered** (two device `C` buffers alternated by
iteration parity), so iteration `k`'s verification overlaps iteration
`k+1`'s GEMM. A buffer-reuse event prevents iteration `k+2` from
overwriting a buffer before iteration `k`'s verification has consumed it.

The timing scope is the wall-clock interval around the entire `R`-iteration
loop (a single `MPI_Barrier` before, a single stream drain after), so the
reported protected time includes all ABFT work and is directly comparable
to the unprotected baseline measured under the identical scope.

```
verify_stream  : [actR_k | detect][localize_k | gated]  [actR_(k+1) | detect] ...
compute_stream :                                         [SGEMMs of iter k+1 on dC[(k+1)%2]]
```

The one-time input encoding (`s_A`, `e`) is issued on the same
`verify_stream` at the start of the loop. Where it falls relative to the
timed window is a measurement choice (`--encoding-mode`): the reported
experiments **overlap** it with the first iteration's GEMM so its cost is
charged to the run rather than hidden as a pre-timed step. So that this
concurrent encode does not starve as a single thread block against a
full-occupancy GEMM, each of its two reductions is computed in `ENC_CHUNKS`
parallel slices and then combined (the `_part` encode kernels described in
[source-reference.md](source-reference.md)).

Implemented in [`pipeline/passes.cuh`](../src/pipeline/passes.cuh)
(`pass_online_loop`) and [`pipeline/buffers.cuh`](../src/pipeline/buffers.cuh).

## 5. Device-resident verification

A naive two-stream design still copies the checksum vectors to the host
every iteration and synchronizes so the CPU can compare them. That host
round-trip is a PCIe latency floor and a hard pipeline stall, and it
dominates the runtime at small and medium sizes. The framework therefore
makes the **entire** detect–localize–correct path resident on the device.

A single-block reduction *detection kernel* (`k_detect_row`) computes the
maximum `|a[j] - e[j]|` over the fragment, writes the faulted column `j*`
and the signed discrepancy to a device-resident fault record, and folds
the fragment's outcome into a device confusion-matrix counter
(`TP/TN/FP/FN`). The row-sum, expected-column, actual-column, and
locate-and-correct kernels are launched unconditionally but **return
immediately on the device** when the detection record says the fragment is
clean. On the overwhelmingly common fault-free path they cost only
kernel-launch latency; the two expensive pieces of localization work — the
`O(M_b·K)` matrix–vector product `ê = A·r_B` and the `O(M_b·N_frag)`
row-sum reduction `â = rowsum(C_frag)` — execute only when a fault is
actually present.

Inside the timed loop there is **zero** device-to-host transfer and
**zero** stream synchronization; the host reads the fault record exactly
once, after the loop.

Implemented in [`kernels/abft_stepwise.cuh`](../src/kernels/abft_stepwise.cuh)
(the device-gated kernels) and [`metrics/metrics.cuh`](../src/metrics/metrics.cuh)
(the confusion-matrix counters).

## 6. Size-aware fragmentation

Two regimes break a fixed fragment count, and each is handled with a
size-aware policy that leaves the evaluated configurations unchanged by
default.

- **Small problems** (`max(M,K,N) < 1024` in this evaluation): the
  per-fragment GEMM is so cheap that fragmentation just multiplies fixed
  overheads by `F` while the GEMM cannot hide them. `F` is collapsed to
  `1`, so protection coverage is identical (one checksum still spans all of
  `C_local`) and the fixed cost is paid once.
- **Very large stripes**: the single-error-per-fragment assumption weakens
  as the fragment grows, since a large stripe packs enough elements that
  two or more independent upsets become likely. A resilience knob `E`
  (`--frag-cap`) caps elements per fragment, raising `F` to
  `ceil(M_b·N_b / E)` (clamped to `N_b`). It is disabled by default so all
  reported results use the canonical fixed fragmentation.

Implemented in [`src/main.cu`](../src/main.cu) (the `SMALL_MATRIX_THRESHOLD`
and `--frag-cap` logic).

## 7. Fault injection (for the overhead study)

To exercise the protection path on demand, a controlled fault is injected
into the output tile after the GEMM and before verification, on the
compute stream. The fault is a large additive perturbation to a single
random element of the targeted fragment (`--inject add`); this guarantees
detection and drives the full detect–localize–correct path on every
protected run, isolating the correction cost from threshold sensitivity.
It is the natural analogue of the additive fault used by the fused-kernel
reference, keeping the comparison fair. A single-bit-flip mode
(`--inject swifi`) exists for detection-accuracy studies but is outside the
scope of the throughput results.

Implemented in [`kernels/swifi.cuh`](../src/kernels/swifi.cuh).
