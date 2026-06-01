# Scripts reference

The scripts in [`../scripts`](../scripts) are kept **comment-free**, like the
source: every explanation that lived in a script comment is collected here.
There are two kinds of file — four experiment drivers (Slurm batch scripts)
and three plotters (Python) — listed below with the rationale behind the
choices that matter for reproducing the paper.

```
scripts/
├── run_abft_felix.sbatch        dual-GPU overhead campaign (Felix, 2× Titan X)
├── run_comparison_felix.sbatch  head-to-head vs the fused reference (Felix, 1× Titan X)
├── run_comparison_pacca.sbatch  head-to-head vs the fused reference (PACCA, 1× A100)
├── run_profile_pacca.sbatch     Nsight profiling: timeline + real GFLOPS (PACCA, A100)
├── plot_metrics.py              renders the overhead "vs size" figures
├── plot_comparison.py           renders the per-regime comparison figures
└── plot_real_gflops.py          renders the measured-GFLOPS figures
```

The `#SBATCH` directives, node names (`felix`, `paccaA100`), and
`module load` lines are specific to the SC3UIS clusters; adapt them to
another site. The experiment logic is portable. Each driver builds
`abft_gemm` from [`../src/main.cu`](../src/main.cu), runs its sweep, writes a
CSV (or profiling report) into [`../data`](../data), and calls a plotter.

---

## Timing methodology (shared by all drivers)

This is the single most important thing to preserve when adapting the
scripts:

- **No reseeding during a performance measurement.** Dense-GEMM timing is
  operand-*value* independent — the same `(M,N,K)` does the same FLOPs and
  the same memory access pattern regardless of the values in `A` and `B`.
  The perf drivers therefore fix one random `(A,B)`, discard a few warm-up
  trials to bring the GPU to boosted clocks, then time many back-to-back
  GEMMs at steady clocks. This mirrors the reference harness exactly
  (`srand` once, then N back-to-back iterations).
- **Do not add `--reseed-per-trial` to a perf run.** Regenerating `A,B`
  re-uploads them over PCIe before every sample; that idle gap lets the GPU
  clocks ramp *down*, so each timed GEMM runs cold and even the plain cuBLAS
  baseline looks slower than it is. Reseeding belongs only in the
  detection-accuracy (CM) campaign, where operand diversity genuinely
  changes the checksum noise floor and a fault's detectability.
- **Mean, not median.** Throughput is reported as the arithmetic mean over
  the timed trials, to match the reference, whose
  `GFLOPS = 2·num_tests·MNK / total_time` is intrinsically a mean over its
  timed iterations.

---

## `run_abft_felix.sbatch` — dual-GPU overhead campaign

Runs on both Titan X GPUs of Felix (`Pr=1, Pc=2`). Two independent phases,
written to separate CSVs so the plotter renders parallel figure sets:

1. **Perf sweep (square and non-square).** 40 sizes, 256-aligned, from 256
   to 10240. Per size, three fast phases — baseline, ABFT no-fault, ABFT
   with an always-on additive fault — timed back-to-back on fixed `A,B`. No
   SWIFI, no calibration; `τ = 1.0` since accuracy is not measured here.
   This drives the overhead "vs size" figures. The non-square sweep fixes
   `K = 1024` (mirroring the comparison job's `big_ns` regime) and sweeps
   `M = N`, so its figure is directly comparable to the square one. →
   [`data/campaign_felix/abft_metrics.csv`](../data/campaign_felix/abft_metrics.csv)
2. **Detection-accuracy (CM) campaign.** A few sizes × every IEEE-754
   bit-region × many samples, with `--reseed-per-trial` so the confusion
   matrix aggregates over many operand contexts. Each SWIFI run does its own
   clean baseline pass per trial to capture the golden `C`. The detection
   threshold is chosen per size by a `--calibrate` pass (the default;
   override with `TAU=<number>` for a manual value or `TAU=formula` for the
   analytical bound). This phase feeds the detection study rather than the
   paper's throughput figures, so its calibration / per-zone-recall CSVs are
   **not shipped** in this release — only the overhead-sweep
   `abft_metrics.csv` above is.

Key knobs are environment-overridable (see the script body): `F`, `R`,
`WARMUPS`, the sweep size lists, the CM size list, and `TAU`. Plotter:
`plot_metrics.py`.

## `run_comparison_felix.sbatch` and `run_comparison_pacca.sbatch` — head-to-head

Same logic on two platforms (Maxwell Titan X / Ampere A100; the PACCA
variant adds defensive module loading for the OpenHPC/Lmod toolchain). Each
sweeps four shape regimes — `small_sq`, `big_sq`, `small_ns`, `big_ns` —
and emits six series per shape into
[`compare_all.csv`](../data/comparison_felix/compare_all.csv):
`ours_baseline`, `ours_online`, `ours_online_inj`, `theirs_cublas`,
`theirs_fused`, `theirs_fused_inj`.

What the script does to the reference to make the comparison fair (full
rationale in [comparison-methodology.md](comparison-methodology.md)):

- **Provision + arch patch.** Clones the reference on demand, supplies its
  `cuda-samples/Common` headers, and patches its `CMakeLists.txt` to the
  target compute capability (sm_52 on Felix, sm_80 on PACCA).
- **Non-square override.** The reference's perf loop is square-only
  (`N = K = M = max_size`); a small idempotent patch makes it read
  `FT_M`/`FT_N`/`FT_K` so it can run the non-square regimes.
- **Warm-up injection.** The reference times its iterations with no warm-up
  (cold clocks — the source of jittery/negative apparent overhead). The
  script injects `FT_WARMUP` discarded iterations and re-records the start
  timestamp after them; the GFLOPS formula is unchanged because it still
  divides by the count of *timed* iterations.
- **Two builds for the fault model.** The reference's fault injection is
  hard-coded in every `ft_sgemm_*.cuh` and cannot be toggled at runtime, so
  the kernels are built **twice** — injection on, and `error_inject` forced
  to `0.0` — to compare their ABFT with and without a fault (the analogue of
  our `--inject add` vs `none`).
- **Matched iteration count.** Their binary times `num_tests = 5` iterations
  per process launch (hard-coded), so to match our `OUR_SAMPLES` timed GEMMs
  the script launches theirs `OUR_SAMPLES / 5` times. Keep the invariant
  `THEIRS_SAMPLES × 5 == OUR_SAMPLES` if you retune.

Shape grids: 20 small sizes in `(0, 1024]` (16-aligned) and 40 big sizes
(128/256-aligned, so the reference's 128×128 "huge" kernel has no
partial-tile). Env-overridable (`REGIMES`, `SMALL_N`, `BIG_N`,
`EXPLICIT_SIZES`, `OUT_TAG`) for quick smoke runs and for replicating
specific reference figures. Plotter: `plot_comparison.py`.

### PACCA toolchain notes

The A100 driver carries extra logic because CUDA on PACCA ships *inside* the
NVIDIA HPC SDK (`nvhpc/23.1`), not as a standalone module:

- `nvcc` is not on `PATH` by default (only `nvc++` is); the script locates it
  under the SDK's `cuda/bin`. Do **not** `module purge` — that drops the
  default compilers.
- The SDK hijacks `mpicc`, so `mpicc --showme` reports the SDK's bundled
  OpenMPI (HPCX). The binary links that MPI, so `OPAL_PREFIX`, `mpirun`, and
  `LD_LIBRARY_PATH` must all point at the SDK MPI (the parent of the
  `--showme` libdir), not at `which mpicc`'s prefix — mismatching them is
  what caused the earlier `opal_shmem_base_open` / `-np` failures.
- `libcudart` and `libcublas` live in *separate* deep SDK subtrees, so the
  script `find`s each rather than guessing the path.

## `run_profile_pacca.sbatch` — Nsight profiling

Runs two complementary profilers on the A100, one representative shape per
regime, profiling both `ours_online` and `theirs_fused`:

- **Nsight Compute (`ncu`)** → *measured* FP32 throughput, the authoritative
  real-GFLOPS source (versus the theoretical `2·M·N·K / walltime`). Real
  FLOPs are counted as `fadd + fmul + 2·ffma`. It also drives a modest
  square-size sweep (`--launch-count` capped, since ncu replays each kernel)
  for the measured GFLOPS-vs-size figure. →
  [`data/profile_pacca/*_metrics.csv`](../data/profile_pacca/), `real_gflops_sweep.csv`
- **Nsight Systems (`nsys`)** → the **timeline only**, for visualizing the
  two-stream pipeline (`compute_stream` GEMMs overlapping `verify_stream`
  ABFT kernels). Open the `.nsys-rep` in `nsys-ui`. `nsys` is never used as
  a GFLOPS source. → [`data/profile_pacca/*.nsys-rep`](../data/profile_pacca/)

Because PACCA's OpenMPI singleton fails inside a Slurm allocation, our run is
launched as `mpirun -np 1`; `ncu --target-processes all` and `nsys`
descendant-tracing capture the `abft_gemm` child. Point `NCU_DIR` / `NSYS_DIR`
at your Nsight installs (or leave empty to use whatever is already on
`PATH`). Plotter: `plot_real_gflops.py`.

---

## Plotters

All three read the raw CSVs from [`../data`](../data) and write image files;
no GPU is needed to regenerate a figure. Invocation examples are in the
top-level [README](../README.md).

- **`plot_metrics.py`** (`--metrics`, `--diffs`, `--out`) — renders the
  overhead study from `abft_metrics.csv`: throughput and overhead versus
  problem size, plus, if the calibration diff dump is present, the noise
  histograms.
- **`plot_comparison.py`** (`--csv`, `--outdir`) — renders one figure per
  shape regime from `compare_all.csv`, drawing all six series
  (`ours_*` and `theirs_*`). Temporal overhead is derived from GFLOPS as
  `t = 2·M·N·K / GFLOPS`.
- **`plot_real_gflops.py`** (`--out`, `--indir` / `--sweep-csv`, `--tool`) —
  renders the measured-GFLOPS comparison from the Nsight Compute metric
  tables, the hardware-counter counterpart to the theoretical throughput.
