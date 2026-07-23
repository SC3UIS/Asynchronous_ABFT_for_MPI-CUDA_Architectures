# Data

Everything under [`../data`](../data) is the experimental output behind the
figures and tables in the paper. The primary artifacts are **raw**: the
per-run CSVs and the Nsight `.nsys-rep` / `.ncu-rep` reports. The line-plot
graphs (throughput-vs-size and the per-regime comparison curves) are *not*
committed — the plotters in [`../scripts`](../scripts) regenerate them from
the CSVs (see the top-level [README](../README.md)).

Two sets of rendered `.png` images **are** committed, because the paper
links to them directly: the per-platform **runtime-overhead figures**
(the percentage-overhead companions to the throughput plots) and the
**Nsight Systems timeline screenshots** (the two-stream overlap made
visible). They are described with their folders below.

The data is organized by the three experiments that produced it.

```
data/
├── campaign_felix/        dual-GPU overhead sweep on Felix (2× Titan X)
│   ├── abft_metrics.csv
│   └── gflops_overhead_vs_size.png            overhead-%-vs-size figure
├── comparison_felix/      head-to-head vs the fused-kernel reference (1× Titan X)
│   ├── compare_all.csv
│   └── overhead_vs_{ours,theirs}_{small,big}_{sq,ns}.png   per-regime overhead figures (8)
├── comparison_pacca/      head-to-head vs the fused-kernel reference (1× A100)
│   ├── compare_all.csv
│   └── overhead_vs_{ours,theirs}_{small,big}_{sq,ns}.png   per-regime overhead figures (8)
└── profile_pacca/         Nsight profiling on the A100 (timeline + real GFLOPS)
    ├── <regime>_<who>.nsys-rep        Nsight Systems timeline
    ├── <regime>_ours_online.png       timeline screenshot (two-stream overlap)
    ├── <regime>_<who>.ncu-rep         Nsight Compute report
    ├── <regime>_<who>_metrics.csv     ncu-derived FP throughput
    ├── <regime>_<who>_{ncu,nsys}_run.log
    └── real_gflops_sweep.csv          measured (not theoretical) GFLOPS
```

## `campaign_felix/` — overhead study (paper §4.2)

Produced by [`scripts/run_abft_felix.sbatch`](../scripts/run_abft_felix.sbatch)
on two Titan X GPUs of the Felix node (`Pr=1, Pc=2`).

### `abft_metrics.csv`

One row per `(size, configuration)`. This is the source for the
throughput-vs-size overhead figure. Columns of interest:

| Column | Meaning |
| ------ | ------- |
| `scheme` | `baseline` or `online` |
| `inject` | `none`, `add` (always-on additive fault), or `swifi` (single bit-flip) |
| `M`, `K`, `N` | problem dimensions |
| `Pr`, `Pc` | process grid (here `1×2` = dual-GPU) |
| `frags_per_rank`, `num_fragments_total` | pipeline depth |
| `baseline_gflops_mean` | mean GFLOPS of the unprotected cuBLAS reference |
| `protected_gflops_mean` | mean GFLOPS of the protected execution |
| `overhead_pct` | `100·(t_protected − t_baseline)/t_baseline` |
| `TP/TN/FP/FN`, `recall`, `precision` | detection bookkeeping |

The `*_min/median/mean/max_ms` and matching `*_gflops_*` columns give the
full timing distribution per row; the paper reports the **mean**. The
overhead figures measure the protected mean against the standalone
`baseline` phase's mean (the same reference the throughput plot draws),
rather than the per-row `overhead_pct`, which is computed against each
run's own in-line baseline pass.

`gflops_overhead_vs_size.png` is the committed percentage-overhead figure
for this sweep — the ABFT throughput cost, without and with an active
fault, as a function of size.

> The `run_abft_felix.sbatch` driver also runs a detection-accuracy (CM)
> campaign that produces calibration and per-zone-recall CSVs (see
> [scripts-reference.md](scripts-reference.md)). Those outputs feed the
> detection study rather than the throughput figures and are **not shipped**
> in this release; the overhead figures rely only on `abft_metrics.csv`.

## `comparison_felix/` and `comparison_pacca/` — head-to-head (paper §4.3)

Produced by
[`scripts/run_comparison_felix.sbatch`](../scripts/run_comparison_felix.sbatch)
(1× Titan X) and
[`scripts/run_comparison_pacca.sbatch`](../scripts/run_comparison_pacca.sbatch)
(1× A100).

### `compare_all.csv`

Long format, several rows per shape. Columns:

| Column | Values |
| ------ | ------ |
| `regime` | `small_sq`, `big_sq`, `small_ns`, `big_ns` |
| `M`, `K`, `N` | problem dimensions |
| `who` | `ours_baseline`, `ours_online`, `ours_online_inj`, `theirs_cublas`, `theirs_fused`, `theirs_fused_inj` |
| `gflops` | mean GFLOPS |
| `time_ms` | mean wall-clock time per GEMM, `2·M·K·N / (gflops·1e9) · 1e3` |

The methodology that makes this comparison fair — dimension matching,
warm-up alignment, and per-shape kernel selection — is documented in
[comparison-methodology.md](comparison-methodology.md).

Alongside each `compare_all.csv` are the committed percentage-overhead
figures, one pair per shape regime: `overhead_vs_ours_<regime>.png` plots
the four ABFT curves' runtime overhead relative to *our* cuBLAS baseline,
and `overhead_vs_theirs_<regime>.png` relative to *their* cuBLAS baseline
(`<regime>` ∈ `{small,big}_{sq,ns}`). They express the same data as the
throughput plots, in percent.

## `profile_pacca/` — Nsight profiling (paper timeline + real GFLOPS)

Produced by
[`scripts/run_profile_pacca.sbatch`](../scripts/run_profile_pacca.sbatch) on
the A100, one representative shape per regime, two profiles each
(`ours_online` and `theirs_fused`):

- **`<regime>_<who>.nsys-rep`** — the Nsight Systems timeline. This is the
  artifact behind the two-stream pipeline visualization: open it in the
  Nsight Systems GUI (`nsys-ui`) to see the cuBLAS SGEMMs on
  `compute_stream` overlapping the ABFT kernels on `verify_stream`.
- **`<regime>_ours_online.png`** — a committed screenshot of that timeline
  for the protected run, one per regime, so the overlap can be seen without
  opening the `.nsys-rep`. These are the images the paper's profiling
  footnote links to.
- **`<regime>_<who>.ncu-rep`** — the Nsight Compute report (per-kernel
  counters), the authoritative source of *measured* FP32 throughput as
  opposed to the theoretical `2·M·N·K / walltime`.
- **`<regime>_<who>_metrics.csv`** — the FP-throughput numbers extracted
  from the ncu report.
- **`real_gflops_sweep.csv`** — measured GFLOPS across the size sweep
  (`M,K,N,who,real_gflops`), the ncu-backed counterpart to the theoretical
  throughput in the comparison CSVs.
- **`*_run.log`** — the raw `ncu` / `nsys` invocation logs.

> Note: the binary `.nsys-rep` / `.ncu-rep` reports embed, in their
> metadata, the cluster working directory under which they were captured.
> The text outputs have been normalized to `<REPO_ROOT>`.
