# Asynchronous ABFT for Fail-Continue Error Mitigation in GEMM (MPI + CUDA)

Source code, documentation, and experimental data accompanying the paper

> **Fail-Continue Error Mitigation in GEMM Operations: An Asynchronous ABFT
> Approach for MPI-CUDA Architectures**

**Authors:** Farid Camilo Rojas Vargas, Santiago Mauricio Caicedo Rodríguez
*(equal contribution)*, and collaborators.
**Group:** SC3UIS-CAGE (Cómputo Avanzado y a Gran Escala).
**Institution:** Universidad Industrial de Santander (UIS), Bucaramanga,
Colombia.

---

## What this is

This project wraps an **unmodified vendor GEMM** (NVIDIA cuBLAS) in an
online **Algorithm-Based Fault Tolerance (ABFT)** layer that detects,
localizes, and corrects silent data corruptions in `C = A·B`. The novelty
is *asynchronous scheduling*: the checksum verification runs
on a **second CUDA stream**, concurrent with the main compute stream, so
the verification of one iteration is hidden behind the matrix
multiplication of the next. The compute path is never blocked by the
verification path, and the multiplication itself is left to the
closed-source vendor library — so any future cuBLAS improvement is
inherited for free.

The framework runs on hybrid **MPI + CUDA** clusters: each MPI rank drives
one GPU, operands are distributed on a 2-D process grid, and the
contraction dimension is replicated so every rank's output block is final
and can be verified locally with no cross-rank communication.

For the full design rationale, start with
[docs/architecture.md](docs/architecture.md).

## Repository layout

```
.
├── README.md                  this file
├── src/                       framework source (header-only + main.cu)
├── docs/                      documentation (the source carries no comments)
│   ├── architecture.md            end-to-end design
│   ├── source-reference.md        file-by-file guide to src/
│   ├── data.md                    what each file in data/ is
│   └── comparison-methodology.md  how the head-to-head is kept fair
├── scripts/                   the exact experiment drivers used in the paper
└── data/                      raw results behind the figures and tables
```

The source files under `src/` and the scripts under `scripts/` are
intentionally **comment-free**: every explanation lives in `docs/`. See
[docs/source-reference.md](docs/source-reference.md) for a map of the code
and [docs/scripts-reference.md](docs/scripts-reference.md) for the scripts.

## Prerequisites

- **CUDA toolkit** (`nvcc`, cuBLAS). Tested with 11.8 and 12.x.
- **OpenMPI** (`mpicc`, `mpirun`). Tested with 4.1.6.
- **GCC 11** or any C++17-compatible host compiler.
- A CUDA-capable GPU. Tested on Maxwell (sm_52, GTX Titan X) and Ampere
  (sm_80, A100). For another architecture, pass `-arch=sm_XX` to `nvcc`.
- **Python 3** with `pandas` and `matplotlib`, only if you want to
  regenerate the figures from the raw data with the plotters in
  `scripts/`.

For the comparison experiment, additionally: `git`, `cmake`, `make`, and
the `cuda-samples` Common headers (auto-fetched by the comparison scripts).

## Build

The whole framework compiles to a single binary, `abft_gemm`, from
`src/main.cu`:

```bash
nvcc -O3 -std=c++17 -ccbin g++ \
     -I"$(dirname $(dirname $(which mpicc)))/include" \
     src/main.cu -o abft_gemm \
     -L"$(dirname $(dirname $(which mpicc)))/lib" -lmpi -lcublas
```

(The experiment scripts run this exact command for you.)

## Run the binary directly

```bash
# square 4096³, online ABFT (detection only), 2 GPUs
mpirun -np 2 ./abft_gemm 4096 --inject none --threshold 1.0 --csv out.csv

# rectangular M K N, unprotected baseline
mpirun -np 1 ./abft_gemm 8192 1024 8192 --baseline --csv out.csv

# online ABFT with a forced additive fault (exercises the correction path)
mpirun -np 1 ./abft_gemm 4096 --inject add --threshold 1.0 --csv out.csv
```

Full flag reference: `./abft_gemm --help`, or
[docs/source-reference.md](docs/source-reference.md) (the `core/cli.cuh`
section).

## Reproduce the paper experiments

The `scripts/` folder contains the **exact Slurm batch scripts** used to
produce the paper's data on the SC3UIS clusters (Felix, 2× Titan X /
Maxwell; PACCA, 1× A100 / Ampere). They are published as a faithful record
of what was run; the `#SBATCH` directives, node names, and `module load`
lines are cluster-specific and will need adapting to another site, but the
experiment logic (sizes, sweeps, fault model, timing protocol) is portable.
Each script builds `abft_gemm` from `src/main.cu`, runs its sweep, writes a
CSV (or profiling report), and then calls a plotter to render the figures.

| Experiment | Script | Raw output (in `data/`) | Plotter |
| ---------- | ------ | ----------------------- | ------- |
| Dual-GPU overhead sweep (paper §4.2) | [`scripts/run_abft_felix.sbatch`](scripts/run_abft_felix.sbatch) | [`data/campaign_felix/`](data/campaign_felix/) | `scripts/plot_metrics.py` |
| Comparison vs fused-kernel ABFT, Maxwell (§4.3) | [`scripts/run_comparison_felix.sbatch`](scripts/run_comparison_felix.sbatch) | [`data/comparison_felix/`](data/comparison_felix/) | `scripts/plot_comparison.py` |
| Comparison vs fused-kernel ABFT, Ampere (§4.3) | [`scripts/run_comparison_pacca.sbatch`](scripts/run_comparison_pacca.sbatch) | [`data/comparison_pacca/`](data/comparison_pacca/) | `scripts/plot_comparison.py` |
| Nsight profiling: timeline + real GFLOPS (Ampere) | [`scripts/run_profile_pacca.sbatch`](scripts/run_profile_pacca.sbatch) | [`data/profile_pacca/`](data/profile_pacca/) | `scripts/plot_real_gflops.py` |

Submit one with, e.g.:

```bash
sbatch scripts/run_abft_felix.sbatch
```

On a non-Slurm machine, read the body of a script and run the inner
`nvcc` / `mpirun` lines by hand; the sizes and flags are all there.

### Regenerate a figure from the committed data

You do not need a GPU to redraw the figures — the raw numbers are already
in `data/`. For example:

```bash
python3 scripts/plot_comparison.py --csv data/comparison_pacca/compare_all.csv --outdir figs/
python3 scripts/plot_metrics.py    --metrics data/campaign_felix/abft_metrics.csv --out figs/
```

To inspect the two-stream pipeline timeline, open any
`data/profile_pacca/<shape>_ours_online.nsys-rep` in the **Nsight Systems**
GUI (`nsys-ui`): the cuBLAS SGEMMs on `compute_stream` visibly overlap the
ABFT kernels on `verify_stream`.

A description of every data file and its columns is in
[docs/data.md](docs/data.md).

## Comparison baseline (Wu et al., ICS 2023)

The §4.3 comparison is run against the fused-kernel ABFT reference of

> S. Wu, Y. Zhai, J. Liu, J. Huang, Z. Jian, B. M. Wong, Z. Chen.
> *Anatomy of High-Performance GEMM with Online Fault Tolerance on GPUs.*
> ICS 2023. <https://doi.org/10.1145/3577193.3593715>

publicly available at
<https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs>. The
comparison scripts **clone it on demand** — its source is not redistributed
here. The fairness protocol (dimension matching, warm-up alignment, and
per-shape kernel selection) is documented in
[docs/comparison-methodology.md](docs/comparison-methodology.md). All credit
for the fused-kernel reference belongs to its original authors.

## Citing

If you use this code or data, please cite the accompanying paper. A BibTeX
entry will be added here once the proceedings are published.

## Acknowledgements

Developed by the **SC3UIS-CAGE** research group (Cómputo Avanzado y a Gran
Escala) at the **Universidad Industrial de Santander (UIS)**. The authors
thank UIS and UniCartagena for the institutional support and the computing
resources (the Felix and PACCA clusters) used in this work.
