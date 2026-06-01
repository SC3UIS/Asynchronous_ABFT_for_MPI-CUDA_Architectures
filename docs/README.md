# Documentation

This folder documents the source code under [`../src`](../src), the
experiment scripts under [`../scripts`](../scripts), and the raw data under
[`../data`](../data). The source and the scripts are kept comment-free on
purpose; every explanation that would normally live in a code or script
comment lives here instead.

## Index

| Document | What it covers |
| -------- | -------------- |
| [architecture.md](architecture.md) | The design of the framework end to end: fault model, distributed decomposition, the Huang–Abraham checksum, the asynchronous two-stream pipeline, and the device-resident verification path. Read this first. |
| [source-reference.md](source-reference.md) | A file-by-file reference of every translation unit in [`../src`](../src): what each header defines, the key functions, and how the pieces fit together. |
| [scripts-reference.md](scripts-reference.md) | A guide to the comment-free experiment drivers and plotters in [`../scripts`](../scripts): what each runs, its knobs, the timing methodology, and the outputs it produces. |
| [data.md](data.md) | What each file under [`../data`](../data) is, which experiment produced it, and which paper figure/table it backs. |
| [comparison-methodology.md](comparison-methodology.md) | How the head-to-head comparison against the fused-kernel reference is set up to be fair: dimension matching, warm-up alignment, and per-shape kernel selection. |

## At a glance

The framework wraps an unmodified vendor GEMM (cuBLAS) in an online
Algorithm-Based Fault Tolerance (ABFT) layer. The checksum verification
runs on a second CUDA stream, concurrent with the main compute stream, so
the localization work of one iteration overlaps with the matrix
multiplications of the next. The contribution is the *asynchronous
scheduling*: the compute path is never blocked by the verification path,
and the multiplication itself is left to the closed-source vendor library.

For build and run instructions see the top-level [README](../README.md).
