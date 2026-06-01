#!/usr/bin/env python3
import argparse, csv, glob, os, re, sys
from collections import defaultdict

FADD = "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
FMUL = "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
FFMA = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
DUR  = "gpu__time_duration.sum"

UNIT_TO_S = {"second": 1.0, "msecond": 1e-3, "usecond": 1e-6,
             "nsecond": 1e-9, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}

def parse_ncu_csv(path, kernel_substr=None):
    """real_gflops = (fadd + fmul + 2*ffma) / sum(duration) / 1e9, summed
    over the profiled kernels.  When ``kernel_substr`` is given, only
    kernels whose name contains that substring are counted — used to drop
    theirs' one-off cutlass *reference* GEMM (a harness verification
    artifact) so the number reflects only their fused ABFT kernel."""
    s = defaultdict(float)
    dur_s = 0.0
    with open(path, newline="", errors="ignore") as f:
        rows = list(csv.reader(f))
    hdr_i = None
    for i, r in enumerate(rows):
        if "Metric Name" in r and "Metric Value" in r:
            hdr_i = i
            break
    if hdr_i is None:
        return None
    hdr = rows[hdr_i]
    ci = {n: hdr.index(n) for n in
          ("Kernel Name", "Metric Name", "Metric Value", "Metric Unit")
          if n in hdr}
    mn, mv = ci["Metric Name"], ci["Metric Value"]
    mu = ci.get("Metric Unit")
    kn = ci.get("Kernel Name")
    for r in rows[hdr_i + 1:]:
        if len(r) <= max(mn, mv):
            continue
        if (kernel_substr and kn is not None and len(r) > kn
                and kernel_substr not in r[kn]):
            continue
        name = r[mn].strip()
        raw = r[mv].strip().replace(",", "")
        if not raw or raw.lower() in ("n/a", "nan"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if name == DUR:
            unit = (r[mu].strip().lower() if mu is not None and len(r) > mu
                    else "nsecond")
            dur_s += val * UNIT_TO_S.get(unit, 1e-9)
        elif name in (FADD, FMUL, FFMA):
            s[name] += val
    if dur_s <= 0:
        return None
    flops = s[FADD] + s[FMUL] + 2.0 * s[FFMA]
    if flops <= 0:
        return None
    return flops / dur_s / 1e9

_NVUNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
           "second": 1.0, "msecond": 1e-3, "usecond": 1e-6, "nsecond": 1e-9}

def _nv_rows(path):
    with open(path, newline="", errors="ignore") as f:
        return list(csv.reader(f))

def _nv_find_header(rows, needles):
    for i, r in enumerate(rows):
        if all(any(n == c.strip() for c in r) for n in needles):
            return i
    return None

def parse_nvprof_flops(path):
    """flop_count_sp = sum(Avg * Invocations) over kernels."""
    if not os.path.exists(path):
        return None
    rows = _nv_rows(path)
    hi = _nv_find_header(rows, ("Metric Name", "Avg", "Invocations"))
    if hi is None:
        return None
    hdr = [c.strip() for c in rows[hi]]
    mn, av, iv = hdr.index("Metric Name"), hdr.index("Avg"), hdr.index("Invocations")
    tot, seen = 0.0, False
    for r in rows[hi + 1:]:
        if len(r) <= max(mn, av, iv) or r[mn].strip() != "flop_count_sp":
            continue
        try:
            tot += (float(r[av].strip().replace(",", ""))
                    * float(r[iv].strip().replace(",", "")))
            seen = True
        except ValueError:
            continue
    return tot if seen else None

def parse_nvprof_time_s(path):
    """Sum GPU-activity time (kernels only, memcpy excluded), in seconds."""
    if not os.path.exists(path):
        return None
    rows = _nv_rows(path)
    hi = _nv_find_header(rows, ("Type", "Time", "Name"))
    if hi is None:
        return None
    hdr = [c.strip() for c in rows[hi]]
    ti, tt, nm = hdr.index("Type"), hdr.index("Time"), hdr.index("Name")
    unit = "ms"
    if hi + 1 < len(rows) and len(rows[hi + 1]) > tt:
        u = rows[hi + 1][tt].strip().lower()
        if u in _NVUNIT:
            unit = u
    scale = _NVUNIT.get(unit, 1e-3)
    tot, seen = 0.0, False
    for r in rows[hi + 1:]:
        if len(r) <= max(ti, tt, nm) or "GPU activities" not in r[ti]:
            continue
        if r[nm].strip().startswith("[CUDA"):
            continue
        try:
            tot += float(r[tt].strip().replace(",", "")) * scale
            seen = True
        except ValueError:
            continue
    return tot if seen else None

def parse_nvprof_pair(indir, label):
    fl = parse_nvprof_flops(os.path.join(indir, f"{label}_flops.csv"))
    ts = parse_nvprof_time_s(os.path.join(indir, f"{label}_summary.csv"))
    if not fl or not ts or ts <= 0:
        return None
    return fl / ts / 1e9

def collect(indir, tool):
    """Return dict[regime][who] = measured GFLOPS."""
    data = defaultdict(dict)
    if tool == "ncu":
        for path in sorted(glob.glob(os.path.join(indir, "*_metrics.csv"))):
            base = os.path.basename(path)[: -len("_metrics.csv")]
            m = re.match(r"^(.*)_(theirs_fused|ours_online)$", base)
            if not m:
                continue

            ksub = "ft_sgemm" if m.group(2) == "theirs_fused" else None
            g = parse_ncu_csv(path, kernel_substr=ksub)
            if g is not None:
                data[m.group(1)][m.group(2)] = g
    else:
        for path in sorted(glob.glob(os.path.join(indir, "*_flops.csv"))):
            base = os.path.basename(path)[: -len("_flops.csv")]
            m = re.match(r"^(.*)_(theirs_fused|ours_online)$", base)
            if not m:
                continue
            g = parse_nvprof_pair(indir, base)
            if g is not None:
                data[m.group(1)][m.group(2)] = g
    return data

def plot_sweep(csv_path, out):
    """GFLOPS-vs-size line plot from a tidy sweep CSV (M,K,N,who,real_gflops),
    the measured (ncu) analog of the theoretical gflops_vs_size figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                M = int(r["M"]); g = float(r["real_gflops"])
            except (KeyError, ValueError):
                continue
            series[r["who"]].append((M, g))
    if not series:
        print(f"ERROR: no rows in {csv_path}", file=sys.stderr)
        sys.exit(1)

    style = {"ours_online":  ("tab:red",  "-D", "ours / online ABFT"),
             "theirs_fused": ("tab:gray", "-^", "theirs / fused ABFT")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    max_x = 0
    for who, (color, fmt, label) in style.items():
        pts = sorted(series.get(who, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, fmt, color=color, lw=2.3, ms=7, label=label)
        max_x = max(max_x, max(xs))
    ax.set_xlim(0, max_x * 1.03 if max_x else None)
    ax.set_xlabel("Matrix size  (M=N=K)")
    ax.set_ylabel("Measured FP32 GFLOPS  (Nsight Compute)")
    ax.set_title("Real (measured) throughput vs size — ours vs theirs")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   required=True)
    ap.add_argument("--indir", help="dir of *_metrics.csv (bar-chart mode)")
    ap.add_argument("--tool",  choices=("ncu", "nvprof"),
                    help="ncu (pacca A100) or nvprof (felix Titan X) — "
                         "required for bar-chart mode")
    ap.add_argument("--sweep-csv", dest="sweep_csv",
                    help="tidy CSV (M,K,N,who,real_gflops) -> GFLOPS-vs-size "
                         "line plot instead of the per-regime bar chart")
    args = ap.parse_args()

    if args.sweep_csv:
        plot_sweep(args.sweep_csv, args.out)
        return

    if not args.indir or not args.tool:
        ap.error("bar-chart mode needs --indir and --tool "
                 "(or use --sweep-csv for the vs-size line plot)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        sys.exit(2)

    data = collect(args.indir, args.tool)
    if not data:
        print(f"ERROR: no parseable {args.tool} CSVs in {args.indir}",
              file=sys.stderr)
        sys.exit(1)

    regimes = [r for r in ("small_sq", "big_sq", "small_ns", "big_ns")
               if r in data] + [r for r in data
                                if r not in ("small_sq", "big_sq",
                                             "small_ns", "big_ns")]
    x = np.arange(len(regimes))
    w = 0.38
    ours = [data[r].get("ours_online", 0.0) for r in regimes]
    thrs = [data[r].get("theirs_fused", 0.0) for r in regimes]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, ours, w, label="ours / online ABFT", color="tab:red")
    b2 = ax.bar(x + w / 2, thrs, w, label="theirs / fused ABFT", color="tab:gray")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:,.0f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(regimes)
    src = "Nsight Compute" if args.tool == "ncu" else "nvprof flop_count_sp"
    ax.set_ylabel(f"Measured FP32 GFLOPS  ({src})")
    ax.set_title("Real (measured) throughput — ours vs theirs")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
