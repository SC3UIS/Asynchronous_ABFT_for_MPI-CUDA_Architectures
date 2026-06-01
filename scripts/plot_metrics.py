#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_MULTI_SIZE = False
out_dir_global = None

def _label(row: pd.Series) -> str:
    """Compact x-tick.  The figure title/legend say what the metric is,
    so labels stay short (the old verbose 'online/swifi[zone]@4096' made
    the x-axis unreadable).  SWIFI rows -> just the zone; faceting by
    size (see _facet_sizes) carries the size instead of a suffix."""
    if int(row.get("calibrate", 0)) == 1:
        return "calib"
    if int(row.get("baseline_only", 0)) == 1:
        return "base"
    inj = str(row.get("inject", ""))
    if inj == "swifi":
        return str(row.get("swifi_zone", "any"))
    return inj

def _bar_with_error(ax, labels, medians, mins, maxs, ylabel, title, color):
    yerr_low = [m - lo for m, lo in zip(medians, mins)]
    yerr_hi  = [hi - m for m, hi in zip(medians, maxs)]
    bars = ax.bar(labels, medians, yerr=[yerr_low, yerr_hi],
                  capsize=4, color=color, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, m in zip(bars, medians):
        ax.text(b.get_x() + b.get_width() / 2.0, m,
                f"{m:.2f}", ha="center", va="bottom", fontsize=8)

def plot_protected_timing(df: pd.DataFrame, out_dir: Path):
    global out_dir_global
    out_dir_global = out_dir

    def draw(s, ax, S):
        s = s.sort_values(["baseline_only", "inject", "swifi_zone"])
        labels = [_label(r) for _, r in s.iterrows()]
        _bar_with_error(ax, labels,
                        s["protected_median_ms"].tolist(),
                        s["protected_min_ms"].tolist(),
                        s["protected_max_ms"].tolist(),
                        "Protected runtime (ms)",
                        f"Protected runtime per config ({int(S)}³, "
                        f"median, min/max bars)", "tab:blue")

    _facet_sizes(df[df["calibrate"] == 0], "protected_timing", draw)

def plot_overhead(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["baseline_only"] == 0) & (df["calibrate"] == 0)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [_label(r) for _, r in sub.iterrows()]
    over   = sub["overhead_pct"].tolist()
    bars = ax.bar(labels, over, color="tab:orange",
                  edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Runtime overhead vs. unprotected baseline")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Runtime overhead per protected configuration")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, over):
        ax.text(b.get_x() + b.get_width() / 2.0, v,
                f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overhead.png", dpi=150)
    plt.close(fig)

def _series_by_size(df, mask):
    sub = df[mask].sort_values("M")
    return sub["M"].tolist(), sub

def plot_gflops_vs_size(df: pd.DataFrame, out_dir: Path):
    """Throughput vs size: baseline cuBLAS / ABFT (no fault) /
    ABFT (additive fault).  Line plot, like the comparison figures.
    X-axis anchored at 0 to match the paper's Fig.21 style."""
    nz = (df["calibrate"] == 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    max_x = 0
    bx, bsub = _series_by_size(df, nz & (df["baseline_only"] == 1))
    if bx:
        ax.plot(bx, bsub["baseline_gflops_median"], "-o", color="tab:green",
                lw=2.2, ms=6, label="baseline cuBLAS"); plotted = True
        max_x = max(max_x, max(bx))
    nx, nsub = _series_by_size(df, nz & (df["inject"] == "none")
                                   & (df["baseline_only"] == 0))
    if nx:
        if not bx:
            ax.plot(nx, nsub["baseline_gflops_median"], "-o",
                    color="tab:green", lw=2.2, ms=6, label="baseline cuBLAS")
        ax.plot(nx, nsub["protected_gflops_median"], "-D", color="tab:red",
                lw=2.2, ms=6, label="ABFT (no fault)"); plotted = True
        max_x = max(max_x, max(nx))
    ax_, asub = _series_by_size(df, nz & (df["inject"] == "add"))
    if ax_:
        ax.plot(ax_, asub["protected_gflops_median"], "--s",
                color="tab:orange", lw=2.2, ms=6,
                label="ABFT (fault)"); plotted = True
        max_x = max(max_x, max(ax_))
    if not plotted:
        plt.close(fig); return
    ax.set_xlim(0, max_x * 1.03 if max_x > 0 else None)
    ax.set_xlabel("Matrix size  (M=N=K)")
    ax.set_ylabel("GFLOPS (median)")
    ax.set_title("Throughput vs size — baseline vs ABFT (±fault)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gflops_vs_size.png", dpi=150)
    plt.close(fig)

def plot_time_vs_size(df: pd.DataFrame, out_dir: Path):
    """Absolute runtime (ms) vs size — baseline cuBLAS, ABFT no-fault,
    ABFT additive-fault.  Analog of plot_gflops_vs_size."""
    nz = (df["calibrate"] == 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    bx, bsub = _series_by_size(df, nz & (df["baseline_only"] == 1))
    if bx:
        ax.plot(bx, bsub["baseline_median_ms"], "-o", color="tab:green",
                lw=2.2, ms=6, label="baseline cuBLAS"); plotted = True
    nx, nsub = _series_by_size(df, nz & (df["inject"] == "none")
                                   & (df["baseline_only"] == 0))
    if nx:
        if not bx:
            ax.plot(nx, nsub["baseline_median_ms"], "-o",
                    color="tab:green", lw=2.2, ms=6, label="baseline cuBLAS")
        ax.plot(nx, nsub["protected_median_ms"], "-D", color="tab:red",
                lw=2.2, ms=6, label="ABFT (no fault)"); plotted = True
    ax_, asub = _series_by_size(df, nz & (df["inject"] == "add"))
    if ax_:
        ax.plot(ax_, asub["protected_median_ms"], "--s",
                color="tab:orange", lw=2.2, ms=6,
                label="ABFT (fault)"); plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlim(left=0)
    ax.set_xlabel("Matrix size  (M=N=K)")
    ax.set_ylabel("Runtime (ms, median)")
    ax.set_title("Runtime vs size — baseline vs ABFT (±fault)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_size.png", dpi=150)
    plt.close(fig)

def plot_runtime_overhead_vs_size(df: pd.DataFrame, out_dir: Path):
    """ABFT runtime overhead (%) vs our cuBLAS baseline — no-fault + fault."""
    nz = (df["calibrate"] == 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    any_ = False
    for inj, sty, col, lab in (
        ("none", "-D",  "tab:red",    "ABFT (no fault)"),
        ("add",  "--s", "tab:orange", "ABFT (fault)"),
    ):
        xs, sub = _series_by_size(df, nz & (df["inject"] == inj)
                                      & (df["baseline_only"] == 0))
        if xs:
            ax.plot(xs, sub["overhead_pct"], sty, color=col, lw=2.2, ms=6,
                    label=lab); any_ = True
    if not any_:
        plt.close(fig); return
    ax.axhline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_xlim(left=0)
    ax.set_xlabel("Matrix size  (M=N=K)")
    ax.set_ylabel("Runtime overhead vs our cuBLAS")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("ABFT runtime overhead vs our cuBLAS baseline")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_overhead_vs_size.png", dpi=150)
    plt.close(fig)

def plot_gflops_overhead_vs_size(df: pd.DataFrame, out_dir: Path):
    """ABFT GFLOPS overhead (%) vs our cuBLAS baseline — no-fault + fault.
    overhead_gflops% = (gflops_baseline / gflops_abft - 1) * 100, the
    throughput-degradation counterpart of the runtime overhead plot."""
    nz = (df["calibrate"] == 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    any_ = False
    for inj, sty, col, lab in (
        ("none", "-D",  "tab:red",    "ABFT (no fault)"),
        ("add",  "--s", "tab:orange", "ABFT (fault)"),
    ):
        xs, sub = _series_by_size(df, nz & (df["inject"] == inj)
                                      & (df["baseline_only"] == 0))
        if xs:
            base = sub["baseline_gflops_median"].to_numpy()
            prot = sub["protected_gflops_median"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                ov = np.where(prot > 0, (base / prot - 1.0) * 100.0, 0.0)
            ax.plot(xs, ov, sty, color=col, lw=2.2, ms=6, label=lab)
            any_ = True
    if not any_:
        plt.close(fig); return
    ax.axhline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_xlim(left=0)
    ax.set_xlabel("Matrix size  (M=N=K)")
    ax.set_ylabel("Throughput overhead vs our cuBLAS")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("ABFT throughput overhead vs our cuBLAS baseline")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gflops_overhead_vs_size.png", dpi=150)
    plt.close(fig)

def plot_overhead_by_zone(df: pd.DataFrame, out_dir: Path):
    """Runtime overhead per SWIFI zone, grouped by size (short labels)."""
    sub = df[(df["inject"] == "swifi") & (df["calibrate"] == 0)].copy()
    if sub.empty:
        return
    zones = ["any", "sign", "exponent", "sig_high", "sig_low"]
    zones = [z for z in zones if z in set(sub["swifi_zone"])]
    sizes = sorted(sub["M"].unique())
    x = range(len(zones))
    n = max(1, len(sizes))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(zones)), 4.6))
    for k, S in enumerate(sizes):
        ov = []
        for z in zones:
            r = sub[(sub["swifi_zone"] == z) & (sub["M"] == S)]
            ov.append(float(r["overhead_pct"].iloc[-1]) if not r.empty else 0.0)
        ax.bar([i + (k - (n - 1) / 2) * w for i in x], ov, w,
               label=f"{int(S)}³", edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(zones)
    ax.set_ylabel("Runtime overhead")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("ABFT overhead by SWIFI injection zone")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="size")
    fig.tight_layout()
    fig.savefig(out_dir / "overhead_by_zone.png", dpi=150)
    plt.close(fig)

def _facet_sizes(df, base_name, draw):
    """Call draw(sub, ax) once per matrix size (separate PNGs when the
    CSV spans several sizes) so each x-axis only has the ~5 short zone
    labels instead of zone×size crammed together."""
    sizes = sorted(df["M"].unique())
    multi = len(sizes) > 1
    for S in sizes:
        sub = df[df["M"] == S]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        draw(sub, ax, S)
        fig.tight_layout()
        name = f"{base_name}_{int(S)}.png" if multi else f"{base_name}.png"
        fig.savefig(out_dir_global / name, dpi=150)
        plt.close(fig)

def plot_detection_metrics(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["inject"] == "swifi") & (df["calibrate"] == 0)].copy()
    if sub.empty:
        return
    global out_dir_global
    out_dir_global = out_dir

    def draw(s, ax, S):
        s = s.sort_values("swifi_zone")
        labels = [_label(r) for _, r in s.iterrows()]
        x = range(len(labels))
        wdt = 0.27
        ax.bar([i - wdt for i in x], s["recall"], wdt, label="Recall",
               color="tab:green", edgecolor="black", linewidth=0.5)
        ax.bar(list(x), s["precision"], wdt, label="Detection precision",
               color="tab:blue", edgecolor="black", linewidth=0.5)
        ax.bar([i + wdt for i in x], s["correction_precision_pct"] / 100.0,
               wdt, label="Correction precision",
               color="tab:purple", edgecolor="black", linewidth=0.5)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate (1.0 = 100%)")
        ax.set_title(f"Detection / correction quality (SWIFI, {int(S)}³)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="lower right")

    _facet_sizes(sub, "detection_metrics", draw)

def plot_confusion_matrix(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["inject"] == "swifi") & (df["calibrate"] == 0)].copy()
    if sub.empty:
        return
    global out_dir_global
    out_dir_global = out_dir

    def draw(s, ax, S):
        s = s.sort_values("swifi_zone")
        labels = [_label(r) for _, r in s.iterrows()]
        bottom = [0] * len(labels)
        for col, color, name in (("TP", "tab:green", "TP"),
                                 ("FN", "tab:red", "FN"),
                                 ("FP", "tab:orange", "FP"),
                                 ("TN", "tab:gray", "TN")):
            vals = s[col].tolist()
            ax.bar(labels, vals, bottom=bottom, color=color,
                   edgecolor="black", linewidth=0.5, label=name)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("Fragments observed")
        ax.set_title(f"Confusion matrix (SWIFI, {int(S)}³)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right")

    _facet_sizes(sub, "confusion_matrix", draw)

def _calib_reference_lines(df_metrics: pd.DataFrame | None):
    """Pull the observed max and the suggested threshold from the
    most recent CALIBRATE row in the metrics CSV, if available."""
    if df_metrics is None or df_metrics.empty:
        return None, None
    calib_rows = df_metrics[df_metrics["calibrate"] == 1]
    if calib_rows.empty:
        return None, None
    last = calib_rows.iloc[-1]
    return float(last["calib_max_diff"]), float(last["calib_suggested_tau"])

def plot_calibration_distribution(diffs_csv: Path,
                                  df_metrics: pd.DataFrame | None,
                                  out_dir: Path):
    df = pd.read_csv(diffs_csv)
    if df.empty:
        print(f"WARNING: {diffs_csv} has no data; skipping distribution plot",
              file=sys.stderr)
        return
    diffs = df["abs_diff"].to_numpy()
    if diffs.size == 0:
        return

    obs_max, sugg_tau = _calib_reference_lines(df_metrics)

    n_total   = diffs.size
    n_zero    = int(np.sum(diffs == 0.0))
    diffs_pos = diffs[diffs > 0.0]

    pcts_q = [0.50, 0.90, 0.99, 0.999, 1.00]
    if diffs_pos.size > 0:
        pcts_v = np.quantile(diffs_pos, pcts_q)
    else:
        pcts_v = [0.0] * len(pcts_q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    if diffs_pos.size > 0:
        log_min = np.log10(diffs_pos.min())
        log_max = np.log10(diffs_pos.max())
        bins = np.logspace(log_min, log_max, 80) if log_max > log_min \
               else np.array([diffs_pos.min(), diffs_pos.max() + 1e-30])
        ax.hist(diffs_pos, bins=bins, color="tab:blue",
                edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|actualRow − expectedRow|")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Calibration noise distribution\n"
                 f"({n_total:,} samples, {n_zero:,} exact zeros)")
    if obs_max is not None:
        ax.axvline(obs_max,  color="red",   linestyle="--",
                   label=f"observed max = {obs_max:.3g}")
    if sugg_tau is not None:
        ax.axvline(sugg_tau, color="green", linestyle="--",
                   label=f"suggested τ  = {sugg_tau:.3g}")
    ax.grid(linestyle=":", alpha=0.5)
    if obs_max is not None or sugg_tau is not None:
        ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    if diffs_pos.size > 0:
        sorted_d = np.sort(diffs_pos)
        cdf = np.arange(1, sorted_d.size + 1) / sorted_d.size
        ax.plot(sorted_d, cdf, color="tab:blue", linewidth=1.4)
        ax.set_xscale("log")
    ax.set_xlabel("|actualRow − expectedRow|")
    ax.set_ylabel("CDF (fraction of samples ≤ x)")
    ax.set_title("Cumulative distribution (positive samples)")
    ax.set_ylim(0, 1.02)
    if obs_max is not None:
        ax.axvline(obs_max,  color="red",   linestyle="--")
    if sugg_tau is not None:
        ax.axvline(sugg_tau, color="green", linestyle="--")
    ax.grid(linestyle=":", alpha=0.5)

    txt_lines = [f"p{int(q*100) if q < 1 else 100}  =  {v:.3g}"
                 for q, v in zip(pcts_q, pcts_v)]
    ax.text(1.02, 0.5, "\n".join(txt_lines),
            transform=ax.transAxes, va="center", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray"))

    fig.tight_layout()
    fig.savefig(out_dir / "calibration_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default="abft_metrics.csv",
                   help="path to abft_metrics.csv")
    p.add_argument("--diffs",   default="abft_calibration_diffs.csv",
                   help="path to calibration diffs CSV (skipped if missing)")
    p.add_argument("--out",     default="scripts/plots",
                   help="directory to write PNGs into")

    p.add_argument("positional", nargs="*", help=argparse.SUPPRESS)
    args = p.parse_args()

    if len(args.positional) >= 1:
        args.metrics = args.positional[0]
    if len(args.positional) >= 2:
        args.out     = args.positional[1]

    metrics_path = Path(args.metrics)
    diffs_path   = Path(args.diffs)
    out_dir      = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = None
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        if not df_metrics.empty:
            global _MULTI_SIZE
            _MULTI_SIZE = (
                df_metrics[["M", "K", "N"]].drop_duplicates().shape[0] > 1
            )
            plot_protected_timing(df_metrics,        out_dir)
            plot_gflops_vs_size(df_metrics,          out_dir)
            plot_time_vs_size(df_metrics,            out_dir)
            plot_runtime_overhead_vs_size(df_metrics, out_dir)
            plot_gflops_overhead_vs_size(df_metrics,  out_dir)
            plot_overhead_by_zone(df_metrics,        out_dir)
            plot_detection_metrics(df_metrics,       out_dir)
            plot_confusion_matrix(df_metrics,        out_dir)
        else:
            print(f"NOTE: {metrics_path} is empty, skipping metrics plots",
                  file=sys.stderr)
    else:
        print(f"NOTE: {metrics_path} not found, skipping metrics plots",
              file=sys.stderr)

    if diffs_path.exists():
        plot_calibration_distribution(diffs_path, df_metrics, out_dir)
    else:
        print(f"NOTE: {diffs_path} not found, skipping calibration "
              f"distribution plot", file=sys.stderr)

    print(f"Wrote plots to {out_dir.resolve()}")
    for png in sorted(out_dir.glob("*.png")):
        print(f"  {png.name}")

if __name__ == "__main__":
    main()
