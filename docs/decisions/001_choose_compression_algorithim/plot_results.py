from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

FAMILY_COLORS = {
    "lossless": "#4878cf",
    "BitRound": "#e87d2c",
    "Quantize": "#6abe45",
    "ZFP":      "#9b59b6",
    "SZ3":      "#e74c3c",
}


def _hbars(ax, y, labels, colors, values, title, xlabel, ref_line=None, fmt="{:.1f}", fontsize=7):
    height = 0.65
    bars = ax.barh(y, values, height=height, color=colors, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    if ref_line is not None:
        ax.axvline(ref_line, color="#555", linestyle="--", linewidth=0.9)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + ax.get_xlim()[1] * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(v), va="center", ha="left", fontsize=fontsize,
        )
    xmax = ax.get_xlim()[1]
    if xmax > 0:
        ax.set_xlim(right=xmax * 1.22)


def _plot_lowhigh_flow(ax, y, labels, colors, low_flow, high_flow):
    # low-flow vs high-flow MAE, grouped per codec: solid = low-flow (bottom decile),
    # hatched = high-flow (top decile) -- shows whether error concentrates at low flow.
    # Log scale + clipped floor since the worst codec's error would otherwise flatten
    # every other codec's bars to invisibility on a linear axis.
    h = 0.32
    floor = 1e-7
    low_vals = low_flow.clip(lower=floor)
    high_vals = high_flow.clip(lower=floor)
    ax.barh(y - h / 2, low_vals,  height=h, color=colors, zorder=3)
    ax.barh(y + h / 2, high_vals, height=h, color=colors, hatch="//", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("MAE: low-flow vs high-flow deciles  ↓", fontsize=9, fontweight="bold")
    ax.set_xlabel("m³/s (log, 0 shown at floor)", fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    # skip labeling exact-zero (lossless) rows -- both bars sit on the same clipped
    # floor point there, so the two text labels would otherwise overlap illegibly.
    for i, v in enumerate(low_flow):
        if v > 0:
            ax.text(low_vals.iloc[i] * 1.3, i - h / 2, "{:.1e}".format(v), va="center", fontsize=6)
    for i, v in enumerate(high_flow):
        if v > 0:
            ax.text(high_vals.iloc[i] * 1.3, i + h / 2, "{:.1e}".format(v), va="center", fontsize=6)
    style_handles = [Patch(facecolor="white", edgecolor="black", label="low-flow (bottom 10%)"),
                     Patch(facecolor="white", edgecolor="black", hatch="//", label="high-flow (top 10%)")]
    ax.legend(handles=style_handles, fontsize=6, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 3)


def _plot_tradeoff(ax, df: pd.DataFrame, ratio_col: str = "ratio",
                    xlabel: str = "Compression ratio (log)", ref_line: float | None = None):
    for family, color in FAMILY_COLORS.items():
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        ax.scatter(sub[ratio_col], sub["mae_nonzero"].clip(lower=1e-9), color=color, label=family, s=50, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["codec"], (row[ratio_col], max(row["mae_nonzero"], 1e-9)),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Compression ratio vs. accuracy loss  ↓", fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("MAE, nonzero cells, m³/s (log)", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    if ref_line is not None:
        ax.axvline(ref_line, color="#555", linestyle="--", linewidth=0.9)


def plot_combined(bench_df: pd.DataFrame, accuracy_df: pd.DataFrame, full_df: pd.DataFrame,
                   output_dir: Path, region_label: str = "Global") -> None:
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.3])
    fig.suptitle(
        f"Zarr v3 Compression Benchmark & Accuracy — {region_label} discharge 1985 (float32, sharded)",
        fontsize=14, fontweight="bold",
    )

    bench_labels = bench_df["codec"].tolist()
    bench_colors = [FAMILY_COLORS[f] for f in bench_df["family"]]
    bench_y = np.arange(len(bench_labels))
    baseline_mb = float(bench_df.loc[bench_df["codec"] == "no-compression", "size_mb"].iloc[0])

    ax_ratio = fig.add_subplot(gs[0, 0])
    ax_size  = fig.add_subplot(gs[0, 1])
    ax_write = fig.add_subplot(gs[1, 0])
    ax_read  = fig.add_subplot(gs[1, 1])

    _hbars(ax_ratio, bench_y, bench_labels, bench_colors, bench_df["ratio"],
           "Compression ratio  ↑", "ratio", ref_line=1, fmt="{:.1f}×")
    _hbars(ax_size, bench_y, bench_labels, bench_colors, bench_df["size_mb"],
           "Disk size  ↓", "MB", ref_line=baseline_mb, fmt="{:.0f}")
    _hbars(ax_write, bench_y, bench_labels, bench_colors, bench_df["write_s"],
           "Write time  ↓", "seconds", fmt="{:.2f}s")
    _hbars(ax_read, bench_y, bench_labels, bench_colors, bench_df["read_s"],
           "Read time  ↓", "seconds", fmt="{:.2f}s")

    acc_labels = accuracy_df["codec"].tolist()
    acc_colors = [FAMILY_COLORS[f] for f in accuracy_df["family"]]
    acc_y = np.arange(len(acc_labels))

    ax_max     = fig.add_subplot(gs[2, 0])
    ax_mae     = fig.add_subplot(gs[2, 1])
    ax_lowhigh = fig.add_subplot(gs[3, 0])
    ax_zeroed  = fig.add_subplot(gs[3, 1])

    _hbars(ax_max, acc_y, acc_labels, acc_colors, accuracy_df["max_abs_err"],
           "Max abs error (all cells)  ↓", "m³/s", fmt="{:.2e}")
    _hbars(ax_mae, acc_y, acc_labels, acc_colors, accuracy_df["mae_nonzero"],
           "MAE (nonzero cells)  ↓", "m³/s", fmt="{:.2e}")
    _plot_lowhigh_flow(ax_lowhigh, acc_y, acc_labels, acc_colors,
                        accuracy_df["mae_low_flow"], accuracy_df["mae_high_flow"])
    _hbars(ax_zeroed, acc_y, acc_labels, acc_colors, accuracy_df["low_flow_zeroed_frac"] * 100,
           "Low-flow values rounded to exactly 0  ↓", "% of low-flow cells", fmt="{:.2f}%")

    ax_tradeoff = fig.add_subplot(gs[4, :])
    _plot_tradeoff(ax_tradeoff, full_df)

    families_present = set(bench_df["family"]) | set(accuracy_df["family"])
    patches = [Patch(color=c, label=f) for f, c in FAMILY_COLORS.items() if f in families_present]
    ax_ratio.legend(handles=patches, loc="lower right", fontsize=6, title="Codec family", title_fontsize=7)

    out = output_dir / "results_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close(fig)


def plot_relative_to_zlib4(bench_df: pd.DataFrame, accuracy_df: pd.DataFrame, full_df: pd.DataFrame,
                            output_dir: Path, region_label: str = "Global") -> None:
    # no-compression's size/time dwarfs every real codec's, which would wreck the axis
    # scale once normalized -- it isn't a meaningful comparison point against zlib-4 anyway.
    bench_rel = bench_df[bench_df["codec"] != "no-compression"].copy()
    accuracy_rel = accuracy_df[accuracy_df["codec"] != "no-compression"].copy()
    full_rel = full_df[full_df["codec"] != "no-compression"].copy()

    zlib4 = bench_df.loc[bench_df["codec"] == "zlib-4"].iloc[0]
    bench_rel["ratio_rel"] = bench_rel["ratio"] / zlib4["ratio"]
    bench_rel["size_rel"] = bench_rel["size_mb"] / zlib4["size_mb"]
    bench_rel["write_rel"] = bench_rel["write_s"] / zlib4["write_s"]
    bench_rel["read_rel"] = bench_rel["read_s"] / zlib4["read_s"]
    full_rel["ratio_rel"] = full_rel["ratio"] / zlib4["ratio"]

    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.3])
    fig.suptitle(
        f"Zarr v3 Compression Benchmark & Accuracy relative to zlib-4 — {region_label} discharge 1985 (float32, sharded)",
        fontsize=14, fontweight="bold",
    )

    bench_labels = bench_rel["codec"].tolist()
    bench_colors = [FAMILY_COLORS[f] for f in bench_rel["family"]]
    bench_y = np.arange(len(bench_labels))

    ax_ratio = fig.add_subplot(gs[0, 0])
    ax_size  = fig.add_subplot(gs[0, 1])
    ax_write = fig.add_subplot(gs[1, 0])
    ax_read  = fig.add_subplot(gs[1, 1])

    _hbars(ax_ratio, bench_y, bench_labels, bench_colors, bench_rel["ratio_rel"],
           "Compression ratio vs zlib-4  ↑", "× zlib-4", ref_line=1.0, fmt="{:.2f}×")
    _hbars(ax_size, bench_y, bench_labels, bench_colors, bench_rel["size_rel"],
           "Disk size vs zlib-4  ↓", "× zlib-4", ref_line=1.0, fmt="{:.2f}×")
    _hbars(ax_write, bench_y, bench_labels, bench_colors, bench_rel["write_rel"],
           "Write time vs zlib-4  ↓", "× zlib-4", ref_line=1.0, fmt="{:.2f}×")
    _hbars(ax_read, bench_y, bench_labels, bench_colors, bench_rel["read_rel"],
           "Read time vs zlib-4  ↓", "× zlib-4", ref_line=1.0, fmt="{:.2f}×")

    acc_labels = accuracy_rel["codec"].tolist()
    acc_colors = [FAMILY_COLORS[f] for f in accuracy_rel["family"]]
    acc_y = np.arange(len(acc_labels))

    ax_max     = fig.add_subplot(gs[2, 0])
    ax_mae     = fig.add_subplot(gs[2, 1])
    ax_lowhigh = fig.add_subplot(gs[3, 0])
    ax_zeroed  = fig.add_subplot(gs[3, 1])

    _hbars(ax_max, acc_y, acc_labels, acc_colors, accuracy_rel["max_abs_err"],
           "Max abs error (all cells)  ↓", "m³/s", fmt="{:.2e}")
    _hbars(ax_mae, acc_y, acc_labels, acc_colors, accuracy_rel["mae_nonzero"],
           "MAE (nonzero cells)  ↓", "m³/s", fmt="{:.2e}")
    _plot_lowhigh_flow(ax_lowhigh, acc_y, acc_labels, acc_colors,
                        accuracy_rel["mae_low_flow"], accuracy_rel["mae_high_flow"])
    _hbars(ax_zeroed, acc_y, acc_labels, acc_colors, accuracy_rel["low_flow_zeroed_frac"] * 100,
           "Low-flow values rounded to exactly 0  ↓", "% of low-flow cells", fmt="{:.2f}%")

    ax_tradeoff = fig.add_subplot(gs[4, :])
    _plot_tradeoff(ax_tradeoff, full_rel, ratio_col="ratio_rel",
                   xlabel="Compression ratio vs zlib-4 (log)", ref_line=1.0)

    families_present = set(bench_rel["family"]) | set(accuracy_rel["family"])
    patches = [Patch(color=c, label=f) for f, c in FAMILY_COLORS.items() if f in families_present]
    ax_ratio.legend(handles=patches, loc="lower right", fontsize=6, title="Codec family", title_fontsize=7)

    out = output_dir / "results_relative_zlib4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single combined benchmark + accuracy plot for the Zarr v3 compression comparison")
    parser.add_argument("--input-dir",  type=Path, required=True, help="Unused by plotting, accepted for CLI parity with the other pipeline steps")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing benchmark_results.csv, accuracy_results.csv, full_results.csv")
    parser.add_argument("--temp-dir",   type=Path, required=True, help="Unused by plotting, accepted for CLI parity with the other pipeline steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from compression_benchmark import ACTIVE_REGION
    region_label = ACTIVE_REGION.replace("_", " ").title()

    bench_csv = args.output_dir / "benchmark_results.csv"
    accuracy_csv = args.output_dir / "accuracy_results.csv"
    full_csv = args.output_dir / "full_results.csv"
    missing = [p for p in (bench_csv, accuracy_csv, full_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result file(s): {missing}")

    bench_df = pd.read_csv(bench_csv)
    accuracy_df = pd.read_csv(accuracy_csv)
    full_df = pd.read_csv(full_csv)

    plot_combined(bench_df, accuracy_df, full_df, args.output_dir, region_label)
    plot_relative_to_zlib4(bench_df, accuracy_df, full_df, args.output_dir, region_label)
