import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CODEC_CONFIGS = Path(__file__).parent / "codec_configs.json"

FAMILY_COLORS = {
    "lossless": "#4878cf",
    "BitRound": "#e87d2c",
}

FLOOR = 1e-8  # clip value so exact-zero (lossless) bars stay visible on a log axis
LOSSLESS_HATCH = "//"  # visually flags the lossless family beyond just its color

LEVEL_SHADE_RANGE = (0.35, 1.0)  # lightest (lowest level) -> full base color (highest level)
LEVEL_LEGEND_COLOR = "#555555"   # neutral swatch base for the level legend

COMPONENT_LABELS = {
    "precipitation": "P",
    "total_evaporation": "E",
    "total_runoff": "R",
}


def _codec_order(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical codec/family ordering shared by every plot, taken from codec_configs.json so
    the y-axis reads identically no matter what row order each result CSV happens to have.
    Codecs missing from the config sort to the end."""
    rank = {c["label"]: i for i, c in enumerate(json.loads(CODEC_CONFIGS.read_text()))}
    return (df.drop_duplicates("codec")[["codec", "family"]]
              .sort_values("codec", key=lambda s: s.map(rank), kind="stable")
              .reset_index(drop=True))


def _shade_factors(n_levels: int) -> np.ndarray:
    """Lightest-to-darkest blend factors, one per level, lowest level first."""
    return np.linspace(*LEVEL_SHADE_RANGE, n_levels) if n_levels > 1 else np.array([LEVEL_SHADE_RANGE[1]])


def _shade_color(color, factor: float):
    """Blend `color` toward white; factor=1.0 is the full color, smaller factor is lighter."""
    base = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(white * (1 - factor) + base * factor)


def _add_level_colorbar(fig, axes, levels: list[int]) -> None:
    """Discrete colorbar explaining the light->dark level shading, ticked with the level numbers."""
    n = len(levels)
    cmap = mcolors.ListedColormap([_shade_color(LEVEL_LEGEND_COLOR, f) for f in _shade_factors(n)])
    norm = mcolors.BoundaryNorm(np.arange(n + 1), n)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="vertical",
                         location="right", shrink=0.6, aspect=15, pad=0.02, ticks=np.arange(n) + 0.5)
    cbar.ax.set_yticklabels([str(level) for level in levels], fontsize=7)
    cbar.set_label("HydroBASIN level", fontsize=8)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.invert_yaxis()  # lowest level on top, highest on bottom -- colors stay tied to their level


def _hbars(ax, y, labels, colors, values, title, xlabel, families=None, ref_line=None, fmt="{:.1f}", fontsize=7):
    hatch = [LOSSLESS_HATCH if f == "lossless" else None for f in families] if families is not None else None
    bars = ax.barh(y, values, height=0.65, color=colors, hatch=hatch, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    # explicit limits, not invert_yaxis() -- with sharey=True the toggle fires once per
    # subplot and an even panel count would cancel itself out.
    ax.set_ylim(len(y) - 0.5, -0.5)
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


def plot_benchmark_stats(bench_df: pd.DataFrame, output_dir: Path,
                          title: str = "Water Balance Compression Benchmark vs zlib-4 — size & speed",
                          out_name: str = "benchmark_stats.png") -> None:
    bench_rel = bench_df.set_index("codec").reindex(_codec_order(bench_df)["codec"]).reset_index()
    zlib4 = bench_df.loc[bench_df["codec"] == "zlib-4"].iloc[0]
    bench_rel["ratio_rel"] = bench_rel["ratio"] / zlib4["ratio"]
    bench_rel["size_rel"] = bench_rel["size_mb"] / zlib4["size_mb"]
    bench_rel["write_rel"] = bench_rel["write_s"] / zlib4["write_s"]
    bench_rel["read_rel"] = bench_rel["read_s"] / zlib4["read_s"]

    labels = bench_rel["codec"].tolist()
    colors = [FAMILY_COLORS[f] for f in bench_rel["family"]]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    families = bench_rel["family"]
    _hbars(axes[0, 0], y, labels, colors, bench_rel["ratio_rel"],
           "Compression ratio vs zlib-4  ↑", "× zlib-4", families=families, ref_line=1.0, fmt="{:.2f}×")
    _hbars(axes[0, 1], y, labels, colors, bench_rel["size_rel"],
           "Disk size vs zlib-4  ↓", "× zlib-4", families=families, ref_line=1.0, fmt="{:.2f}×")
    _hbars(axes[1, 0], y, labels, colors, bench_rel["write_rel"],
           "Write time vs zlib-4  ↓", "× zlib-4", families=families, ref_line=1.0, fmt="{:.2f}×")
    _hbars(axes[1, 1], y, labels, colors, bench_rel["read_rel"],
           "Read time vs zlib-4  ↓", "× zlib-4", families=families, ref_line=1.0, fmt="{:.2f}×")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    families_present = set(bench_rel["family"])
    patches = [Patch(color=c, label=f, hatch=LOSSLESS_HATCH if f == "lossless" else None) for f, c in FAMILY_COLORS.items() if f in families_present]
    fig.legend(handles=patches, loc="outside lower center", ncols=len(patches),
               fontsize=8, title="Codec family", title_fontsize=9)

    out = output_dir / out_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close(fig)


def _hbars_by_level(ax, codecs, colors, pivot, levels, title, xlabel, families=None, fmt="{:.1e}",
                     log_scale=True, value_range=None):
    n_levels = len(levels)
    height = 0.8 / n_levels
    y = np.arange(len(codecs))
    span = (value_range[1] - value_range[0]) if value_range else None
    shade_factors = _shade_factors(n_levels)
    hatch = [LOSSLESS_HATCH if f == "lossless" else None for f in families] if families is not None else None
    for i, level in enumerate(levels):
        offset = (i - (n_levels - 1) / 2) * height
        raw = pivot[level].values
        values = np.clip(raw, FLOOR, None) if log_scale else np.nan_to_num(raw, nan=0.0)
        level_colors = [_shade_color(c, shade_factors[i]) for c in colors]
        bars = ax.barh(y + offset, values, height=height, color=level_colors, hatch=hatch, zorder=3)
        for bar, v, true_v in zip(bars, values, raw):
            if log_scale:
                label_x, ha = bar.get_width() * 1.3, "left"
            else:
                pad = (span or abs(v) or 1) * 0.02
                label_x, ha = (bar.get_width() + pad, "left") if v >= 0 else (bar.get_width() - pad, "right")
            label = "0" if (log_scale and true_v <= FLOOR) else fmt.format(true_v)
            ax.text(
                label_x, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha, fontsize=5,
            )
    ax.set_yticks(y); ax.set_yticklabels(codecs, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlim(left=FLOOR / 2, right=ax.get_xlim()[1] * 20)
    else:
        if value_range is not None:
            ax.set_xlim(*value_range)
        # Widen xlim if any rendered label text would otherwise spill past the right edge --
        # the anchor x-position can be in-bounds while the text's own width pushes it past.
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        cur_left, cur_right = ax.get_xlim()
        max_right = cur_right
        for txt in ax.texts:
            x1_data = ax.transData.inverted().transform((txt.get_window_extent(renderer).x1, 0))[0]
            max_right = max(max_right, x1_data)
        if max_right > cur_right:
            ax.set_xlim(left=cur_left, right=cur_left + (max_right - cur_left) * 1.03)


def plot_accuracy_by_level(basin_df: pd.DataFrame, output_dir: Path,
                            title: str = "Water Balance Compression Accuracy by HydroBASIN Level",
                            out_name: str = "accuracy_by_basin_level.png") -> None:
    """Per-component reconstruction fidelity, normalized to % of precipitation (not the
    component's own magnitude -- total_runoff in particular is frequently near-zero at basin
    scale, which would blow up small absolute errors into meaningless percentages). Using the
    same denominator as plot_closure_fidelity keeps all four plots on a comparable scale."""
    codec_order = _codec_order(basin_df)
    codecs = codec_order["codec"].tolist()
    colors = [FAMILY_COLORS[f] for f in codec_order["family"]]
    levels = sorted(basin_df["level"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, metric, metric_title in [
        (axes[0], "mean_abs_bias_pct", "Mean |basin bias|  ↓"),
        (axes[1], "mean_mae_pct", "Mean basin MAE  ↓"),
        (axes[2], "p95_mae_pct", "p95 basin MAE  ↓"),
    ]:
        pivot = basin_df.pivot(index="codec", columns="level", values=metric).reindex(codecs)
        _hbars_by_level(ax, codecs, colors, pivot, levels, metric_title, "% of precipitation (log)",
                         families=codec_order["family"])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    families_present = set(codec_order["family"])
    family_patches = [Patch(color=c, label=f, hatch=LOSSLESS_HATCH if f == "lossless" else None) for f, c in FAMILY_COLORS.items() if f in families_present]
    fig.legend(handles=family_patches, loc="outside lower center",
               ncols=len(family_patches), fontsize=8, title="Codec family", title_fontsize=9)
    _add_level_colorbar(fig, axes, levels)

    out = output_dir / out_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close(fig)


def plot_closure_fidelity(closure_df: pd.DataFrame, output_dir: Path,
                           title: str = "Water Balance Compression — Closure Fidelity (% of precipitation)",
                           out_name: str = "closure_fidelity_by_basin_level.png") -> None:
    codec_order = _codec_order(closure_df)
    codecs = codec_order["codec"].tolist()
    colors = [FAMILY_COLORS[f] for f in codec_order["family"]]
    levels = sorted(closure_df["level"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    families = codec_order["family"]
    pivot_drift = closure_df.pivot(index="codec", columns="level", values="p95_abs_drift_pp").reindex(codecs)
    _hbars_by_level(axes[0], codecs, colors, pivot_drift, levels,
                     "p95 |closure drift|  ↓", "pp of P (log)", families=families, fmt="{:.1e}")

    pivot_rank = closure_df.pivot(index="codec", columns="level", values="rank_fidelity_spearman").reindex(codecs)
    _hbars_by_level(axes[1], codecs, colors, pivot_rank, levels,
                     "Basin rank fidelity (Spearman ρ)  ↑", "ρ", families=families, fmt="{:.3f}",
                     log_scale=False, value_range=(-1.05, 1.05))

    pivot_pass = closure_df.pivot(index="codec", columns="level", values="pct_basins_within_tol").reindex(codecs)
    _hbars_by_level(axes[2], codecs, colors, pivot_pass, levels,
                     "Basins within tolerance  ↑", "% of basins", families=families, fmt="{:.1f}",
                     log_scale=False, value_range=(0, 108))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    families_present = set(codec_order["family"])
    family_patches = [Patch(color=c, label=f, hatch=LOSSLESS_HATCH if f == "lossless" else None) for f, c in FAMILY_COLORS.items() if f in families_present]
    fig.legend(handles=family_patches, loc="outside lower center",
               ncols=len(family_patches), fontsize=8, title="Codec family", title_fontsize=9)
    _add_level_colorbar(fig, axes, levels)

    out = output_dir / out_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render accuracy plots for the water balance compression comparison")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Root dir; compression_results/ (containing benchmark_results.csv and basin_water_balance/) is derived from this")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_dir = args.output_dir / "output" / "compression_results"
    bench_csv = results_dir / "benchmark_results.csv"
    basin_water_balance_dir = results_dir / "basin_water_balance"
    component_basin_csv = basin_water_balance_dir / "basin_summary_stats_components.csv"
    closure_csv = basin_water_balance_dir / "closure_summary.csv"
    missing = [p for p in (bench_csv, component_basin_csv, closure_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result file(s): {missing}")

    plots_dir = results_dir / "compression_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_benchmark_stats(pd.read_csv(bench_csv), plots_dir)
    plot_closure_fidelity(pd.read_csv(closure_csv), plots_dir)

    component_basin_df = pd.read_csv(component_basin_csv)
    for component, letter in COMPONENT_LABELS.items():
        plot_accuracy_by_level(
            component_basin_df[component_basin_df["component"] == component],
            plots_dir,
            title=f"Water Balance Compression Accuracy by HydroBASIN Level — annual {letter} ({component})",
            out_name=f"accuracy_by_basin_level_{component}.png",
        )
