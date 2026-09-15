import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plots for fidelity_diagnostics.py: water-balance trend/drought/flood fidelity + benchmark speed/size")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


TREND_VARS = ["precipitation", "total_evaporation", "total_runoff", "storage_change"]
QQ_TAIL_PCT = 10  # quantiles below/above this are colored as the drought/flood-relevant tails
QQ_PROBS = np.linspace(0.001, 0.999, 300)


def export_water_balance_fidelity_html(monthly_sample: pd.DataFrame, output_path: Path) -> None:
    """Interactive Q-Q + trend viewer (SVG, wheel-zoom/drag-pan/hover, no CDN) for water-balance
    fidelity: central-estimate trend (top row) vs. whole-distribution Q-Q (bottom row), one column
    per variable. Quantiles are computed independently on reference and compressed (not paired by
    cell), so the Q-Q panel shows whether compression preserves distribution *shape* -- including
    low-bit quantization artifacts a static plot can't be zoomed into -- while tail coloring keeps
    the drought (low quantiles) / flood (high quantiles) framing visible."""
    candidates = monthly_sample["candidate"].unique()
    plot_candidate, other = candidates[0], candidates[1:]
    m = monthly_sample[monthly_sample["candidate"] == plot_candidate]

    trend_times = sorted(m["time"].unique())
    trend = {
        var: {
            "true": m.groupby("time")[f"{var}_true"].mean().reindex(trend_times).round(6).tolist(),
            "pred": m.groupby("time")[f"{var}_pred"].mean().reindex(trend_times).round(6).tolist(),
        }
        for var in TREND_VARS
    }
    qq = {
        var: {
            "true": np.quantile(m[f"{var}_true"].to_numpy(), QQ_PROBS).round(6).tolist(),
            "pred": np.quantile(m[f"{var}_pred"].to_numpy(), QQ_PROBS).round(6).tolist(),
        }
        for var in TREND_VARS
    }
    data = {
        "vars": TREND_VARS,
        "timeLabels": [pd.Timestamp(t).strftime("%Y-%m") for t in trend_times],
        "trend": trend,
        "qq": qq,
        "probs": QQ_PROBS.round(5).tolist(),
        "tailPct": QQ_TAIL_PCT,
        "candidate": plot_candidate,
        "otherCount": int(len(other)),
    }
    other_note = f"(other {len(other)} candidates decode identically, not shown.)" if len(other) else ""
    html = (_FIDELITY_VIEWER_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__CANDIDATE__", plot_candidate)
            .replace("__OTHER_NOTE__", other_note)
            .replace("__TAIL_PCT__", str(QQ_TAIL_PCT)))
    output_path.write_text(html)


def export_water_balance_fidelity_png(monthly_sample: pd.DataFrame, output_path: Path) -> None:
    """Static twin of export_water_balance_fidelity_html's trend/Q-Q grid (no zoom/pan/hover)."""
    candidates = monthly_sample["candidate"].unique()
    plot_candidate, other = candidates[0], candidates[1:]
    m = monthly_sample[monthly_sample["candidate"] == plot_candidate]
    tail_frac = QQ_TAIL_PCT / 100

    fig, axes = plt.subplots(2, len(TREND_VARS), figsize=(4.4 * len(TREND_VARS), 6.6))
    for col, var in enumerate(TREND_VARS):
        trend = m.groupby("time")[[f"{var}_true", f"{var}_pred"]].mean().sort_index()
        ax = axes[0, col]
        ax.plot(trend.index, trend[f"{var}_true"], color="black", marker="o", markersize=3, label="reference")
        ax.plot(trend.index, trend[f"{var}_pred"], color="#2563eb", linestyle="--", marker="o", markersize=3, label="compressed")
        ax.set_title(var.replace("_", " "))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        if col == 0:
            ax.set_ylabel("domain-mean monthly (m)")
            ax.legend(fontsize=8)

        true_q = np.quantile(m[f"{var}_true"].to_numpy(), QQ_PROBS)
        pred_q = np.quantile(m[f"{var}_pred"].to_numpy(), QQ_PROBS)
        tail = (QQ_PROBS <= tail_frac) | (QQ_PROBS >= 1 - tail_frac)
        ax2 = axes[1, col]
        ax2.scatter(true_q[~tail], pred_q[~tail], s=6, color="#2563eb", alpha=0.75, label="middle")
        ax2.scatter(true_q[tail], pred_q[tail], s=6, color="#dc2626", alpha=0.75, label=f"tail {QQ_TAIL_PCT}%")
        lo, hi = min(true_q.min(), pred_q.min()), max(true_q.max(), pred_q.max())
        ax2.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("reference quantile (m)")
        if col == 0:
            ax2.set_ylabel("compressed quantile (m)")
            ax2.legend(fontsize=8)

    note = f" ({len(other)} other candidates decode identically, not shown)" if len(other) else ""
    fig.suptitle(f"Water-balance fidelity -- {plot_candidate}{note}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_benchmark(timings: dict, output_path: Path, baseline: str = "lossless+zstd") -> None:
    labels = list(timings)
    panels = (
        ("size_bytes", "store size", "MB", 1e6),
        ("write_mbps", "write throughput", "MB/s", 1),
        ("read_mbps", "read throughput", "MB/s", 1),
    )
    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 9))
    for col, (key, title, ylabel, scale) in enumerate(panels):
        values = [timings[l][key] / scale for l in labels]
        bars = axes[0, col].bar(labels, values, color="tab:blue")
        axes[0, col].bar_label(bars, fmt="%.1f")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel(ylabel)

        # ratio to baseline: >1 = faster / larger than baseline, <1 = slower / smaller than
        # baseline in the same units the metric is already stored in (mbps, bytes) -- so
        # "half the size" -> 0.5 and "twice as fast" -> 2 both fall out of the same ratio.
        ratio = [timings[l][key] / timings[baseline][key] for l in labels]
        ratio_bars = axes[1, col].bar(labels, ratio, color="tab:orange")
        axes[1, col].bar_label(ratio_bars, fmt="%.2fx")
        axes[1, col].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axes[1, col].set_title(f"{title} relative to {baseline}")
        axes[1, col].set_ylabel("ratio")

        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def _png_b64(rgba: np.ndarray) -> str:
    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _colorize(data: np.ndarray, cmap_name: str, vmin: float, vmax: float, flip_v: bool, max_dim: int = 1200) -> str:
    step = max(1, max(data.shape) // max_dim)
    small = data[::step, ::step]
    if flip_v:
        small = small[::-1]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return _png_b64(cmap(norm(np.ma.masked_invalid(small))))


def export_compression_sweep_html(data_dir: Path, results_dir: Path) -> None:
    """Standalone interactive viewer (no server, no CDN) for the compression sweep: per
    variable/level, a zoomable/pannable map toggling between the rounded value and its
    difference from the unrounded reference. Static side-by-side panels made it hard to
    actually see fine-scale differences; this lets you zoom into a region and step through
    levels at the same zoom/pan."""
    meta = pd.DataFrame(json.loads((data_dir / "compression_sweep_meta.json").read_text()))
    fields = np.load(data_dir / "compression_sweep_fields.npz")
    lat, lon = fields["latitude"], fields["longitude"]
    extent_label = f"{lon.min():.1f}-{lon.max():.1f} deg E, {lat.min():.1f}-{lat.max():.1f} deg N"
    flip_v = bool(lat[0] < lat[-1])  # image row 0 must be north; flip if latitude is stored ascending

    data = {}
    for var, group in meta.groupby("variable", sort=False):
        group = group.sort_values("level")
        raw = fields[f"{var}__raw"]
        levels = [fields[f"{var}__{lvl}"] for lvl in group["level"]]
        diffs = [lvl - raw for lvl in levels]

        # robust limits: true min/max are dominated by rare extremes (e.g. flood-magnitude
        # runoff cells), which flattens the rest of the map to a single color under a linear scale
        v_stack = np.concatenate([a[np.isfinite(a)] for a in levels])
        vmin_v, vmax_v = (float(x) for x in np.nanpercentile(v_stack, [2, 98]))
        d_stack = np.concatenate([np.abs(d[np.isfinite(d)]) for d in diffs])
        vmax_d = float(np.nanpercentile(d_stack, 98)) if d_stack.size else 1e-9
        vmax_d = vmax_d if vmax_d > 0 else 1e-9

        data[var] = {
            "vmin_value": vmin_v, "vmax_value": vmax_v, "vmax_diff": vmax_d,
            "levels": [
                {
                    "inflevel": float(row["inflevel"]),
                    "png_value": _colorize(arr, "viridis", vmin_v, vmax_v, flip_v),
                    "png_diff": _colorize(diff, "RdBu_r", -vmax_d, vmax_d, flip_v),
                }
                for (_, row), arr, diff in zip(group.iterrows(), levels, diffs)
            ],
        }

    cb_value = _png_b64(plt.get_cmap("viridis")(np.tile(np.linspace(0, 1, 256), (20, 1))))
    cb_diff = _png_b64(plt.get_cmap("RdBu_r")(np.tile(np.linspace(0, 1, 256), (20, 1))))
    html = (_VIEWER_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__EXTENT_LABEL__", extent_label)
            .replace("__CB_VALUE__", cb_value)
            .replace("__CB_DIFF__", cb_diff))
    (results_dir / "compression_sweep_viewer.html").write_text(html)


_VIEWER_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Compression sweep viewer</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: #f7f7f8; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1c; color: #e8e8e8; }
    header, .controls { border-color: #333 !important; }
    .panel { background: #26262a !important; border-color: #3a3a3f !important; }
    button, select { background: #333 !important; color: #eee !important; border-color: #4a4a4f !important; }
    button.active { background: #3b6fd1 !important; border-color: #3b6fd1 !important; }
  }
  header { padding: 14px 24px; border-bottom: 1px solid #ddd; }
  header h1 { margin: 0 0 4px; font-size: 1.25rem; }
  header p { margin: 0; font-size: 0.82rem; opacity: 0.7; }
  .layout { display: flex; height: calc(100vh - 64px); }
  .controls { width: 270px; padding: 14px; border-right: 1px solid #ddd; overflow-y: auto; flex-shrink: 0; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .panel h3 { margin: 0 0 8px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.6; }
  .tabs { display: flex; flex-direction: column; gap: 4px; }
  button { border: 1px solid #ccc; background: #fff; padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; text-align: left; }
  button.active { background: #2563eb; color: #fff; border-color: #2563eb; }
  .mode-row { display: flex; gap: 4px; }
  .mode-row button { flex: 1; text-align: center; }
  input[type=range] { width: 100%; }
  .level-info { font-size: 0.8rem; line-height: 1.6; margin-top: 8px; }
  .level-info b { display: inline-block; min-width: 105px; opacity: 0.65; font-weight: 500; }
  .viewer { flex: 1; position: relative; overflow: hidden; background: #111; cursor: grab; }
  .viewer.dragging { cursor: grabbing; }
  .viewer img { position: absolute; top: 0; left: 0; transform-origin: 0 0; image-rendering: pixelated; user-select: none; -webkit-user-drag: none; }
  .colorbar img { width: 100%; height: 14px; border-radius: 3px; border: 1px solid #ccc; display: block; }
  .cb-labels { display: flex; justify-content: space-between; font-size: 0.7rem; opacity: 0.7; margin-top: 4px; }
  .reset-btn { width: 100%; }
</style>
</head>
<body>
<header>
  <h1>Compression sweep viewer</h1>
  <p>Bitround+zstd at increasing information levels, single-month snapshot (__EXTENT_LABEL__). Scroll to zoom, drag to pan -- zoom/pan is kept when switching level/variable/mode.</p>
</header>
<div class="layout">
  <div class="controls">
    <div class="panel"><h3>Variable</h3><div class="tabs" id="varTabs"></div></div>
    <div class="panel">
      <h3>View</h3>
      <div class="mode-row">
        <button id="modeDiff" class="active">Difference</button>
        <button id="modeValue">Value</button>
      </div>
    </div>
    <div class="panel">
      <h3>Information level</h3>
      <input type="range" id="levelSlider" min="0" max="5" value="5" step="1">
      <div class="level-info" id="levelInfo"></div>
    </div>
    <div class="panel">
      <h3>Color scale</h3>
      <div class="colorbar"><img id="cbImg" src=""></div>
      <div class="cb-labels"><span id="cbMin"></span><span id="cbMid"></span><span id="cbMax"></span></div>
    </div>
    <button class="reset-btn" id="resetBtn">Reset view</button>
  </div>
  <div class="viewer" id="viewer"><img id="mapImg" src=""></div>
</div>
<script>
const DATA = __DATA__;
const CB_VALUE = "data:image/png;base64,__CB_VALUE__";
const CB_DIFF = "data:image/png;base64,__CB_DIFF__";

let currentVar = Object.keys(DATA)[0];
let currentMode = "diff";
let currentLevel = DATA[currentVar].levels.length - 1;

const varTabs = document.getElementById("varTabs");
Object.keys(DATA).forEach(v => {
  const b = document.createElement("button");
  b.textContent = v;
  b.dataset.var = v;
  b.onclick = () => {
    currentVar = v;
    currentLevel = Math.min(currentLevel, DATA[v].levels.length - 1);
    slider.max = DATA[v].levels.length - 1;
    slider.value = currentLevel;
    updateTabs();
    render();
  };
  varTabs.appendChild(b);
});
function updateTabs() {
  [...varTabs.children].forEach(b => b.classList.toggle("active", b.dataset.var === currentVar));
}
updateTabs();

const modeDiffBtn = document.getElementById("modeDiff");
const modeValueBtn = document.getElementById("modeValue");
modeDiffBtn.onclick = () => { currentMode = "diff"; modeDiffBtn.classList.add("active"); modeValueBtn.classList.remove("active"); render(); };
modeValueBtn.onclick = () => { currentMode = "value"; modeValueBtn.classList.add("active"); modeDiffBtn.classList.remove("active"); render(); };

const slider = document.getElementById("levelSlider");
slider.max = DATA[currentVar].levels.length - 1;
slider.value = currentLevel;
slider.oninput = () => { currentLevel = +slider.value; render(); };

const img = document.getElementById("mapImg");
const levelInfo = document.getElementById("levelInfo");
const cbImg = document.getElementById("cbImg");
const cbMin = document.getElementById("cbMin");
const cbMid = document.getElementById("cbMid");
const cbMax = document.getElementById("cbMax");

function render() {
  const v = DATA[currentVar];
  const lvl = v.levels[currentLevel];
  img.src = "data:image/png;base64," + (currentMode === "diff" ? lvl.png_diff : lvl.png_value);
  const infoPct = Number((lvl.inflevel * 100).toPrecision(6));
  levelInfo.innerHTML = "<div><b>Info retained</b>&ge; " + infoPct + "%</div>";
  if (currentMode === "diff") {
    cbImg.src = CB_DIFF;
    cbMin.textContent = (-v.vmax_diff).toExponential(2);
    cbMid.textContent = "0";
    cbMax.textContent = v.vmax_diff.toExponential(2);
  } else {
    cbImg.src = CB_VALUE;
    cbMin.textContent = v.vmin_value.toPrecision(3);
    cbMid.textContent = ((v.vmin_value + v.vmax_value) / 2).toPrecision(3);
    cbMax.textContent = v.vmax_value.toPrecision(3);
  }
}
render();

const viewer = document.getElementById("viewer");
let scale = 1, panX = 0, panY = 0, dragging = false, lastX = 0, lastY = 0, initialized = false;

function applyTransform() { img.style.transform = "translate(" + panX + "px, " + panY + "px) scale(" + scale + ")"; }
function fitToViewport() {
  const vw = viewer.clientWidth, vh = viewer.clientHeight;
  const iw = img.naturalWidth, ih = img.naturalHeight;
  if (!iw || !ih) return;
  scale = Math.min(vw / iw, vh / ih);
  panX = (vw - iw * scale) / 2;
  panY = (vh - ih * scale) / 2;
  applyTransform();
}
img.onload = () => { if (!initialized) { initialized = true; fitToViewport(); } };

viewer.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = viewer.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newScale = Math.min(Math.max(scale * factor, 0.05), 60);
  panX = mx - (mx - panX) * (newScale / scale);
  panY = my - (my - panY) * (newScale / scale);
  scale = newScale;
  applyTransform();
}, { passive: false });
viewer.addEventListener("mousedown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; viewer.classList.add("dragging"); });
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  panX += e.clientX - lastX; panY += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  applyTransform();
});
window.addEventListener("mouseup", () => { dragging = false; viewer.classList.remove("dragging"); });
document.getElementById("resetBtn").onclick = fitToViewport;
</script>
</body>
</html>
"""


_FIDELITY_VIEWER_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Water-balance fidelity viewer</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background: #f7f7f8; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1c; color: #e8e8e8; }
    header, .cell { border-color: #333 !important; }
    button { background: #333 !important; color: #eee !important; border-color: #4a4a4f !important; }
  }
  header { padding: 14px 24px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: baseline; }
  header h1 { margin: 0 0 4px; font-size: 1.25rem; }
  header p { margin: 0; font-size: 0.82rem; opacity: 0.7; }
  button { border: 1px solid #ccc; background: #fff; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  .rowlabel { padding: 10px 24px 0; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.6; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 10px 24px 24px; }
  .cell { border: 1px solid #ddd; border-radius: 8px; padding: 8px; }
  .cell h3 { margin: 0 0 4px; font-size: 0.85rem; font-weight: 500; }
  svg { width: 100%; height: auto; touch-action: none; }
  #tooltip { position: fixed; display: none; background: #222; color: #fff; font-size: 0.78rem; padding: 6px 8px; border-radius: 5px; pointer-events: none; z-index: 10; line-height: 1.4; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Water-balance fidelity -- __CANDIDATE__</h1>
    <p>Central-estimate trend (top) vs. distribution Q-Q (bottom). Scroll to zoom, drag to pan, hover for values. __OTHER_NOTE__</p>
  </div>
  <button id="resetAll">Reset all</button>
</header>
<div class="rowlabel">Domain-mean monthly trend (m)</div>
<div class="grid" id="trendGrid"></div>
<div class="rowlabel">Quantile-quantile (m) -- red = bottom/top __TAIL_PCT__% (drought/flood range)</div>
<div class="grid" id="qqGrid"></div>
<div id="tooltip"></div>
<script>
const DATA = __DATA__;
const VARS = DATA.vars;
const NS = "http://www.w3.org/2000/svg";

function scale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const s = v => r0 + (d1 === d0 ? 0 : (v - d0) / (d1 - d0)) * (r1 - r0);
  s.domain = domain; s.range = range;
  s.invert = p => d0 + (r1 === r0 ? 0 : (p - r0) / (r1 - r0)) * (d1 - d0);
  return s;
}
function niceStep(span, targetCount) {
  const raw = span / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
}
function niceTicks(domain, targetCount) {
  const [a, b] = domain;
  if (b <= a) return [a];
  const step = niceStep(b - a, targetCount);
  const ticks = [];
  for (let t = Math.ceil(a / step) * step; t <= b + step * 1e-9; t += step) ticks.push(+t.toPrecision(10));
  return ticks;
}
function niceIntTicks(domain, targetCount) {
  const [a, b] = domain;
  const step = Math.max(1, Math.round(Math.max(1, b - a) / targetCount));
  const ticks = [];
  for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) ticks.push(t);
  return ticks;
}
function fmt(v) {
  if (Math.abs(v) < 1e-9) return "0";
  return Math.abs(v) < 0.001 || Math.abs(v) >= 1000 ? v.toExponential(2) : v.toPrecision(3);
}
function el(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function drawAxes(svg, x, y, xTicksFn, xFmt) {
  const [xr0, xr1] = x.range, [yr0, yr1] = y.range;
  svg.appendChild(el("line", { x1: xr0, y1: yr0, x2: xr1, y2: yr0, stroke: "currentColor", "stroke-opacity": 0.4 }));
  svg.appendChild(el("line", { x1: xr0, y1: yr0, x2: xr0, y2: yr1, stroke: "currentColor", "stroke-opacity": 0.4 }));
  (xTicksFn || niceTicks)(x.domain, 6).forEach(t => {
    const px = x(t);
    if (px < xr0 - 1 || px > xr1 + 1) return;
    svg.appendChild(el("line", { x1: px, y1: yr0, x2: px, y2: yr0 + 4, stroke: "currentColor", "stroke-opacity": 0.4 }));
    const label = el("text", { x: px, y: yr0 + 15, "font-size": 9, "text-anchor": "middle", fill: "currentColor", opacity: 0.7 });
    label.textContent = xFmt ? xFmt(t) : fmt(t);
    svg.appendChild(label);
  });
  niceTicks(y.domain, 6).forEach(t => {
    const py = y(t);
    if (py > yr0 + 1 || py < yr1 - 1) return;
    svg.appendChild(el("line", { x1: xr0 - 4, y1: py, x2: xr0, y2: py, stroke: "currentColor", "stroke-opacity": 0.4 }));
    const label = el("text", { x: xr0 - 7, y: py + 3, "font-size": 9, "text-anchor": "end", fill: "currentColor", opacity: 0.7 });
    label.textContent = fmt(t);
    svg.appendChild(label);
  });
}

const tooltip = document.getElementById("tooltip");
function showTip(cx, cy, html) {
  tooltip.style.display = "block";
  tooltip.style.left = (cx + 14) + "px";
  tooltip.style.top = (cy + 14) + "px";
  tooltip.innerHTML = html;
}
function hideTip() { tooltip.style.display = "none"; }

const allPanels = [];
class Panel {
  constructor(svg, xDomain, yDomain, draw, xTicksFn, xFmt) {
    this.svg = svg;
    this.xDomain0 = xDomain.slice(); this.yDomain0 = yDomain.slice();
    this.xDomain = xDomain.slice(); this.yDomain = yDomain.slice();
    this.W = +svg.getAttribute("width"); this.H = +svg.getAttribute("height");
    this.pad = { l: 52, r: 8, t: 8, b: 22 };
    this.draw = draw; this.xTicksFn = xTicksFn; this.xFmt = xFmt;
    this.hitTest = null;
    this.render();
    this._wire();
    allPanels.push(this);
  }
  scales() {
    return {
      x: scale(this.xDomain, [this.pad.l, this.W - this.pad.r]),
      y: scale(this.yDomain, [this.H - this.pad.b, this.pad.t]),
    };
  }
  render() {
    while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);
    const { x, y } = this.scales();
    drawAxes(this.svg, x, y, this.xTicksFn, this.xFmt);
    this.hitTest = this.draw(this.svg, x, y) || null;
  }
  reset() { this.xDomain = this.xDomain0.slice(); this.yDomain = this.yDomain0.slice(); this.render(); }
  _wire() {
    const svg = this.svg;
    svg.style.cursor = "grab";
    svg.addEventListener("wheel", e => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const scaleFactor = this.W / rect.width;
      const { x, y } = this.scales();
      const px = (e.clientX - rect.left) * scaleFactor, py = (e.clientY - rect.top) * scaleFactor;
      const dx0 = x.invert(px), dy0 = y.invert(py);
      const f = e.deltaY < 0 ? 0.85 : 1 / 0.85;
      this.xDomain = [dx0 + (this.xDomain[0] - dx0) * f, dx0 + (this.xDomain[1] - dx0) * f];
      this.yDomain = [dy0 + (this.yDomain[0] - dy0) * f, dy0 + (this.yDomain[1] - dy0) * f];
      this.render();
    }, { passive: false });
    let dragging = false, lastX = 0, lastY = 0;
    svg.addEventListener("mousedown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; svg.style.cursor = "grabbing"; });
    window.addEventListener("mousemove", e => {
      const rect = svg.getBoundingClientRect();
      const scaleFactor = this.W / rect.width;
      if (dragging) {
        const { x, y } = this.scales();
        const dx = (e.clientX - lastX) * scaleFactor / (x.range[1] - x.range[0]) * (x.domain[1] - x.domain[0]);
        const dy = (e.clientY - lastY) * scaleFactor / (y.range[1] - y.range[0]) * (y.domain[1] - y.domain[0]);
        this.xDomain = [this.xDomain[0] - dx, this.xDomain[1] - dx];
        this.yDomain = [this.yDomain[0] - dy, this.yDomain[1] - dy];
        lastX = e.clientX; lastY = e.clientY;
        this.render();
        hideTip();
        return;
      }
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;
      if (this.hitTest) {
        const hit = this.hitTest((e.clientX - rect.left) * scaleFactor, (e.clientY - rect.top) * scaleFactor);
        if (hit) showTip(e.clientX, e.clientY, hit); else hideTip();
      }
    });
    window.addEventListener("mouseup", () => { dragging = false; svg.style.cursor = "grab"; });
    svg.addEventListener("mouseleave", () => { if (!dragging) hideTip(); });
  }
}

function pad(lo, hi, frac) { const m = (hi - lo) * frac || 1; return [lo - m, hi + m]; }

function makeTrendPanel(container, varName) {
  const t = DATA.trend[varName];
  const n = t.true.length;
  const allV = t.true.concat(t.pred);
  const svg = el("svg", { width: 440, height: 300 });
  const cellDiv = document.createElement("div");
  cellDiv.className = "cell";
  const h3 = document.createElement("h3"); h3.textContent = varName.replace(/_/g, " ");
  cellDiv.appendChild(h3); cellDiv.appendChild(svg); container.appendChild(cellDiv);

  new Panel(svg, [-0.5, n - 0.5], pad(Math.min(...allV), Math.max(...allV), 0.08), (svg, x, y) => {
    const path = (arr) => arr.map((v, i) => (i === 0 ? "M" : "L") + x(i) + "," + y(v)).join(" ");
    svg.appendChild(el("path", { d: path(t.true), fill: "none", stroke: "currentColor", "stroke-width": 1.5 }));
    svg.appendChild(el("path", { d: path(t.pred), fill: "none", stroke: "#2563eb", "stroke-width": 1.5, "stroke-dasharray": "4,3" }));
    const pts = [];
    for (let i = 0; i < n; i++) {
      const cx = x(i);
      svg.appendChild(el("circle", { cx, cy: y(t.true[i]), r: 2.5, fill: "currentColor" }));
      svg.appendChild(el("circle", { cx, cy: y(t.pred[i]), r: 2.5, fill: "#2563eb" }));
      pts.push({ px: cx, i });
    }
    return (mx) => {
      let best = null, bd = Infinity;
      pts.forEach(p => { const d = Math.abs(p.px - mx); if (d < bd) { bd = d; best = p; } });
      if (!best || bd > 25) return null;
      const i = best.i;
      return `<b>${DATA.timeLabels[i]}</b><br>reference: ${fmt(t.true[i])} m<br>compressed: ${fmt(t.pred[i])} m`;
    };
  }, niceIntTicks, t2 => DATA.timeLabels[Math.round(t2)] ?? "");
}

function makeQQPanel(container, varName) {
  const q = DATA.qq[varName];
  const probs = DATA.probs;
  const tailFrac = DATA.tailPct / 100;
  const allV = q.true.concat(q.pred);
  const [lo, hi] = pad(Math.min(...allV), Math.max(...allV), 0.05);
  const svg = el("svg", { width: 440, height: 300 });
  const cellDiv = document.createElement("div");
  cellDiv.className = "cell";
  const h3 = document.createElement("h3"); h3.textContent = varName.replace(/_/g, " ");
  cellDiv.appendChild(h3); cellDiv.appendChild(svg); container.appendChild(cellDiv);

  new Panel(svg, [lo, hi], [lo, hi], (svg, x, y) => {
    const dlo = Math.min(x.domain[0], y.domain[0]), dhi = Math.max(x.domain[1], y.domain[1]);
    svg.appendChild(el("line", { x1: x(dlo), y1: y(dlo), x2: x(dhi), y2: y(dhi), stroke: "currentColor", "stroke-opacity": 0.4, "stroke-dasharray": "4,3" }));
    const pts = [];
    for (let i = 0; i < probs.length; i++) {
      const tail = probs[i] <= tailFrac || probs[i] >= 1 - tailFrac;
      const cx = x(q.true[i]), cy = y(q.pred[i]);
      svg.appendChild(el("circle", { cx, cy, r: 2.5, fill: tail ? "#dc2626" : "#2563eb", "fill-opacity": 0.75 }));
      pts.push({ px: cx, py: cy, i });
    }
    return (mx, my) => {
      let best = null, bd = Infinity;
      pts.forEach(p => { const d = Math.hypot(p.px - mx, p.py - my); if (d < bd) { bd = d; best = p; } });
      if (!best || bd > 20) return null;
      const i = best.i;
      return `<b>p${(probs[i] * 100).toFixed(1)}</b><br>reference: ${fmt(q.true[i])} m<br>compressed: ${fmt(q.pred[i])} m`;
    };
  });
}

const trendGrid = document.getElementById("trendGrid");
const qqGrid = document.getElementById("qqGrid");
VARS.forEach(v => makeTrendPanel(trendGrid, v));
VARS.forEach(v => makeQQPanel(qqGrid, v));
document.getElementById("resetAll").onclick = () => allPanels.forEach(p => p.reset());
</script>
</body>
</html>
"""


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.output_dir / "output" / "compression_data"
    results_dir = args.output_dir / "output" / "compression_results"

    monthly_sample = pd.read_csv(results_dir / "fidelity_monthly_cell_sample.csv", parse_dates=["time"])
    export_water_balance_fidelity_html(monthly_sample, results_dir / "water_balance_fidelity.html")
    export_water_balance_fidelity_png(monthly_sample, results_dir / "water_balance_fidelity.png")

    timings = json.loads((data_dir / "benchmark_timings.json").read_text())
    plot_benchmark(timings, results_dir / "benchmark_speed.png")

    export_compression_sweep_html(data_dir, results_dir)

    print(f"plot_fidelity.py finished: {results_dir}")
