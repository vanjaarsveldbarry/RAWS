import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.stats import spearmanr

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("zarr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


COMPONENT_NAMES = ["precipitation", "total_evaporation", "total_runoff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare water-balance accuracy loss across benchmarked Zarr v3 compression codecs")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Root dir; compression_data/, compression_results/, and hydrobasins/processed/ subfolders are derived from this")
    parser.add_argument("--levels", type=str, required=True,
                         help="HydroBASINS levels to aggregate, e.g. '1-12' or '5,8,12'")
    parser.add_argument("--closure-tol-pp", type=float, required=True,
                         help="Closure-drift tolerance in percentage points of precipitation, "
                              "used for the pct_basins_within_tol pass rate")
    return parser.parse_args()


def water_balance_fields(ds: xr.Dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-pixel annual sum of each component, plus the water balance residual: precipitation - evaporation - runoff."""
    components = {name: ds[name].sum("time", skipna=False).values for name in COMPONENT_NAMES}
    wb = components["precipitation"] - components["total_evaporation"] - components["total_runoff"]
    return wb, components


def get_grid_transform(ds: xr.Dataset):
    """Derive an affine transform + shape from a dataset's lat/lon-like coords, whatever they're named."""
    y_name = "latitude" if "latitude" in ds.coords else "lat"
    x_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[y_name].values
    lon = ds[x_name].values

    yres = float(lat[0] - lat[1])
    xres = float(lon[1] - lon[0])
    west = float(lon[0]) - xres / 2
    north = float(lat[0]) + yres / 2

    transform = from_origin(west, north, xres, yres)
    shape = (len(lat), len(lon))
    bounds = (float(lon.min()) - xres / 2, float(lat.min()) - yres / 2,
              float(lon.max()) + xres / 2, float(lat.max()) + yres / 2)
    return transform, shape, bounds


def basin_mean_count(zone_flat: np.ndarray, mask_flat: np.ndarray, values_flat: np.ndarray, n_basins: int) -> tuple[np.ndarray, np.ndarray]:
    mask_flat = mask_flat & (zone_flat >= 0)
    z = zone_flat[mask_flat]
    v = values_flat[mask_flat]
    counts = np.bincount(z, minlength=n_basins)
    sums = np.bincount(z, weights=v, minlength=n_basins)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return means, counts


def closure_ratio_pct(wb_means: np.ndarray, precip_means: np.ndarray) -> np.ndarray:
    """Basin closure expressed as % of precipitation: mean(delta_S)/mean(P) x 100."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(precip_means > 0, wb_means / precip_means * 100, np.nan)


def relative_pct(values: np.ndarray, true_means: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Express a per-basin error metric as % of that basin's own true magnitude for the variable."""
    denom = np.abs(true_means)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > eps, values / denom * 100, np.nan)


def component_error_row(data: dict, mask_flat: np.ndarray, err_flat: np.ndarray, label: str, family: str,
                         level: int, component: str, precip_mean: np.ndarray) -> dict:
    """Basin bias/MAE as % of basin-mean precipitation -- same denominator plot_closure_fidelity uses."""
    mask_flat = mask_flat & (data["zone_flat"] >= 0)
    z = data["zone_flat"][mask_flat]
    e = err_flat[mask_flat]
    counts = np.bincount(z, minlength=data["n_basins"])
    sum_e = np.bincount(z, weights=e, minlength=data["n_basins"])
    sum_ae = np.bincount(z, weights=np.abs(e), minlength=data["n_basins"])
    with np.errstate(invalid="ignore"):
        bias_pct = relative_pct(np.where(counts > 0, sum_e / counts, np.nan), precip_mean)
        mae_pct = relative_pct(np.where(counts > 0, sum_ae / counts, np.nan), precip_mean)
    return {
        "level": level, "codec": label, "component": component, "family": family,
        "mean_abs_bias_pct": np.nanmean(np.abs(bias_pct)),
        "mean_mae_pct": np.nanmean(mae_pct),
        "p95_mae_pct": np.nanpercentile(mae_pct, 95),
    }


def geometry_metrics(label: str, family: str, pred: np.ndarray, true_components: dict, pred_components: dict,
                      valid_mask_flat: np.ndarray, valid_mask_components: dict, level_data: dict,
                      closure_tol_pp: float) -> tuple[list, list]:
    """Basin-aggregated component accuracy and water-balance closure drift for one codec, per HydroBASINS level.
    Returns (component_summary_rows, closure_summary_rows)."""
    component_summary_rows, closure_summary_rows = [], []

    for level, data in level_data.items():
        pred_basin_wb, _ = basin_mean_count(data["zone_flat"], valid_mask_flat, pred.ravel(), data["n_basins"])
        pred_basin_precip, _ = basin_mean_count(
            data["zone_flat"], valid_mask_components["precipitation"].ravel(),
            pred_components["precipitation"].ravel(), data["n_basins"],
        )
        closure_pred_pct = closure_ratio_pct(pred_basin_wb, pred_basin_precip)
        drift_pp = closure_pred_pct - data["closure_true_pct"]

        closure_valid = np.isfinite(drift_pp) & (data["n_pixels_true"] > 0)
        abs_drift = np.abs(drift_pp[closure_valid])
        rank_fidelity = (
            spearmanr(data["closure_true_pct"][closure_valid], closure_pred_pct[closure_valid])[0]
            if closure_valid.sum() >= 2 else np.nan
        )
        closure_summary_rows.append({
            "level": level, "codec": label, "family": family,
            "p95_abs_drift_pp": np.percentile(abs_drift, 95) if abs_drift.size else np.nan,
            "rank_fidelity_spearman": rank_fidelity,
            "pct_basins_within_tol": 100 * np.mean(abs_drift < closure_tol_pp) if abs_drift.size else np.nan,
        })

    for component in COMPONENT_NAMES:
        vm_flat_c = valid_mask_components[component].ravel()
        err_flat_c = (pred_components[component] - true_components[component]).ravel()
        for level, data in level_data.items():
            component_summary_rows.append(component_error_row(
                data, vm_flat_c, err_flat_c, label, family, level, component,
                data["component_basin_true"]["precipitation"][0],
            ))

    return component_summary_rows, closure_summary_rows


if __name__ == "__main__":
    args = parse_args()
    zarr_dir = args.output_dir / "output" / "compression_data"
    results_dir = args.output_dir / "output" / "compression_results"
    hydrobasins_dir = args.output_dir / "input" / "hydrobasins" / "processed"
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_df = pd.read_csv(results_dir / "benchmark_results.csv")

    ref_ds = xr.open_zarr(zarr_dir / "bench_no-compression.zarr", consolidated=True).compute()
    true, true_components = water_balance_fields(ref_ds)
    valid_mask_components = {c: ~np.isnan(v) for c, v in true_components.items()}
    valid_mask_flat = (~np.isnan(true)).ravel()
    transform, shape, bounds = get_grid_transform(ref_ds)

    if "-" in args.levels:
        lo, hi = args.levels.split("-")
        levels = list(range(int(lo), int(hi) + 1))
    else:
        levels = [int(x) for x in args.levels.split(",")]

    basin_save_dir = results_dir / "basin_water_balance"
    basin_save_dir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = bounds
    level_data = {}
    for level in levels:
        gdf = gpd.read_parquet(hydrobasins_dir / f"hybas_global_lev{level:02d}.parquet").cx[xmin:xmax, ymin:ymax].reset_index(drop=True)
        n_basins = len(gdf)
        zone_flat = rasterize(
            list(zip(gdf.geometry, np.arange(n_basins, dtype="int32"))),
            out_shape=shape, transform=transform, fill=-1, dtype="int32",
        ).ravel()
        basin_wb_true, n_pixels_true = basin_mean_count(zone_flat, valid_mask_flat, true.ravel(), n_basins)
        component_basin_true = {
            c: basin_mean_count(zone_flat, valid_mask_components[c].ravel(), true_components[c].ravel(), n_basins)
            for c in COMPONENT_NAMES
        }
        closure_true_pct = closure_ratio_pct(basin_wb_true, component_basin_true["precipitation"][0])
        level_data[level] = {
            "zone_flat": zone_flat, "n_basins": n_basins,
            "n_pixels_true": n_pixels_true,
            "component_basin_true": component_basin_true,
            "closure_true_pct": closure_true_pct,
        }

    component_summary_rows, closure_summary_rows = [], []
    for _, bench_row in bench_df.iterrows():
        label, family = bench_row["codec"], bench_row["family"]
        pred_ds = xr.open_zarr(zarr_dir / f"bench_{label}.zarr", consolidated=True).compute()
        pred, pred_components = water_balance_fields(pred_ds)

        csr, clr = geometry_metrics(label, family, pred, true_components, pred_components,
                                     valid_mask_flat, valid_mask_components, level_data, args.closure_tol_pp)
        component_summary_rows += csr
        closure_summary_rows += clr

    pd.DataFrame(component_summary_rows).to_csv(basin_save_dir / "basin_summary_stats_components.csv", index=False)
    pd.DataFrame(closure_summary_rows).to_csv(basin_save_dir / "closure_summary.csv", index=False)
