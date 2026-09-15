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

import dask
import numpy as np
import pandas as pd
import xarray as xr

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("zarr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

COMPONENT_NAMES = ["precipitation", "total_evaporation", "total_runoff"]
CANDIDATE_LABELS = ["bitround-informed+zstd", "bitround-informed+pco", "bitround-informed+blosc"]
CHUNK_DICT = {"time": 2, "latitude": 1280, "longitude": 1200}
CELL_SAMPLE_N = 20_000  # neighbouring cells are highly correlated; a random subset covers the tails without every row being near-duplicate
CELL_SAMPLE_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample cell-level monthly time series: compressed archive vs uncompressed reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sample_cells(true_ds: xr.Dataset, pred_ds: xr.Dataset, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cells valid (non-NaN annual total) in all 3 vars, for both true and pred -- a spatially
    consistent set the monthly time series below can be closed over."""
    true_annual = {v: true_ds[v].sum("time", skipna=False).compute().values for v in COMPONENT_NAMES}
    pred_annual = {v: pred_ds[v].sum("time", skipna=False).compute().values for v in COMPONENT_NAMES}
    valid = {v: ~np.isnan(true_annual[v]) & ~np.isnan(pred_annual[v]) for v in COMPONENT_NAMES}
    cell_valid = valid["precipitation"] & valid["total_evaporation"] & valid["total_runoff"]
    ila, ilo = np.nonzero(cell_valid)
    sample = rng.choice(ila.size, size=min(CELL_SAMPLE_N, ila.size), replace=False)
    return ila[sample], ilo[sample]


def monthly_cell_sample(true_ds: xr.Dataset, pred_ds: xr.Dataset, ila: np.ndarray, ilo: np.ndarray,
                         time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly time series at the sampled cells, wide by variable -- feeds the seasonal-trend and
    drought/flood magnitude comparisons in plot_fidelity.py. cell_valid (all 3 vars valid annually
    under skipna=False) guarantees every month is non-NaN here, so no dropna is needed."""
    lat_idx, lon_idx = xr.DataArray(ila, dims="cell"), xr.DataArray(ilo, dims="cell")
    true_pt = true_ds[COMPONENT_NAMES].isel(latitude=lat_idx, longitude=lon_idx).compute()
    pred_pt = pred_ds[COMPONENT_NAMES].isel(latitude=lat_idx, longitude=lon_idx).compute()
    nt, ncell = len(time_index), len(ila)
    df = pd.DataFrame({"time": np.repeat(time_index, ncell)})
    for var in COMPONENT_NAMES:
        df[f"{var}_true"] = true_pt[var].values.ravel()
        df[f"{var}_pred"] = pred_pt[var].values.ravel()
    df["storage_change_true"] = df["precipitation_true"] - df["total_evaporation_true"] - df["total_runoff_true"]
    df["storage_change_pred"] = df["precipitation_pred"] - df["total_evaporation_pred"] - df["total_runoff_pred"]
    return df


if __name__ == "__main__":
    args = parse_args()
    dask.config.set(scheduler="threads", num_workers=len(os.sched_getaffinity(0)))

    data_dir = args.output_dir / "output" / "compression_data"
    results_dir = args.output_dir / "output" / "compression_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ref_ds = xr.open_zarr(data_dir / "deploy_no-compression.zarr", consolidated=True).chunk(CHUNK_DICT)
    time_index = ref_ds["time"].values

    rng = np.random.default_rng(CELL_SAMPLE_SEED)
    monthly_sample_rows = []
    for candidate in CANDIDATE_LABELS:
        pred_ds = xr.open_zarr(data_dir / f"deploy_{candidate}.zarr", consolidated=True).chunk(CHUNK_DICT)
        ila, ilo = sample_cells(ref_ds, pred_ds, rng)
        monthly_sample = monthly_cell_sample(ref_ds, pred_ds, ila, ilo, time_index)
        monthly_sample.insert(0, "candidate", candidate)
        monthly_sample_rows.append(monthly_sample)

    pd.concat(monthly_sample_rows, ignore_index=True).to_csv(results_dir / "fidelity_monthly_cell_sample.csv", index=False)

    print(f"fidelity_diagnostics.py finished: {results_dir}")
