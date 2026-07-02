import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"

from pathlib import Path
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import imagecodecs.numcodecs as icn
icn.register_codecs()

from zarr.codecs.numcodecs._codecs import _NumcodecsArrayBytesCodec as _ABCodec
from zarr.registry import register_codec


class _SZ3Codec(_ABCodec, codec_name="imagecodecs_sz3"):
    pass


register_codec("numcodecs.imagecodecs_sz3", _SZ3Codec)

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("zarr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare accuracy loss across benchmarked Zarr v3 compression codecs")
    parser.add_argument("--input-dir",  type=Path, required=True, help="Unused, accepted for CLI parity with the other pipeline steps")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing bench_*.zarr stores and benchmark_results.csv")
    parser.add_argument("--temp-dir",   type=Path, required=True, help="Unused, accepted for CLI parity with the other pipeline steps")
    return parser.parse_args()


def load_reference(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    store_path = output_dir / "bench_no-compression.zarr"
    ds = xr.open_zarr(store_path, consolidated=True).compute()
    true = ds["discharge"].values
    valid_mask = ~np.isnan(true)
    return true, valid_mask


def load_codec_values(output_dir: Path, label: str) -> np.ndarray:
    store_path = output_dir / f"bench_{label}.zarr"
    ds = xr.open_zarr(store_path, consolidated=True).compute()
    return ds["discharge"].values


def compute_metrics(true: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> dict:
    t = true[valid_mask]
    p = pred[valid_mask]
    err = p - t
    abs_err = np.abs(err)

    zero_mask = t == 0
    nonzero_mask = ~zero_mask
    t_nz = t[nonzero_mask]
    p_nz = p[nonzero_mask]
    err_nz = p_nz - t_nz

    zero_violations = np.abs(p[zero_mask]) > 0
    zero_violation_frac = float(zero_violations.mean()) if zero_mask.any() else 0.0
    zero_violation_max = float(np.abs(p[zero_mask]).max()) if zero_violations.any() else 0.0

    low_thresh = np.quantile(t_nz, 0.10)
    high_thresh = np.quantile(t_nz, 0.90)
    low_mask = t_nz <= low_thresh
    high_mask = t_nz >= high_thresh

    # low, nonzero true flows that compression rounds down to exactly zero --
    # i.e. real low-flow signal that gets erased outright, not just distorted.
    low_flow_zeroed = p_nz[low_mask] == 0
    low_flow_zeroed_frac = float(low_flow_zeroed.mean()) if low_mask.any() else 0.0
    low_flow_zeroed_max_true = float(t_nz[low_mask][low_flow_zeroed].max()) if low_flow_zeroed.any() else 0.0

    def mae_of(mask_err):
        return float(np.mean(np.abs(mask_err)))

    def mape_of(err_subset, t_subset):
        return float(np.mean(np.abs(err_subset) / np.abs(t_subset)) * 100)

    return {
        "mae":                      round(float(np.mean(abs_err)), 6),
        "max_abs_err":              round(float(np.max(abs_err)), 6),
        "bias":                     round(float(np.mean(err)), 6),
        "zero_violation_frac":      round(zero_violation_frac, 6),
        "zero_violation_max":       round(zero_violation_max, 6),
        "mae_nonzero":              round(mae_of(err_nz), 6),
        "mape_nonzero":             round(mape_of(err_nz, t_nz), 4),
        "mae_low_flow":             round(mae_of(err_nz[low_mask]), 6),
        "mape_low_flow":            round(mape_of(err_nz[low_mask], t_nz[low_mask]), 4),
        "mae_high_flow":            round(mae_of(err_nz[high_mask]), 6),
        "mape_high_flow":           round(mape_of(err_nz[high_mask], t_nz[high_mask]), 4),
        "low_flow_zeroed_frac":     round(low_flow_zeroed_frac, 6),
        "low_flow_zeroed_max_true": round(low_flow_zeroed_max_true, 6),
    }


if __name__ == "__main__":
    args = parse_args()

    bench_csv = args.output_dir / "benchmark_results.csv"
    if not bench_csv.exists():
        raise FileNotFoundError(f"benchmark_results.csv not found: {bench_csv}")
    bench_df = pd.read_csv(bench_csv)

    print("Loading reference (no-compression) discharge array...")
    true, valid_mask = load_reference(args.output_dir)

    col_w = 22
    print(f"\n{'Codec':<{col_w}}  {'MaxErr':>10}  {'MAE(nz)':>10}  {'MAE(low)':>10}  {'MAE(high)':>10}  {'Zeroed(low)':>12}")
    print("-" * 90)

    rows = []
    for _, bench_row in bench_df.iterrows():
        label = bench_row["codec"]
        pred = load_codec_values(args.output_dir, label)
        metrics = compute_metrics(true, pred, valid_mask)

        if bench_row["family"] == "lossless":
            assert metrics["mae"] == 0.0 and metrics["max_abs_err"] == 0.0, (
                f"Lossless codec '{label}' introduced error: mae={metrics['mae']}, max_abs_err={metrics['max_abs_err']}"
            )

        row = {"codec": label, "family": bench_row["family"], "lossy": bench_row["lossy"], **metrics}
        rows.append(row)

        print(
            f"{label:<{col_w}}  {metrics['max_abs_err']:>10.2e}  {metrics['mae_nonzero']:>10.2e}"
            f"  {metrics['mae_low_flow']:>10.2e}  {metrics['mae_high_flow']:>10.2e}"
            f"  {metrics['low_flow_zeroed_frac'] * 100:>11.2f}%"
        )

    accuracy_df = pd.DataFrame(rows)
    accuracy_csv = args.output_dir / "accuracy_results.csv"
    accuracy_df.to_csv(accuracy_csv, index=False)
    print(f"\nAccuracy results saved → {accuracy_csv}")

    full_df = bench_df.merge(accuracy_df.drop(columns=["family", "lossy"]), on="codec", how="left")
    full_csv = args.output_dir / "full_results.csv"
    full_df.to_csv(full_csv, index=False)
    print(f"Combined results saved → {full_csv}")
