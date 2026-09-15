import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import logging
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import xbitinfo as xb
from dask.distributed import Client
from zarr.codecs import BloscCodec, PCodec, ZstdCodec
from zarr.codecs.numcodecs import BitRound

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("zarr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy informed-bitround (xbitinfo) compressors")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root dir; input/, output/, and _temp/ subfolders are derived from this")
    parser.add_argument("--n-workers", type=int, default=24)
    parser.add_argument("--worker-memory-limit", type=str, default="7GB")
    parser.add_argument("--inflevel", type=float, default=0.99, help="fraction of bitwise information to retain")
    parser.add_argument("--dask-local-directory", type=Path, default=None, help="dask worker spill/shuffle dir; defaults to output-dir/_temp (GPFS). Point this at node-local scratch if the cluster/partition provisions one per job")
    return parser.parse_args()


def _bitinfo_piece(piece: xr.Dataset, dim: str) -> xr.Dataset:
    dask.config.set(num_workers=4)  # cap the local scheduler's threads -- many of these run concurrently
    return xb.get_bitinformation(piece.load(), dim=dim, implementation="python")


def get_bitinformation(ds: xr.Dataset, dims: list[str], n_splits: int = 4, max_workers: int = 12) -> xr.Dataset:
    # dims are the spatial axes providing independent samples for the entropy estimate.
    # xbitinfo's python kernel unpacks every value to individual bits and builds float64
    # joint-probability arrays -- ~70x the raw array size observed at full global res (one
    # var/month/dim peaked ~255GB; OOM-killed a 1440GB-cgroup job when done for all 6 months
    # at once). Instead of subsampling pixels, partition every (var, dim, month) into
    # `n_splits` pieces along the OTHER spatial axis and run them concurrently across
    # processes: every pixel is still used, just in smaller/more-parallel batches, which also
    # turns this from a serial loop into something that actually uses the node's 192 cores.
    # Combine with max (not mean): a signal visible in any piece must not be truncated,
    # matching the max-across-dims policy below.
    n_time = ds.sizes["time"]
    tasks = []
    for dim in dims:
        split_dim = "longitude" if dim == "latitude" else "latitude"
        edges = np.linspace(0, ds.sizes[split_dim], n_splits + 1, dtype=int)
        for var in ds.data_vars:
            for t in range(n_time):
                for s in range(n_splits):
                    piece = ds[[var]].isel(
                        time=slice(t, t + 1),
                        **{split_dim: slice(int(edges[s]), int(edges[s + 1]))},
                    )
                    tasks.append(((var, dim), piece, dim))

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_bitinfo_piece, piece, dim): key for key, piece, dim in tasks}
        for n_done, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            results.setdefault(key, []).append(fut.result())
            print(f"bitinformation piece {n_done}/{len(tasks)} done", flush=True)

    per_dim = {}
    for dim in dims:
        per_var = [xr.concat(results[(var, dim)], dim="piece").max("piece") for var in ds.data_vars]
        per_dim[dim] = xr.merge(per_var).expand_dims("dim", axis=0).assign_coords(dim=[dim])
    return xr.merge(per_dim.values(), join="outer", compat="no_conflicts").squeeze()


def keepbits_from_info(info_per_bit: xr.Dataset, inflevel: float | list[float]) -> xr.Dataset:
    # when dim was a list, take the max keepbits across dims: info visible on one spatial
    # axis but not the other must not be truncated.
    keepbits = xb.get_keepbits(info_per_bit, inflevel=inflevel)
    return keepbits.max("dim") if "dim" in keepbits.dims else keepbits


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def compressed_size(field_da: xr.DataArray, keepbits: int, tmp_dir: Path) -> int:
    tmp_path = tmp_dir / f"_sweep_{field_da.name}_{keepbits}.zarr"
    shape = field_da.shape
    encoding = {field_da.name: {
        "chunks": shape, "shards": shape,
        "filters": [BitRound(keepbits=keepbits)],
        "compressors": [ZstdCodec(level=4)],
        "dtype": "float32",
    }}
    field_da.to_dataset().to_zarr(tmp_path, mode="w", encoding=encoding, consolidated=True, zarr_format=3, write_empty_chunks=False)
    size = dir_size_bytes(tmp_path)
    shutil.rmtree(tmp_path)
    return size


def sweep_snapshot(ds: xr.Dataset, info_per_bit: xr.Dataset, inflevels: list[float], time_index: int) -> tuple[xr.Dataset, xr.Dataset]:
    # only step in the sweep that touches the dask/distributed graph -- must run inside `with client:`
    keepbits_ds = keepbits_from_info(info_per_bit, inflevels)
    return ds.isel(time=time_index).compute(), keepbits_ds


def compression_sweep(snap: xr.Dataset, keepbits_ds: xr.Dataset, inflevels: list[float], tmp_dir: Path) -> tuple[dict, list[dict]]:
    """Bitround+zstd a single time-slice snapshot at several information levels -- one map per level, per variable.
    snap must already be resolved to plain numpy (not dask) -- this does ~2*len(inflevels)*len(data_vars) synchronous
    zarr writes, which will starve a live distributed scheduler's heartbeat if run inside its `with client:` block."""
    fields = {"latitude": snap["latitude"].values, "longitude": snap["longitude"].values}
    meta = []
    for var in snap.data_vars:
        raw = snap[var].astype("float32").values
        fields[f"{var}__raw"] = raw
        raw_bytes = raw.nbytes
        for i, inflevel in enumerate(inflevels):
            keepbits = max(0, int(keepbits_ds[var].isel(inflevel=i).item()))
            fields[f"{var}__{i}"] = xb.xr_bitround(snap[var], keepbits).values.astype("float32")
            size = compressed_size(snap[var], keepbits, tmp_dir)
            meta.append({"variable": var, "level": i, "inflevel": inflevel, "keepbits": keepbits,
                         "compression_factor": raw_bytes / size, "size_bytes": size, "raw_bytes": raw_bytes})
    return fields, meta


def build_encoding(
    ds: xr.Dataset,
    filters_by_var: dict,
    chunk_dict: dict,
    shard_dict: dict,
    compressors: list,
    serializer=None,
) -> dict:
    encoding = {}
    for name, var in ds.data_vars.items():
        enc = {
            "chunks": tuple(chunk_dict.get(d, 1) for d in var.dims),
            "shards": tuple(shard_dict.get(d, 1) for d in var.dims),
            "filters": filters_by_var.get(name, []),
            "compressors": compressors,
            "dtype": "float32",
        }
        if serializer is not None:  # zarr rejects an explicit serializer=None in the encoding dict
            enc["serializer"] = serializer
        encoding[name] = enc
    return encoding


# baselines to compare the informed-bitround stores against: raw size, and the
# best a lossless-only codec can do on the same (still full-precision) data
LOSSLESS_FAMILIES = {
    "no-compression": dict(compressors=[], serializer=None),
    "lossless+zstd": dict(compressors=[ZstdCodec(level=4)], serializer=None),
}

INFORMED_FAMILIES = {
    "bitround-informed+zstd": dict(compressors=[ZstdCodec(level=4)], serializer=None),
    "bitround-informed+pco": dict(compressors=[], serializer=PCodec(level=4)),
    "bitround-informed+blosc": dict(compressors=[BloscCodec(cname="zstd", clevel=4, shuffle="shuffle")], serializer=None),
}

REGIONS = {
    "global": {"latitude": slice(90.0, -90.0), "longitude": slice(-180.0, 180.0)},
    "south_africa": {"latitude": slice(-22.0, -35.0), "longitude": slice(16.0, 33.0)},
    "africa": {"latitude": slice(37.5, -35.0), "longitude": slice(-18.0, 52.0)},
    "australia": {"latitude": slice(-10.0, -44.0), "longitude": slice(112.0, 154.0)},
}
ACTIVE_REGION = "global"
BENCH_YEAR = 1985
SWEEP_INFLEVELS = [0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0]
SWEEP_TIME_INDEX = 0  # first month of BENCH_YEAR; same snapshot reused across all variables


if __name__ == "__main__":
    args = parse_args()
    work_dir = args.output_dir / "output"
    (work_dir / "compression_data").mkdir(parents=True, exist_ok=True)

    final_dask_chunk_dict = {
        "time": 2,
        "latitude": 1280,
        "longitude": 1200,
    }
    shard_chunk_dict = {
        "time": 6,
        "latitude": 2560,
        "longitude": 3600,
    }

    ds_ref = xr.open_zarr(
        args.output_dir / "input" / "forcing.zarr",
        consolidated=True,
        chunks=shard_chunk_dict,
    )
    bounds = REGIONS[ACTIVE_REGION]
    # forcing.zarr is monthly (end-of-month timestamps); one shard-worth (6 months) along time.
    ds_ref = ds_ref.sel(latitude=bounds["latitude"], longitude=bounds["longitude"], time=slice(f"{BENCH_YEAR}-07-01", f"{BENCH_YEAR}-12-31"))

    # run before the distributed Client exists: get_bitinformation loads one month at a time
    # itself (see there), so ds_ref stays lazy here -- loading it whole would defeat that.
    info_per_bit = get_bitinformation(ds_ref, ["latitude", "longitude"])
    keepbits_ds = keepbits_from_info(info_per_bit, args.inflevel)
    keepbits = {var: max(0, int(keepbits_ds[var].item())) for var in ds_ref.data_vars}
    (work_dir / "compression_data" / "keepbits.json").write_text(json.dumps(keepbits, indent=2))
    bitround_filters = {name: [BitRound(keepbits=keepbits[name])] for name in ds_ref.data_vars}

    local_directory = args.dask_local_directory or args.output_dir / "_temp"
    local_directory.mkdir(parents=True, exist_ok=True)
    client = Client(
        processes=True,
        n_workers=args.n_workers,
        threads_per_worker=1,
        memory_limit=args.worker_memory_limit,
        local_directory=local_directory,
    )
    dask.config.set(
        {
            "array.slicing.split_large_chunks": True,
            "distributed.worker.memory.target": 0.8,
            "distributed.worker.memory.spill": 0.9,
        }
    )
    ds_write = ds_ref.chunk(shard_chunk_dict)

    timings = {}
    with client:
        sweep_snap, sweep_keepbits_ds = sweep_snapshot(ds_write, info_per_bit, SWEEP_INFLEVELS, SWEEP_TIME_INDEX)

        for label, family in {**LOSSLESS_FAMILIES, **INFORMED_FAMILIES}.items():
            filters_by_var = bitround_filters if label in INFORMED_FAMILIES else {}
            store_path = work_dir / "compression_data" / f"deploy_{label}.zarr"
            encoding = build_encoding(ds_write, filters_by_var, final_dask_chunk_dict, shard_chunk_dict, **family)

            t0 = time.perf_counter()
            ds_write.to_zarr(
                store_path,
                mode="w",
                encoding=encoding,
                consolidated=True,
                zarr_format=3,
                write_empty_chunks=False,
            )
            write_seconds = time.perf_counter() - t0

            # page cache is warm from the write just above, so this is a best-case (not cold)
            # read; the bias is the same for all 5 stores so relative comparisons still hold.
            t0 = time.perf_counter()
            xr.open_zarr(store_path, consolidated=True, chunks=final_dask_chunk_dict).load()
            read_seconds = time.perf_counter() - t0

            size_bytes = dir_size_bytes(store_path)
            timings[label] = {
                "write_seconds": write_seconds,
                "read_seconds": read_seconds,
                "size_bytes": size_bytes,
                "write_mbps": size_bytes / 1e6 / write_seconds,
                "read_mbps": size_bytes / 1e6 / read_seconds,
            }

    (work_dir / "compression_data" / "benchmark_timings.json").write_text(json.dumps(timings, indent=2))

    # runs after `with client:` closes: the writes below are synchronous local zarr I/O in the
    # main thread, which would otherwise starve the LocalCluster scheduler thread's heartbeat
    sweep_fields, sweep_meta = compression_sweep(sweep_snap, sweep_keepbits_ds, SWEEP_INFLEVELS, args.output_dir / "_temp")
    np.savez(work_dir / "compression_data" / "compression_sweep_fields.npz", **sweep_fields)
    (work_dir / "compression_data" / "compression_sweep_meta.json").write_text(json.dumps(sweep_meta, indent=2))

    print(f"informed_bitround.py finished: {work_dir / 'compression_data'}")
