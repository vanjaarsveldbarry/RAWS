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
from pathlib import Path

import dask
import pandas as pd
import xarray as xr
from dask.distributed import Client
from tqdm import tqdm
from zarr.codecs import GzipCodec, PCodec, ZstdCodec
from zarr.codecs.numcodecs import BitRound

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("zarr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Zarr v3 compression codecs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root dir; input/, output/, and _temp/ subfolders are derived from this")
    parser.add_argument("--n-workers", type=int, default=24)
    parser.add_argument("--worker-memory-limit", type=str, default="7GB")
    return parser.parse_args()


def load_codec_configs(path: Path) -> list[dict]:
    # maps the "type" tag in codec_configs.json to the class that builds it
    codec_types = {
        "GzipCodec": GzipCodec,
        "ZstdCodec": ZstdCodec,
        "BitRound": BitRound,
        "PCodec": PCodec,
    }

    build = lambda spec: None if spec is None else codec_types[spec["type"]](**spec["params"])

    configs = [c for c in json.loads(path.read_text()) if c.get("enabled", True)]
    for c in configs:
        c["filters"] = [build(f) for f in c["filters"]]
        c["compressors"] = [build(cp) for cp in c["compressors"]]
        c["serializer"] = build(c.get("serializer"))
    return configs


def create_encoding(
    ds: xr.Dataset,
    chunk_dict: dict,
    shard_dict: dict,
    filters: list,
    compressors: list,
    serializer=None,
) -> dict:
    encoding = {}
    for name, var in ds.data_vars.items():
        enc = {
            "chunks": tuple(chunk_dict.get(d, 1) for d in var.dims),
            "shards": tuple(shard_dict.get(d, 1) for d in var.dims),
            "filters": filters,
            "compressors": compressors,
            "dtype": "float32",
        }
        if serializer is not None:  # zarr rejects an explicit serializer=None in the encoding dict
            enc["serializer"] = serializer
        encoding[name] = enc
    return encoding


def _drop_page_cache(path: Path) -> None:
    # per-file POSIX_FADV_DONTNEED, no root needed, unlike /proc/sys/vm/drop_caches
    for f in path.rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def benchmark_codec(
    cfg: dict,
    ds_write: xr.Dataset,
    chunk_dict: dict,
    shard_dict: dict,
    output_dir: Path,
) -> dict:
    label = cfg["label"]
    store_path = output_dir / f"bench_{label}.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    encoding = create_encoding(
        ds_write, chunk_dict, shard_dict, cfg["filters"], cfg["compressors"], cfg.get("serializer")
    )

    t0 = time.perf_counter()
    ds_write.to_zarr(
        store_path,
        mode="w",
        encoding=encoding,
        consolidated=True,
        zarr_format=3,
        write_empty_chunks=False,
    )
    write_s = time.perf_counter() - t0

    size_mb = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file()) / (1024**2)
    _drop_page_cache(store_path)

    t0 = time.perf_counter()
    ds_read = xr.open_zarr(store_path, consolidated=True, chunks=chunk_dict)
    dask.compute(*(ds_read[var].sum() for var in ds_read.data_vars))
    read_s = time.perf_counter() - t0

    return {
        "codec": label,
        "family": cfg["family"],
        "lossy": cfg["lossy"],
        "write_s": round(write_s, 2),
        "read_s": round(read_s, 2),
        "size_mb": round(size_mb, 1),
    }


REGIONS = {
    "global": {"latitude": slice(90.0, -90.0), "longitude": slice(-180.0, 180.0)},
    "south_africa": {"latitude": slice(-22.0, -35.0), "longitude": slice(16.0, 33.0)},
    "africa": {"latitude": slice(37.5, -35.0), "longitude": slice(-18.0, 52.0)},
    "australia": {"latitude": slice(-10.0, -44.0), "longitude": slice(112.0, 154.0)},
}
ACTIVE_REGION = "australia"
BENCH_YEAR = 1985


if __name__ == "__main__":
    args = parse_args()
    codec_configs = load_codec_configs(Path("codec_configs.json"))
    work_dir = args.output_dir / "output"
    (work_dir / "compression_data").mkdir(parents=True, exist_ok=True)
    (work_dir / "compression_results").mkdir(parents=True, exist_ok=True)

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

    client = Client(
        processes=True,
        n_workers=args.n_workers,
        threads_per_worker=1,
        memory_limit=args.worker_memory_limit,
        local_directory=args.output_dir / "_temp",
    )
    dask.config.set(
        {
            "array.slicing.split_large_chunks": True,
            "distributed.worker.memory.target": 0.8,
            "distributed.worker.memory.spill": 0.9,
        }
    )
    ds_ref = xr.open_zarr(
        args.output_dir / "input" / "forcing.zarr",
        consolidated=True,
        chunks=shard_chunk_dict,
    )
    bounds = REGIONS[ACTIVE_REGION]
    ds_ref = ds_ref.sel(latitude=bounds["latitude"], longitude=bounds["longitude"], time=slice(f"{BENCH_YEAR}-01-01", f"{BENCH_YEAR}-12-31"))

    ds_write = ds_ref.chunk(shard_chunk_dict)

    results = []
    baseline_mb = None
    with client:
        for cfg in tqdm(codec_configs, desc="Benchmarking codecs", unit="codec"):
            result = benchmark_codec(
                cfg, ds_write, final_dask_chunk_dict, shard_chunk_dict, work_dir / "compression_data"
            )
            if cfg["label"] == "no-compression":
                baseline_mb = result["size_mb"]
            result["ratio"] = (
                round(baseline_mb / result["size_mb"], 2)
                if baseline_mb and result["size_mb"] > 0
                else float("nan")
            )
            results.append(result)

    pd.DataFrame(results).to_csv(work_dir / "compression_results" / "benchmark_results.csv", index=False)
