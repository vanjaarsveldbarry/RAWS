import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"

from pathlib import Path
import argparse
import time
import shutil
import logging
import warnings

import pandas as pd
import xarray as xr
import dask
from dask.distributed import Client
from tqdm import tqdm
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
import zarr.codecs.numcodecs as nc_codecs

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
    parser = argparse.ArgumentParser(description="Benchmark Zarr v3 compression codecs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, required=True)
    parser.add_argument("--n-workers", type=int, default=24)
    parser.add_argument("--worker-memory-limit", type=str, default="7GB")
    return parser.parse_args()


def raw_size_mb(ds: xr.Dataset) -> float:
    total = 0
    for var in ds.data_vars:
        elements = 1
        for d in ds[var].dims:
            elements *= ds.sizes[d]
        total += elements * 4  # float32 = 4 bytes
    return total / (1024**2)


def create_encoding(
    ds: xr.Dataset,
    chunk_dict: dict,
    shard_dict: dict,
    filters: list,
    compressors: list,
    serializer=None,
    sz3_params: dict | None = None,   # NEW — optional, defaults to None
) -> dict:
    encoding = {}
    for name, var in ds.data_vars.items():
        chunks = tuple(chunk_dict.get(d, 1) for d in var.dims)
        shards = tuple(shard_dict.get(d, 1) for d in var.dims)
        enc = {
            "chunks": chunks,
            "shards": shards,
            "filters": filters,
            "compressors": compressors,
            "dtype": "float32",
        }
        if sz3_params is not None:                                    # NEW branch
            enc["serializer"] = _SZ3Codec(shape=chunks, dtype="float32", **sz3_params)
        elif serializer is not None:                                  # unchanged
            enc["serializer"] = serializer
        encoding[name] = enc
    return encoding


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
            ds_write,
            chunk_dict,
            shard_dict,
            cfg["filters"],
            cfg["compressors"],
            cfg.get("serializer"),
            cfg.get("sz3_params"),   # NEW — .get() is safe, returns None for every non-SZ3 cfg
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

    t0 = time.perf_counter()
    ds_read = xr.open_zarr(store_path, consolidated=True, chunks=chunk_dict)
    for var in ds_read.data_vars:
        ds_read[var].sum().compute()
    read_s = time.perf_counter() - t0

    return {
        "codec": label,
        "family": cfg["family"],
        "lossy": cfg["lossy"],
        "write_s": round(write_s, 2),
        "read_s": round(read_s, 2),
        "size_mb": round(size_mb, 1),
    }


CODEC_CONFIGS = [
    # lossless
    dict(
        label="no-compression",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[],
        serializer=None,
    ),
    dict(
        label="zlib-4",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[GzipCodec(level=4)],
        serializer=None,
    ),
    dict(
        label="zstd-3",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),
    dict(
        label="lz4-blosc",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")],
        serializer=None,
    ),
    dict(
        label="blosclz-blosc",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[BloscCodec(cname="blosclz", clevel=5, shuffle="shuffle")],
        serializer=None,
    ),
    dict(
        label="lz4-bitshuffle",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")],
        serializer=None,
    ),
    dict(
        label="blosclz-bitshuffle",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[BloscCodec(cname="blosclz", clevel=5, shuffle="bitshuffle")],
        serializer=None,
    ),
    # Zstd tends to beat lz4/blosclz on scientific float arrays; cheap
    # to add given the bitshuffle infra is already here for the others.
    dict(
        label="zstd-bitshuffle",
        family="lossless",
        lossy=False,
        filters=[],
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")],
        serializer=None,
    ),

    # BitRound + zstd
    dict(
        label="bitround-14+zstd",
        family="BitRound",
        lossy=True,
        filters=[nc_codecs.BitRound(keepbits=14)],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),
    dict(
        label="bitround-10+zstd",
        family="BitRound",
        lossy=True,
        filters=[nc_codecs.BitRound(keepbits=10)],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),
    dict(
        label="bitround-6+zstd",
        family="BitRound",
        lossy=True,
        filters=[nc_codecs.BitRound(keepbits=6)],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),

    # BitRound + PCO
    dict(
        label="bitround-14+pco",
        family="BitRound",
        lossy=True,
        filters=[nc_codecs.BitRound(keepbits=14)],
        compressors=[],
        serializer=nc_codecs.PCodec(level=8),
    ),
    dict(
        label="bitround-10+pco",
        family="BitRound",
        lossy=True,
        filters=[nc_codecs.BitRound(keepbits=10)],
        compressors=[],
        serializer=nc_codecs.PCodec(level=8),
    ),
    dict(
        label="bitround-6+pco",
        family="BitRound",
        lossy=True,
        filters=[nc_codecs.BitRound(keepbits=6)],
        compressors=[],
        serializer=nc_codecs.PCodec(level=8),
    ),
    # Quantize + zstd
    dict(
        label="quantize-4+zstd",
        family="Quantize",
        lossy=True,
        filters=[nc_codecs.Quantize(digits=4, dtype="float32")],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),
    dict(
        label="quantize-3+zstd",
        family="Quantize",
        lossy=True,
        filters=[nc_codecs.Quantize(digits=3, dtype="float32")],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),
    dict(
        label="quantize-2+zstd",
        family="Quantize",
        lossy=True,
        filters=[nc_codecs.Quantize(digits=2, dtype="float32")],
        compressors=[ZstdCodec(level=3)],
        serializer=None,
    ),

    # ZFP (absolute error bound in m³/s)
    dict(
        label="zfpy-1e-4",
        family="ZFP",
        lossy=True,
        filters=[],
        compressors=[],
        serializer=nc_codecs.ZFPY(tolerance=1e-4),
    ),
    dict(
        label="zfpy-1e-3",
        family="ZFP",
        lossy=True,
        filters=[],
        compressors=[],
        serializer=nc_codecs.ZFPY(tolerance=1e-3),
    ),
    dict(
        label="zfpy-1e-2",
        family="ZFP",
        lossy=True,
        filters=[],
        compressors=[],
        serializer=nc_codecs.ZFPY(tolerance=1e-2),
    ),

    # SZ3: abs bounds mirror ZFP range for direct comparison; pw_rel is the key
    dict(label="sz3-abs-1e-4", family="SZ3", lossy=True, filters=[], compressors=[],
         sz3_params=dict(mode="abs", abs=1e-4)),
    dict(label="sz3-abs-1e-3", family="SZ3", lossy=True, filters=[], compressors=[],
         sz3_params=dict(mode="abs", abs=1e-3)),
    dict(label="sz3-abs-1e-2", family="SZ3", lossy=True, filters=[], compressors=[],
         sz3_params=dict(mode="abs", abs=1e-2)),
    dict(label="sz3-rel-1e-3", family="SZ3", lossy=True, filters=[], compressors=[],
         sz3_params=dict(mode="rel", rel=1e-3)),
    dict(label="sz3-rel-1e-2", family="SZ3", lossy=True, filters=[], compressors=[],
         sz3_params=dict(mode="rel", rel=1e-2)),
]


REGIONS = {
    "global": dict(lat=None, lon=None),
    "south_africa": dict(lat=slice(-22.0, -35.0), lon=slice(16.0, 33.0)),
    "africa": dict(lat=slice(37.5, -35.0), lon=slice(-18.0, 52.0)),
    "australia": dict(lat=slice(-10.0, -44.0), lon=slice(112.0, 154.0)),
}
ACTIVE_REGION = "global"


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_dask_chunk_dict = {
        "time": 2,
        "lat": 1280,
        "lon": 1200,
    }
    shard_chunk_dict = {
        "time": 6,
        "lat": 2560,
        "lon": 3600,
    }

    client = Client(
        processes=True,
        n_workers=args.n_workers,
        threads_per_worker=1,
        memory_limit=args.worker_memory_limit,
        local_directory=args.temp_dir,
    )
    dask.config.set(
        {
            "array.slicing.split_large_chunks": True,
            "distributed.worker.memory.target": 0.8,
            "distributed.worker.memory.spill": 0.9,
        }
    )
    print(
        f"Dask dashboard : {client.dashboard_link} ---- "
        f"Workers : {args.n_workers} @ {args.worker_memory_limit} each"
    )

    ds_ref = xr.open_dataset(
        args.input_dir / "discharge_1985_uncompressed.nc",
        engine="h5netcdf",
        chunks=shard_chunk_dict,
    )
    bounds = REGIONS[ACTIVE_REGION]
    if bounds["lat"] is not None:
        ds_ref = ds_ref.sel(lat=bounds["lat"], lon=bounds["lon"])
    region_label = ACTIVE_REGION.replace("_", " ").title()
    print(f"\nRegion     : {region_label}")

    ds_write = ds_ref.chunk(shard_chunk_dict)

    uncompressed_mb = raw_size_mb(ds_ref)
    print(f"Raw size (float32) : {uncompressed_mb:.1f} MB")
    print(f"Inner chunks : {final_dask_chunk_dict}")
    print(f"Outer shards : {shard_chunk_dict}\n")

    col_w = 22
    print(
        f"{'Codec':<{col_w}}  {'Write(s)':>9}  {'Read(s)':>8}  {'MB':>8}  {'Ratio':>7}"
    )
    print("-" * 60)

    results = []
    baseline_mb = None
    with client:
        for cfg in tqdm(CODEC_CONFIGS, desc="Benchmarking codecs", unit="codec"):
            result = benchmark_codec(
                cfg, ds_write, final_dask_chunk_dict, shard_chunk_dict, args.output_dir
            )
            if cfg["label"] == "no-compression":
                baseline_mb = result["size_mb"]
            result["ratio"] = (
                round(baseline_mb / result["size_mb"], 2)
                if baseline_mb and result["size_mb"] > 0
                else float("nan")
            )
            results.append(result)

            print(
                f"{result['codec']:<{col_w}}  {result['write_s']:>9.2f}  {result['read_s']:>8.2f}"
                f"  {result['size_mb']:>8.1f}  {result['ratio']:>7.2f}×"
            )

    df = pd.DataFrame(results)
    csv_path = args.output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
