"""Merge per-clone 30sec forcing zarr stores into one global forcing.zarr.

Each clone is a basin/sub-basin bounding box on the global 30 arcsec grid;
valid (non-NaN) cells of overlapping clones are disjoint, so a first-non-NaN
composite is exact. One task = one complete output shard region, so workers
write without any contention.

Usage: taskset -c 0-191 python 1_merge_forcing.py <globgm_input_dir> <output_dir>
"""

import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

RES = 1.0 / 120.0
GLOBAL_NLAT = 21600
GLOBAL_NLON = 43200
LAT_TOP = 90.0 - RES / 2  # centre of global row 0
LON_LEFT = -180.0 + RES / 2  # centre of global column 0

PERIODS = ["1979_1994", "1995_2009", "2010_2019"]
STORE_VARS = {  # source store name -> variable name (kept as-is in output)
    "totalEvaporation_monthTot_output.zarr": "total_evaporation",
    "precipitation_monthTot_output.zarr": "precipitation",
    "totalRunoff_monthTot_output.zarr": "total_runoff",

}

REF_STORE = "totalEvaporation_monthTot_output.zarr"

# Inner dask/zarr chunks nested inside larger on-disk shards.
# Both must tile the grid exactly and shards must be a multiple of chunks:
#   21600/1800=12, 43200/1800=24, 12/2=6, 21600/7200=3, 43200/7200=6, 492/12=41
final_dask_chunk_dict = {"time": 2, "latitude": 1800, "longitude": 1800}  # inner chunks
shard_chunk_dict = {"time": 12, "latitude": 7200, "longitude": 7200}  # outer shards
TIME_BLOCK = shard_chunk_dict["time"]
TILE = shard_chunk_dict["latitude"]  # spatial shard edge (lat == lon)


def create_encoding(ds):
    """Build per-variable Zarr v3 encoding with inner chunks + outer shards."""
    compressor = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
    encoding = {}
    for name, var in ds.variables.items():
        if name in ds.data_vars:
            chunks = tuple(final_dask_chunk_dict.get(d, 1) for d in var.dims)
            shards = tuple(shard_chunk_dict.get(d, 1) for d in var.dims)
            encoding[name] = {
                "chunks": chunks,  # inner chunks
                "shards": shards,  # outer shards
                "compressors": [compressor],
                "dtype": "float32",
            }
        else:
            encoding[name] = {"compressors": None}
    encoding["time"]["units"] = "days since 1901-01-01"
    encoding["time"]["dtype"] = "int64"
    return encoding


def scan_clones(clone_dir):
    """Validate grid alignment of every clone and return placement table + time axes.

    Returns (clones, period_times) where clones is a list of
    (name, row0, col0, nlat, nlon) global-grid placements and period_times maps
    period -> float array of days since 1901-01-01.
    """
    clone_names = sorted(p.name for p in clone_dir.iterdir() if p.is_dir())
    if not clone_names:
        sys.exit(f"no clones found in {clone_dir}")

    def scan(name):
        g = zarr.open_group(str(clone_dir / name / PERIODS[0] / "netcdf" / REF_STORE), mode="r")
        lat, lon = g["lat"][:], g["lon"][:]
        row0f = (LAT_TOP - float(lat[0])) / RES
        col0f = (float(lon[0]) - LON_LEFT) / RES
        row0, col0 = round(row0f), round(col0f)
        assert abs(row0f - row0) < 0.01 and abs(col0f - col0) < 0.01, f"{name}: off-grid origin"
        assert abs((float(lat[0]) - float(lat[-1])) / (len(lat) - 1) - RES) < 1e-6, f"{name}: lat step"
        assert abs((float(lon[-1]) - float(lon[0])) / (len(lon) - 1) - RES) < 1e-6, f"{name}: lon step"
        assert 0 <= row0 and row0 + len(lat) <= GLOBAL_NLAT, f"{name}: lat out of range"
        assert 0 <= col0 and col0 + len(lon) <= GLOBAL_NLON, f"{name}: lon out of range"
        times = {}
        for period in PERIODS:
            gp = zarr.open_group(str(clone_dir / name / period / "netcdf" / REF_STORE), mode="r")
            times[period] = gp["time"][:].astype(np.float64)
        return (name, row0, col0, len(lat), len(lon)), times

    clones, period_times = [], None
    with ThreadPoolExecutor(max_workers=64) as pool:
        for placement, times in tqdm(pool.map(scan, clone_names), total=len(clone_names), desc="Scanning clones"):
            clones.append(placement)
            if period_times is None:
                period_times = times
            else:
                for period in PERIODS:
                    assert np.array_equal(times[period], period_times[period]), (
                        f"{placement[0]}: time axis differs for {period}"
                    )
    return clones, period_times


def create_template(grid_file, out_store, period_times):
    """Write the empty global store (metadata + coords only) and return period offsets."""
    import dask.array as da
    import xarray as xr

    grid_ref = xr.open_dataset(grid_file, chunks={})
    grid_ref = grid_ref.drop_attrs()
    grid_ref = grid_ref.rename({"lat": "latitude", "lon": "longitude"})
    assert grid_ref.sizes == {"latitude": GLOBAL_NLAT, "longitude": GLOBAL_NLON}

    days = np.concatenate([period_times[p] for p in PERIODS])
    time = np.datetime64("1901-01-01", "ns") + days.astype(np.int64).astype("timedelta64[D]")
    ntime = len(time)
    assert ntime % TIME_BLOCK == 0, f"{ntime} months not divisible by time shard {TIME_BLOCK}"

    shape = (ntime, GLOBAL_NLAT, GLOBAL_NLON)
    chunks = tuple(shard_chunk_dict[d] for d in ("time", "latitude", "longitude"))
    ds_forcing = xr.Dataset(
        {v: (("time", "latitude", "longitude"), da.full(shape, np.nan, chunks=chunks, dtype=np.float32))
         for v in STORE_VARS.values()},
        coords={
            "time": time,
            "latitude": grid_ref["latitude"],
            "longitude": grid_ref["longitude"],
        },
    )
    encoding = create_encoding(ds_forcing)
    if out_store.exists():
        shutil.rmtree(out_store)
    ds_forcing.to_zarr(
        out_store,
        mode="w",
        encoding=encoding,
        zarr_format=3,
        write_empty_chunks=False,
        compute=False,  # metadata + coords only; workers fill the data
    )

    offsets = np.cumsum([0] + [len(period_times[p]) for p in PERIODS])
    for off in offsets:
        assert off % TIME_BLOCK == 0, "period boundary not aligned to time shard"
    return {p: (int(offsets[i]), int(offsets[i + 1])) for i, p in enumerate(PERIODS)}


# --- worker side ---------------------------------------------------------

_W = {}


def _init_worker(out_store, clone_dir, clones, period_offsets):
    zarr.config.set({"array.write_empty_chunks": False, "threading.max_workers": 2})
    root = zarr.open_group(str(out_store), mode="r+")
    _W.update(
        out={v: root[v] for v in STORE_VARS.values()},
        clone_dir=clone_dir,
        clones=clones,
        period_offsets=period_offsets,
        src_cache={},
    )


def _open_src(clone, period, store_name):
    key = (clone, period, store_name)
    arr = _W["src_cache"].get(key)
    if arr is None:
        path = _W["clone_dir"] / clone / period / "netcdf" / store_name
        arr = zarr.open_group(str(path), mode="r")[STORE_VARS[store_name]]
        _W["src_cache"][key] = arr
    return arr


def merge_task(task):
    """Composite one (variable, 12-month block, 7200x7200 tile) output shard."""
    store_name, block, tile_r, tile_c = task
    gt0 = block * TIME_BLOCK
    period = next(p for p, (t0, t1) in _W["period_offsets"].items() if t0 <= gt0 < t1)
    lt0 = gt0 - _W["period_offsets"][period][0]
    R0, C0 = tile_r * TILE, tile_c * TILE

    buf = np.full((TIME_BLOCK, TILE, TILE), np.nan, dtype=np.float32)
    any_valid = False
    for name, row0, col0, nlat, nlon in _W["clones"]:
        r_lo, r_hi = max(row0, R0), min(row0 + nlat, R0 + TILE)
        c_lo, c_hi = max(col0, C0), min(col0 + nlon, C0 + TILE)
        if r_lo >= r_hi or c_lo >= c_hi:
            continue
        src = _open_src(name, period, store_name)
        nt = _W["period_offsets"][period][1] - _W["period_offsets"][period][0]
        assert src.shape == (nt, nlat, nlon), f"{name}/{period}/{store_name}: unexpected shape"
        data = src[lt0:lt0 + TIME_BLOCK, r_lo - row0:r_hi - row0, c_lo - col0:c_hi - col0]
        valid = ~np.isnan(data)
        if not valid.any():
            continue
        view = buf[:, r_lo - R0:r_hi - R0, c_lo - C0:c_hi - C0]
        # a few basin-boundary cells are valid in more than one clone with
        # slightly different values; first clone (sorted order) wins, matching
        # the combine_first semantics of the original merge
        np.copyto(view, data, where=valid & np.isnan(view))
        any_valid = True

    if any_valid:
        _W["out"][STORE_VARS[store_name]][gt0:gt0 + TIME_BLOCK, R0:R0 + TILE, C0:C0 + TILE] = buf
    return any_valid


# --- driver --------------------------------------------------------------

def main():
    data_dir = Path(sys.argv[1])

    clone_dir = data_dir / "_temp/perClone/30sec/newDownscale"
    out_store = data_dir / "input/forcing.zarr"
    grid_file = data_dir / "input/cdo_grid_area_30sec_map_correct_lat.nc"

    clones, period_times = scan_clones(clone_dir)
    print(f"{len(clones)} clones aligned to global grid")
    period_offsets = create_template(grid_file, out_store, period_times)
    ntime = period_offsets[PERIODS[-1]][1]

    tasks = []
    for store_name in STORE_VARS:
        for block in range(ntime // TIME_BLOCK):
            for tile_r in range(GLOBAL_NLAT // TILE):
                for tile_c in range(GLOBAL_NLON // TILE):
                    R0, C0 = tile_r * TILE, tile_c * TILE
                    if any(row0 < R0 + TILE and row0 + nlat > R0 and col0 < C0 + TILE and col0 + nlon > C0
                           for _, row0, col0, nlat, nlon in clones):
                        tasks.append((store_name, block, tile_r, tile_c))
    random.Random(0).shuffle(tasks)  # spread I/O across clones and tiles
    print(f"{len(tasks)} shard tasks on 64 workers")

    written = 0
    with ProcessPoolExecutor(
        max_workers=64,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(out_store, clone_dir, clones, period_offsets),
    ) as pool:
        futures = [pool.submit(merge_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Merging", smoothing=0.05):
            written += fut.result()  # re-raises worker exceptions
    print(f"done: {written}/{len(tasks)} shards contained data -> {out_store}")


if __name__ == "__main__":
    main()
