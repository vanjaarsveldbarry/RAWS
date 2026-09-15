import concurrent.futures
import math
import pathlib
import shutil
import subprocess
from typing import NamedTuple

import numpy as np
import pcraster as pcr
import xarray as xr
import yaml
import zarr
from netCDF4 import Dataset
from osgeo import gdal

gdal.UseExceptions()

CATALOGUE = pathlib.Path("CATALOGUE")
GLOBAL_INPUTS = CATALOGUE / "GLOBAL_INPUTS"
TEST_CASES = CATALOGUE / "TEST_CASES"
NETCDF_EXTS = (".nc", ".nc4")
PCRMAP_EXTS = (".map", ".ldd")
X_NAMES = ("lon", "longitude")
Y_NAMES = ("lat", "latitude")
EARTH_RADIUS = 6371007.181
WORKERS = 8
READ_BYTES = 256 << 20
REL_TOL = 1e-3
TOL = 1e-9
MARGIN = 1


class Basin(NamedTuple):
    mask: np.ndarray
    cell: float
    west: float
    east: float
    south: float
    north: float


def delineate(src, outlet, snap, search):
    west, east, south, north = search
    ds = gdal.Open(str(src))
    x0, cs, _, y0, _, _ = ds.GetGeoTransform()
    per = round(1 / cs)
    cell = 1 / per
    c0, c1 = max(round((west - x0) * per), 0), min(round((east - x0) * per), ds.RasterXSize)
    r0, r1 = max(round((y0 - north) * per), 0), min(round((y0 - south) * per), ds.RasterYSize)
    codes = ds.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0, r1 - r0)
    codes = np.where((codes >= 1) & (codes <= 9), codes, 0).astype(np.int32)
    rows, cols = codes.shape
    xul, yul = round((x0 + c0 * cs) * per) / per, round((y0 - r0 * cs) * per) / per

    pcr.setclone(rows, cols, cell, xul, yul)
    ldd = pcr.lddrepair(pcr.ldd(pcr.numpy2pcr(pcr.Ldd, codes, 0)))
    acc = pcr.pcr2numpy(pcr.accuflux(ldd, 1.0), -1.0)

    lon = xul + (np.arange(cols) + 0.5) * cell
    lat = yul - (np.arange(rows) + 0.5) * cell
    near = (np.abs(lat - outlet[1]) <= snap)[:, None] & (np.abs(lon - outlet[0]) <= snap)[None, :]
    if not near.any() or acc[near].max() <= 0:
        raise SystemExit(f"{src.name}: no LDD cell within {snap} deg of {outlet}")
    orow, ocol = divmod(int(np.where(near, acc, -1.0).argmax()), cols)

    seed = np.zeros((rows, cols), np.int32)
    seed[orow, ocol] = 1
    mask = pcr.pcr2numpy(pcr.catchment(ldd, pcr.numpy2pcr(pcr.Nominal, seed, 0)), 0) == 1
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise SystemExit(f"{src.name}: catchment reaches the edge of search window {search}; "
                         f"widen `search` in testcases.yml")

    r, c = np.nonzero(mask)
    top, bot = np.radians(lat[r] + cell / 2), np.radians(lat[r] - cell / 2)
    area = (EARTH_RADIUS ** 2 * math.radians(cell) * (np.sin(top) - np.sin(bot))).sum() / 1e6
    print(f"  {src.name}: outlet {lon[ocol]:.5f}, {lat[orow]:.5f}; {len(r):,} cells, {area:,.0f} km2")
    return Basin(mask[r.min():r.max() + 1, c.min():c.max() + 1], cell,
                 xul + c.min() * cell, xul + (c.max() + 1) * cell,
                 yul - (r.max() + 1) * cell, yul - r.min() * cell)


def write_clone(name, res, basin, box):
    west, east, south, north = box
    cell = basin.cell
    rows, cols = round((north - south) / cell), round((east - west) / cell)
    r, c = np.nonzero(basin.mask)
    mask = np.zeros((rows, cols), np.int32)
    mask[r + round((north - basin.north) / cell), c + round((basin.west - west) / cell)] = 1

    out = TEST_CASES / name / "clone_maps"
    out.mkdir(parents=True, exist_ok=True)
    pcr.setclone(rows, cols, cell, west, north)
    pcr.report(pcr.numpy2pcr(pcr.Boolean, np.ones_like(mask), 0), str(out / f"{name}_{res}.clone.map"))
    pcr.report(pcr.numpy2pcr(pcr.Boolean, mask, 0), str(out / f"{name}_{res}.mask.map"))
    (out / f"{name}_{res}.grid.txt").write_text(
        f"gridtype = lonlat\nxsize = {cols}\nysize = {rows}\n"
        f"xfirst = {west + cell / 2!r}\nxinc = {cell!r}\n"
        f"yfirst = {north - cell / 2!r}\nyinc = {-cell!r}\n")


def is_zarr(path):
    return (path / ".zgroup").is_file()


def files_in(src):
    if src.is_dir() and not is_zarr(src):
        return sorted(p for p in src.rglob("*") if p.is_file())
    return [src]


def native_cell(src):
    for path in files_in(src):
        if path.suffix in PCRMAP_EXTS:
            return gdal.Open(str(path)).GetGeoTransform()[1]
        if path.suffix in NETCDF_EXTS or is_zarr(path):
            with xr.open_dataset(path, engine="zarr" if is_zarr(path) else None, decode_times=False) as ds:
                xn = next(n for n in ds.variables if n.lower() in X_NAMES)
                return float(np.median(np.diff(ds[xn].values)))


def clone_axes(box, cell):
    west, east, south, north = box
    tx = west + cell / 2 + np.arange(round((east - west) / cell)) * cell
    ty = north - cell / 2 - np.arange(round((north - south) / cell)) * cell
    return tx, ty


def axis_window(coord, targets, inc):
    coord = np.asarray(coord, dtype="f8")
    step = float(np.median(np.abs(np.diff(coord))))
    edge = targets[0] - inc / 2
    start = int(np.argmin(np.abs(coord - edge - math.copysign(step / 2, inc))))
    n = math.ceil(len(targets) / round(step / abs(inc)))
    if (coord[1] > coord[0]) != (inc > 0) or abs(coord[start] - edge) > step + TOL or start + n > len(coord):
        raise ValueError(f"axis {coord[0]}..{coord[-1]} step {step} does not cover "
                         f"clone {targets[0]}..{targets[-1]} step {inc}")
    vals = coord[start:start + n]
    if n == len(targets) and np.abs(vals - targets).max() <= abs(inc) / 2:
        return slice(start, start + n), targets
    lead, trail = min(MARGIN, start), min(MARGIN, len(coord) - start - n)
    start, n = start - lead, n + lead + trail
    return slice(start, start + n), coord[start:start + n]


def crop_netcdf(src_path, dst_path, tx, ty, cell):
    with Dataset(src_path) as src:
        names = {n.lower(): n for n in src.variables}
        xn = next(names[k] for k in X_NAMES if k in names)
        yn = next(names[k] for k in Y_NAMES if k in names)
        xs, xvals = axis_window(src.variables[xn][:], tx, cell)
        ys, yvals = axis_window(src.variables[yn][:], ty, -cell)
        nx, ny = len(xvals), len(yvals)
        fields = [n for n, v in src.variables.items() if v.dimensions[-2:] == (yn, xn)]
        tn = "time" if "time" in src.dimensions else None
        fmt = "NETCDF4_CLASSIC" if src.data_model.startswith("NETCDF3") else src.data_model

        with Dataset(dst_path, "w", format=fmt) as dst:
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
            axes = {yn: yvals, xn: xvals}
            if tn:
                axes = {tn: src.variables[tn][:]} | axes
                dst.createDimension(tn, None)
            dst.createDimension(yn, ny)
            dst.createDimension(xn, nx)
            for n, vals in axes.items():
                sv = src.variables[n]
                dv = dst.createVariable(n, sv.dtype, (n,))
                dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs() if k != "_FillValue"})
                dv[:] = vals

            for n in fields:
                sv = src.variables[n]
                fill = sv.getncattr("_FillValue") if "_FillValue" in sv.ncattrs() else None
                dv = dst.createVariable(n, sv.dtype, sv.dimensions, fill_value=fill, zlib=True,
                                        complevel=1, shuffle=False,
                                        chunksizes=[1] * (sv.ndim - 2) + [ny, nx])
                dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs() if k != "_FillValue"})
                if tn in sv.dimensions:
                    per = max(1, READ_BYTES // (nx * ny * sv.dtype.itemsize))
                    for t0 in range(0, sv.shape[0], per):
                        t = slice(t0, min(t0 + per, sv.shape[0]))
                        dv[t] = sv[t, ys, xs]
                else:
                    dv[:] = sv[ys, xs]


def crop_zarr(src_path, dst_path, tx, ty, cell):
    src = zarr.open(str(src_path), mode="r")
    name = next(k for k in src.array_keys() if k not in ("lon", "lat", "time"))
    var = src[name]
    dims = list(var.attrs["_ARRAY_DIMENSIONS"])
    xs, xvals = axis_window(src["lon"][:], tx, cell)
    ys, yvals = axis_window(src["lat"][:], ty, -cell)
    if len(xvals) != len(tx) or len(yvals) != len(ty):
        raise ValueError(f"{src_path} is not on the clone grid")
    nx, ny = len(xvals), len(yvals)

    def read_step(t):
        return var[tuple({"time": t, "lon": xs, "lat": ys}[d] for d in dims)]

    attrs = {k: v for k, v in var.attrs.items() if k != "_ARRAY_DIMENSIONS"}
    tattrs = {k: v for k, v in src["time"].attrs.items() if k != "_ARRAY_DIMENSIONS"}
    tvals = src["time"][:]
    chunks = tuple({"time": 1, "lon": nx, "lat": ny}[d] for d in dims)
    encoding = {name: {"chunks": chunks, "_FillValue": var.fill_value}}
    per = max(1, READ_BYTES // (nx * ny * var.dtype.itemsize))
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        for t0 in range(0, len(tvals), per):
            t1 = min(t0 + per, len(tvals))
            block = np.stack(list(ex.map(read_step, range(t0, t1))), axis=dims.index("time"))
            ds = xr.Dataset({name: (dims, block, attrs)},
                            coords={"time": ("time", tvals[t0:t1], tattrs), "lon": xvals, "lat": yvals})
            if t0 == 0:
                ds.to_zarr(str(dst_path), mode="w", zarr_format=2, encoding=encoding)
            else:
                ds.to_zarr(str(dst_path), mode="a", append_dim="time")


def crop_one(src, dst, box, cell):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in NETCDF_EXTS:
        crop_netcdf(src, dst, *clone_axes(box, cell), cell)
    elif src.suffix in PCRMAP_EXTS:
        dst.unlink(missing_ok=True)
        west, east, south, north = box
        subprocess.run(["gdal_translate", "-q", "-of", "PCRaster", "-projwin", str(west), str(north),
                        str(east), str(south), str(src), str(dst)], check=True)
    elif is_zarr(src):
        crop_zarr(src, dst, *clone_axes(box, cell), cell)
    else:
        shutil.copy2(src, dst)


def crop_case(name, manifest, box, cells):
    finest = min(cells.values())
    work = []
    for entry in manifest.values():
        rel = pathlib.Path(entry["dst"])
        src = GLOBAL_INPUTS / rel
        native = native_cell(src)
        if native < finest * (1 - REL_TOL):
            continue
        cell = finest
        if is_zarr(src):
            cell = next((c for c in cells.values() if abs(c - native) <= native * REL_TOL), None)
            if cell is None:
                continue
        work += [(path, TEST_CASES / name / rel / path.relative_to(src), box, cell) for path in files_in(src)]
    with concurrent.futures.ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(crop_one, *zip(*work)))


if __name__ == "__main__":
    cases = yaml.safe_load(pathlib.Path("testcases.yml").read_text())
    manifest = yaml.safe_load((GLOBAL_INPUTS / "MANIFEST.yml").read_text())
    ldds = {k.removesuffix("/routing/ldd"): GLOBAL_INPUTS / v["dst"]
            for k, v in manifest.items() if k.endswith("/routing/ldd")}

    for name, case in sorted(cases.items()):
        lon, lat = case["outlet"]
        snap = case.get("snap", 0.25)
        search = case.get("search", [lon - 5, lon + 5, lat - 5, lat + 5])
        print(name)
        basins = {res: delineate(src, (lon, lat), snap, search) for res, src in ldds.items()}
        box = (math.floor(min(b.west for b in basins.values())),
               math.ceil(max(b.east for b in basins.values())),
               math.floor(min(b.south for b in basins.values())),
               math.ceil(max(b.north for b in basins.values())))
        for res, basin in basins.items():
            write_clone(name, res, basin, box)
        crop_case(name, manifest, box, {res: b.cell for res, b in basins.items()})
