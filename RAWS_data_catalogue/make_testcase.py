import concurrent.futures
import ctypes
import itertools
import math
import pathlib
import shutil
import struct
import subprocess
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pcraster as pcr
import xarray as xr
import yaml
import zarr
from netCDF4 import Dataset, set_chunk_cache
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()
# Most sources hold one uncompressed global slice per chunk. With a chunk cache HDF5 reads each
# whole chunk to serve a window; without one it reads only the window's bytes.
set_chunk_cache(0, 0, 0)
BLOSC = ctypes.CDLL("libblosc.so.1")

CATALOGUE = pathlib.Path("CATALOGUE")
GLOBAL_INPUTS = CATALOGUE / "GLOBAL_INPUTS"
TEST_CASES = CATALOGUE / "TEST_CASES"
NETCDF_EXTS = (".nc", ".nc4")
PCRMAP_EXTS = (".map", ".ldd")
X_NAMES = ("lon", "longitude")
Y_NAMES = ("lat", "latitude")
EARTH_RADIUS = 6371007.181
WORKERS = 32
READ_BYTES = 256 << 20
REL_TOL = 1e-3
TOL = 1e-9
MARGIN = 1
# Row and column step to the downstream cell per LDD code.
DR = np.array([0, 1, 1, 1, 0, 0, 0, -1, -1, -1])
DC = np.array([0, -1, 0, 1, -1, 0, 1, -1, 0, 1])


class Basin(NamedTuple):
    mask: np.ndarray
    cell: float
    west: float
    east: float
    south: float
    north: float
    ldd: np.ndarray


def cell_area(lat, cell):
    top, bot = np.radians(lat + cell / 2), np.radians(lat - cell / 2)
    return EARTH_RADIUS ** 2 * math.radians(cell) * (np.sin(top) - np.sin(bot)) / 1e6


def write_table(path, rows):
    widths = [max(len(str(v)) for v in col) for col in zip(*rows)]
    path.write_text("".join("  ".join(str(v).ljust(w) for v, w in zip(row, widths)).rstrip() + "\n"
                            for row in rows))


def fill(ids, mask, ldd):
    # Cells draining to an inland pit join the zone nearest the pit, all of them, so that no flow
    # crosses between zones.
    near = pcr.spreadzone(pcr.numpy2pcr(pcr.Nominal, ids, -1), 0, 1)
    pits = pcr.pcr2numpy(pcr.catchment(ldd, pcr.ifthen(pcr.pit(ldd) != 0, near)), 0)
    return np.where(mask, np.where(ids > 0, ids, pits), 0)


def delineate(src, outlets, snap, search):
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
    acc = pcr.pcr2numpy(pcr.ifthen(pcr.pit(ldd) != 0, pcr.accuflux(ldd, 1.0)), 0)

    lon = xul + (np.arange(cols) + 0.5) * cell
    lat = yul - (np.arange(rows) + 0.5) * cell
    seed = np.zeros((rows, cols), np.int32)
    for i, (x, y) in enumerate(outlets, 1):
        near = (np.abs(lat - y) <= snap)[:, None] & (np.abs(lon - x) <= snap)[None, :]
        best = near * acc
        if not best.any():
            raise SystemExit(f"{src.name}: no LDD pit within {snap} deg of ({x}, {y})")
        orow, ocol = divmod(int(best.argmax()), cols)
        seed[orow, ocol] = i
        print(f"  {src.name}: outlet {lon[ocol]:.5f}, {lat[orow]:.5f}")
    ids = pcr.pcr2numpy(pcr.catchment(ldd, pcr.numpy2pcr(pcr.Nominal, seed, 0)), 0)
    mask = ids > 0
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise SystemExit(f"{src.name}: catchment reaches the edge of search window {search}; "
                         f"widen `search` in testcases.yml")
    # Inland pits (lakes, reservoirs) leave holes in the catchment; take them in.
    clumps = pcr.pcr2numpy(pcr.clump(pcr.numpy2pcr(pcr.Boolean, (~mask).astype(np.int32), 0)), 0)
    # The edge check above guarantees the window border is one non-catchment clump.
    mask = clumps != clumps[0, 0]
    ids = fill(ids, mask, ldd)
    codes = np.where(mask, pcr.pcr2numpy(ldd, 0), 0)

    r, c = np.nonzero(mask)
    area = cell_area(lat[r], cell).sum()
    print(f"  {src.name}: {len(r):,} cells, {area:,.0f} km2")
    crop = np.s_[r.min():r.max() + 1, c.min():c.max() + 1]
    return Basin(ids[crop], cell, xul + c.min() * cell, xul + (c.max() + 1) * cell,
                 yul - (r.max() + 1) * cell, yul - r.min() * cell,
                 # numpy2pcr misreads non-contiguous arrays.
                 codes[crop].copy())


def write_sub_basins(name, res, basin, count):
    rows, cols = basin.mask.shape
    cell = basin.cell
    pcr.setclone(rows, cols, cell, basin.west, basin.north)
    ldd = pcr.numpy2pcr(pcr.Ldd, basin.ldd, 0)
    r, c = np.indices((rows, cols))
    flows = (basin.ldd > 0) & (basin.ldd != 5)
    down = np.where(flows, (r + DR[basin.ldd]) * cols + c + DC[basin.ldd], -1).ravel()
    order = np.argsort(down)
    sdown = down[order]
    area = cell_area(basin.north - (r + 0.5) * cell, cell)
    upa = pcr.pcr2numpy(pcr.accuflux(ldd, pcr.numpy2pcr(pcr.Scalar, area, -1)), 0).ravel()

    # Pfafstetter-like: each outlet's main stem is cut at the confluences of the largest tributaries,
    # taken largest first across all outlets for as long as that gives at most `count` sub-basins.
    # A tributary is one sub-basin and its confluence splits a stretch of main stem in two.
    pos = np.full(down.size, -1)
    mains = []
    for o in range(1, basin.mask.max() + 1):
        main = [int(np.where(basin.mask.ravel() == o, upa, 0).argmax())]
        while True:
            ups = order[np.searchsorted(sdown, main[-1]):np.searchsorted(sdown, main[-1], "right")]
            if not len(ups):
                break
            main.append(int(ups[upa[ups].argmax()]))
        pos[main] = np.arange(len(main))
        mains.append(main)
    tribs = np.flatnonzero((pos < 0) & (down >= 0) & (pos[down] >= 0))
    chosen, joins = [], set()
    for t in tribs[np.argsort(-upa[tribs])]:
        if len(mains) + len(chosen) + len(joins | {down[t]}) + 1 > count:
            break
        chosen.append(t)
        joins.add(down[t])
    seed = np.zeros(down.size, np.int32)
    seed[chosen] = np.arange(1, len(chosen) + 1)
    for main in mains:
        cuts = np.sort([pos[j] for j in joins if j in main])
        seed[main] = seed.max() + 1 + np.searchsorted(cuts, np.arange(len(main)))
    labels = pcr.pcr2numpy(pcr.subcatchment(ldd, pcr.numpy2pcr(pcr.Nominal, seed.reshape(rows, cols), 0)), 0)
    labels = fill(np.where(basin.mask > 0, labels, 0), basin.mask > 0, ldd)
    masks = [labels == i for i in range(1, labels.max() + 1)]

    out = TEST_CASES / name / "clone_maps" / "sub_basins" / res
    stem = f"{name}_{res}"

    # Each sub-basin leaves through one cell, into the inlet cell of the next; a sub-basin
    # without one ends at the basin outlet. Batch k runs once batches 1..k-1 are done, so it is
    # one more than the longest chain of sub-basins upstream.
    lab = labels.ravel()
    cross = np.flatnonzero((lab > 0) & (down >= 0))
    cross = cross[lab[cross] != lab[down[cross]]]
    outlet = np.zeros(len(masks) + 1, int)
    for i, sub in enumerate(masks, 1):
        cells = np.flatnonzero(sub)
        outlet[i] = cells[upa[cells].argmax()]
    outlet[lab[cross]] = cross
    to = np.zeros(len(masks) + 1, int)
    to[lab[cross]] = lab[down[cross]]
    batch = np.ones(len(masks) + 1, int)
    for _ in masks:
        np.maximum.at(batch, to, batch + 1)
    order = sorted(range(1, len(masks) + 1), key=lambda i: batch[i])
    codes = {0: "-"}
    for b, group in itertools.groupby(order, key=lambda i: batch[i]):
        codes |= {i: f"{b}_{n}" for n, i in enumerate(group, 1)}

    table = [["batch", "code", "downstream", "outlet_lon", "outlet_lat", "inlet_lon", "inlet_lat", "xsize",
              "ysize", "xfirst", "xinc", "yfirst", "yinc", "cells", "area_km2"]]
    for i in order:
        sub = masks[i - 1]
        y, x = np.nonzero(sub)
        west, north = math.floor(basin.west + x.min() * cell), math.ceil(basin.north - y.min() * cell)
        box = (west, math.ceil(basin.west + (x.max() + 1) * cell),
               math.floor(basin.north - (y.max() + 1) * cell), north)
        ysize, xsize = write_clone(out, f"{name}_{res}_{codes[i]}", basin._replace(mask=sub), box).shape
        ends = [outlet[i], down[outlet[i]]] if to[i] else [outlet[i]]
        ends = [f"{v:.5f}" for k in ends
                for v in (basin.west + (k % cols + 0.5) * cell, basin.north - (k // cols + 0.5) * cell)]
        table.append([batch[i], codes[i], codes[to[i]], *(ends + ["-", "-"])[:4], xsize, ysize, west + cell / 2,
                      cell, north - cell / 2, -cell, len(y), f"{area[sub].sum():.0f}"])
    write_table(out / f"{stem}.txt", table)

    fig, ax = plt.subplots()
    ax.set_facecolor("lightgray")
    # Shade by batch; vmax above the last batch keeps the black labels readable on it.
    ax.imshow(np.ma.masked_equal(np.where(labels > 0, batch[labels], 0), 0),
              extent=(basin.west, basin.east, basin.south, basin.north),
              cmap="Blues", vmin=0, vmax=1.5 * batch[1:].max())
    # Borders along the cell edges where the sub-basin changes.
    y, x = np.nonzero(labels[:, 1:] != labels[:, :-1])
    segs = [[(basin.west + (j + 1) * cell, basin.north - i * cell),
             (basin.west + (j + 1) * cell, basin.north - (i + 1) * cell)] for i, j in zip(y, x)]
    y, x = np.nonzero(labels[1:] != labels[:-1])
    segs += [[(basin.west + j * cell, basin.north - (i + 1) * cell),
              (basin.west + (j + 1) * cell, basin.north - (i + 1) * cell)] for i, j in zip(y, x)]
    ax.add_collection(LineCollection(segs, colors="white", linewidths=0.5))
    for i, sub in enumerate(masks, 1):
        y, x = np.nonzero(sub)
        # Label the cell nearest the centroid, which may itself lie outside the sub-basin.
        k = np.argmin((y - y.mean()) ** 2 + (x - x.mean()) ** 2)
        ax.text(basin.west + (x[k] + 0.5) * cell, basin.north - (y[k] + 0.5) * cell, codes[i],
                fontsize=4, ha="center", va="center")
    ax.set_title(stem)
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def write_clone(out, stem, basin, box):
    west, east, south, north = box
    cell = basin.cell
    rows, cols = round((north - south) / cell), round((east - west) / cell)
    r, c = np.nonzero(basin.mask)
    mask = np.zeros((rows, cols), np.int32)
    mask[r + round((north - basin.north) / cell), c + round((basin.west - west) / cell)] = 1

    out.mkdir(parents=True, exist_ok=True)
    pcr.setclone(rows, cols, cell, west, north)
    pcr.report(pcr.numpy2pcr(pcr.Boolean, np.ones_like(mask), 0), str(out / f"{stem}.clone.map"))
    pcr.report(pcr.numpy2pcr(pcr.Boolean, mask, 0), str(out / f"{stem}.mask.map"))
    return mask


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

    # zarr reads and decompresses whole chunks, ~100 MB each at 30 arcsec for a window of under
    # 1 MB, and the NFS link is the bottleneck. blosc_getitem needs only the header, the block
    # offsets and the blocks under the window, so only those bytes are read into `comp`.
    def read_chunk(path, sel):
        first = int(np.ravel_multi_index([s.start for s in sel], var.chunks))
        n = int(np.ravel_multi_index([s.stop - 1 for s in sel], var.chunks)) - first + 1
        size = var.dtype.itemsize
        with open(path, "rb") as f:
            head = f.read(16)
            flags, nbytes, blocksize, cbytes = struct.unpack("<2xBx3I", head)
            if flags & 2:  # stored uncompressed, no block offsets
                lo, hi = 16 + first * size, 16 + (first + n) * size
            else:
                head += f.read(4 * -(-nbytes // blocksize))
                starts = np.frombuffer(head, "<u4", offset=16).astype("i8")
                order = np.sort(starts)
                ends = np.append(order, cbytes)[np.searchsorted(order, starts) + 1]
                blocks = slice(first * size // blocksize, ((first + n) * size - 1) // blocksize + 1)
                lo, hi = starts[blocks].min(), ends[blocks].max()
            comp = np.empty(cbytes, "u1")
            comp[:len(head)] = np.frombuffer(head, "u1")
            f.seek(lo)
            f.readinto(comp[lo:hi])
        full = np.empty(var.chunks, var.dtype)
        if BLOSC.blosc_getitem(ctypes.c_void_p(comp.ctypes.data), first, n,
                               ctypes.c_void_p(full.reshape(-1)[first:].ctypes.data)) != n * size:
            raise ValueError(f"{path}: blosc_getitem failed")
        return full[tuple(sel)]

    def read_step(t):
        want = [{"time": slice(t, t + 1), "lon": xs, "lat": ys}[d] for d in dims]
        out = np.empty([s.stop - s.start for s in want], var.dtype)
        for idx in itertools.product(*(range(s.start // c, (s.stop - 1) // c + 1)
                                       for s, c in zip(want, var.chunks))):
            part = [slice(max(s.start, i * c), min(s.stop, (i + 1) * c))
                    for s, i, c in zip(want, idx, var.chunks)]
            out[tuple(slice(p.start - s.start, p.stop - s.start) for p, s in zip(part, want))] = read_chunk(
                src_path / name / ".".join(map(str, idx)),
                [slice(p.start - i * c, p.stop - i * c) for p, i, c in zip(part, idx, var.chunks)])
        return out.squeeze(dims.index("time"))

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
        if native is not None and native < finest * (1 - REL_TOL):
            continue
        cell = finest
        if is_zarr(src):
            cell = next((c for c in cells.values() if abs(c - native) <= native * REL_TOL), None)
            if cell is None:
                continue
        work += [(path, TEST_CASES / name / rel / path.relative_to(src), box, cell) for path in files_in(src)]
    with concurrent.futures.ProcessPoolExecutor(WORKERS) as ex:
        list(tqdm(ex.map(crop_one, *zip(*work)), total=len(work), desc=name))


if __name__ == "__main__":
    cases = yaml.safe_load(pathlib.Path("testcases.yml").read_text())
    manifest = yaml.safe_load((GLOBAL_INPUTS / "MANIFEST.yml").read_text())
    ldds = {k.removesuffix("/routing/ldd"): GLOBAL_INPUTS / v["dst"]
            for k, v in manifest.items() if k.endswith("/routing/ldd")}

    for name, case in sorted(cases.items()):
        outlets = case["outlets"]
        if isinstance(outlets, list):
            outlets = {res: outlets for res in ldds}
        lons, lats = zip(*(p for pts in outlets.values() for p in pts))
        snap = case.get("snap", 0.25)
        search = case.get("search", [min(lons) - 5, max(lons) + 5, min(lats) - 5, max(lats) + 5])
        print(name)
        shutil.rmtree(TEST_CASES / name, ignore_errors=True)
        basins = {res: delineate(src, outlets[res], snap, search) for res, src in ldds.items()}
        box = (math.floor(min(b.west for b in basins.values())),
               math.ceil(max(b.east for b in basins.values())),
               math.floor(min(b.south for b in basins.values())),
               math.ceil(max(b.north for b in basins.values())))
        out = TEST_CASES / name / "clone_maps"
        table = [["resolution", "xsize", "ysize", "xfirst", "xinc", "yfirst", "yinc", "cells", "area_km2"]]
        fig, axs = plt.subplots(1, len(basins), figsize=(6 * len(basins), 5))
        for ax, (res, basin) in zip(axs, basins.items()):
            cell = basin.cell
            mask = write_clone(out, f"{name}_{res}", basin, box)
            r, _ = np.nonzero(basin.mask)
            area = cell_area(basin.north - (r + 0.5) * cell, cell).sum()
            table.append([res, mask.shape[1], mask.shape[0], box[0] + cell / 2, cell, box[3] - cell / 2,
                         -cell, len(r), f"{area:.0f}"])
            ax.set_facecolor("lightgray")
            ax.imshow(np.ma.masked_equal(mask, 0), extent=box, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(res)
            if case.get("sub_basins"):
                write_sub_basins(name, res, basin, case["sub_basins"])
        write_table(out / f"{name}.txt", table)
        fig.suptitle(name)
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        crop_case(name, manifest, box, {res: b.cell for res, b in basins.items()})
