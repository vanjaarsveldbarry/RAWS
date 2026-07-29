import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def timing_decorator(func):
    '''@timing_decorator ontop of function you want to time'''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_min = execution_time / 60
        print(f"\n{func.__name__} took: {execution_time_min:.1f}min / {execution_time:.1f}s to run ")
        return result
    return wrapper


REGIONS = ["af", "ar", "as", "au", "eu", "gr", "na", "sa", "si"]
LEVELS = range(1, 13)

# Defaults, overridable via argv[1] (standard input dir) and argv[2] (output dir).
DEFAULT_INPUT = Path(
    "/projects/prjs1222/RAWS/RAWS/output/001_choose_compression_algorithim"
    "/water_balance/input/hydrobasins/standard"
)
DEFAULT_OUTPUT = Path(
    "/projects/prjs1222/RAWS/RAWS/output/001_choose_compression_algorithim"
    "/water_balance/input/hydrobasins/processed"
)


def merge_level(standard_dir, level):
    """Concatenate all 9 continental HydroBASINS shapefiles for a single level
    into one global GeoDataFrame. A `region` column records provenance."""
    standard_dir = Path(standard_dir)
    frames = []
    for region in REGIONS:
        shp = standard_dir / region / f"hybas_{region}_lev{level:02d}_v1c.shp"
        gdf = gpd.read_file(shp, engine="pyogrio")
        gdf["region"] = region
        frames.append(gdf)

    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)


@timing_decorator
def process_all(standard_dir, out_dir):
    """Write one global GeoParquet file per level (12 files total)."""
    standard_dir = Path(standard_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in tqdm(LEVELS, desc="levels"):
        gdf = merge_level(standard_dir, level)
        out_path = out_dir / f"hybas_global_lev{level:02d}.parquet"
        gdf.to_parquet(out_path, compression="snappy", index=False)
        print(f"level {level:02d}: {len(gdf):>9,} features -> {out_path}")


if __name__ == "__main__":
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    process_all(input_dir, output_dir)
