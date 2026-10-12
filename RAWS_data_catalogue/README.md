# RAWS data catalogue

One tidy, consistently named folder tree of all PCR-GLOBWB and QUAlloc input
data for RAWS, plus small test domains cut out of it.

1. `manifest.yml` lists every dataset: where it really lives, what it should be
   called, and which ini option it feeds.
2. `build.py` turns that list into a tree of symlinks, `CATALOGUE/GLOBAL_INPUTS/`.
3. `make_testcase.py` delineates the river basins listed in `testcases.yml` and
   crops every dataset to them, giving `CATALOGUE/TEST_CASES/<name>/`.

`CATALOGUE/` is generated and is not in git. The two yml files and the scripts
are the record; anyone with access to the same file systems can rebuild it.

## Quick start

The environment is managed with [pixi](https://pixi.sh); `pixi.toml` and
`pixi.lock` pin it. Run both steps from this folder with:

```bash
./create_catalogue.sh
```

or one at a time, in this order:

```bash
pixi run python build.py
pixi run python make_testcase.py
```

`build.py` is quick: it only makes links, and removes the links of entries that
were dropped from `manifest.yml` or given a new `dst`.

`make_testcase.py` reads `CATALOGUE/GLOBAL_INPUTS/MANIFEST.yml`, so `build.py`
must have run first, and must be rerun after every change to `manifest.yml`.
It deletes and rewrites each test case in `testcases.yml` from scratch, and
reads a lot of data from `/scratch`, so it is the slow step.

## What you get

```
CATALOGUE/
├── GLOBAL_INPUTS/
│   ├── MANIFEST.yml        what was built: every dataset with its path, ini_key,
│   │                       source, size and build date
│   ├── forcing/            W5E5 and CRU forcing, and the downscaling factors
│   ├── 30arcminutes/
│   ├── 5arcminutes/
│   ├── 30arcseconds/       PCR-GLOBWB inputs per resolution
│   └── qualloc/            QUAlloc inputs, 5 arcmin only
│                           all of it symlinks to the real global files
└── TEST_CASES/
    └── <name>/             one per test case: tugela/, rhine/
        ├── clone_maps/     clone and basin mask per resolution, one table of
        │                   their grids and areas, and one map of both
        │   └── sub_basins/ if `sub_basins` is set in testcases.yml: per
        │                   <res>/ a clone and mask per sub-basin, one table
        │                   of their grids, areas, run batch, downstream
        │                   sub-basin and the outlet and inlet cells for
        │                   discharge handoff, and one map of them all
        ├── forcing/
        ├── 30arcminutes/
        ├── 5arcminutes/
        ├── 30arcseconds/
        └── qualloc/        real files, cropped to the test domain
```

## Test cases

`testcases.yml` defines the domains; the comments at the top of that file
explain the options.

- `tugela`: Tugela basin, South Africa, ~29,000 km2. One domain, no sub-basins.
- `rhine`: Rhine and Meuse, ~197,000 km2, split into up to 6 sub-basins, so it
  also has `clone_maps/sub_basins/`.


## Adding a dataset

Add an entry to `manifest.yml` and rerun `create_catalogue.sh`.

```yaml
5arcminutes:
  routing:
    cell_area:
      src: /scratch/.../cdo_gridarea_clone_global_05min_correct_lats.nc
      dst: cell_area.nc
      ini_key: cellAreaMap
```

- `src` is where the data really lives: a file or a directory.
- `dst` is the name it gets in the catalogue. The folder comes from the nesting
  in `manifest.yml`, so `dst` is only the file name.
- `ini_key` is the PCR-GLOBWB ini option, or options, it feeds.
