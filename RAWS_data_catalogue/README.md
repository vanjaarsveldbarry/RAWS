# RAWS data catalogue

One tidy, consistently named folder tree of all PCR-GLOBWB input data for RAWS,
plus small test domains cut out of it.

1. `manifest.yml` lists every dataset: where it really lives, what it should be
   called, and which ini option it feeds.
2. `build.py` turns that list into a tree of symlinks, `CATALOGUE/GLOBAL_INPUTS/`.
3. `make_testcase.py` delineates the river basins listed in `testcases.yml` and
   crops every dataset to them, giving `CATALOGUE/TEST_CASES/<name>/`.

`CATALOGUE/` is generated and is not in git. The two yml files and the scripts
are the record; anyone with access to the same file systems can rebuild it.

## Quick start

Run both steps from this folder, in this order:

`build.py` is quick: it only makes links.

`make_testcase.py` reads `CATALOGUE/GLOBAL_INPUTS/MANIFEST.yml`, so `build.py`
must have run first, and must be rerun after every change to `manifest.yml`.

`create_catalogue.sh` is meant to run both.

## What you get

```
CATALOGUE/
├── GLOBAL_INPUTS/
│   ├── MANIFEST.yml        what was built: every dataset with its path, ini_key,
│   │                       source, size and build date
│   ├── forcing/
│   ├── 30arcminutes/
│   ├── 5arcminutes/
│   └── 30arcseconds/       symlinks to the real global files
└── TEST_CASES/
    └── tugela/
        ├── clone_maps/     clone, basin mask and CDO grid file per resolution
        ├── forcing/
        ├── 30arcminutes/
        ├── 5arcminutes/
        └── 30arcseconds/   real files, cropped to the test domain
```

## Adding a dataset

Add an entry to `manifest.yml` and rerun `create_catalogue.sh`.

```yaml
5arcminutes:
  routing:
    cell_area:
      src: /scratch/.../cdo_gridarea_clone_global_05min_correct_lats.nc
      dst: cell_area_05min.nc
      ini_key: cellAreaMap
```

- `src` is where the data really lives: a file or a directory.
- `dst` is the name it gets in the catalogue.
- `ini_key` is the PCR-GLOBWB ini option, or options, it feeds.
