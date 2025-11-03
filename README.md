# handy
Application of Height Above Nearest Drainage algorithm to estimate groundwater interaction with riparian polygon objects (agricultural fields).

## Installation

Below are two reproducible ways to set up a working environment. Both target Python 3.10 and include a path that installs RichDEM, which we use for depression filling.

### Option A: Conda/Mamba (recommended)

- Create a fresh env (Python 3.10) with geospatial stack from conda-forge:
  - `mamba create -n handy -c conda-forge python=3.10 geopandas rasterio rioxarray xarray scipy rasterstats pynhd py3dep numpy=1.26 -y`
  - `mamba activate handy`
- Install RichDEM (requires C++ toolchain; works well in this setup):
  - `pip install -U pybind11`
  - `pip install richdem`

Notes:
- Pinning `numpy<2` (e.g., `numpy=1.26`) ensures compatibility with current RichDEM releases.
- On Debian/Ubuntu, if building from source, install OS toolchain: `sudo apt-get install -y build-essential`.

### Option B: uv + pip (lightweight, Python 3.10)

- Install Python 3.10 and create a venv:
  - `uv python install 3.10`
  - `uv venv --python 3.10 .venv && source .venv/bin/activate`
- Install dependencies and pin NumPy for RichDEM:
  - `uv pip install -U pip setuptools wheel "pybind11>=2.10" "numpy<2"`
  - `uv pip install geopandas rasterio rioxarray xarray scipy rasterstats pynhd py3dep richdem`

If you maintain a local RichDEM checkout and want to install from source instead of PyPI:
- `pip install -v --no-binary :all: /path/to/richdem/wrappers/pyrichdem`

## Usage

- Basic run (replace the `--fields` path with your dataset):
  - `python scripts/run_beaverhead.py --huc10 1002000207 --fields /path/to/fields.shp --out-dir ./outputs/beaverhead -v`
  - Note: By default, the pipeline requests ~1 m DEM (LiDAR where available). Override with `--dem-resolution` if needed.

Caching and performance
- The DEM is cached to `dem_huc10_<HUC>_<res>m.tif` in the chosen `--out-dir` and reused unless `--overwrite-dem` is provided.
- Large LiDAR requests are tiled automatically. Control the max tile size in pixels with `--dem-tile-max-px` (default 4096). If you see network timeouts, try a smaller value (e.g., 3072 or 2048).

Outputs (written to the chosen `--out-dir`):
- REM GeoTIFF, stratified fields (GPKG/SHP), and an interactive debug map (`debug_map.html`).

## Troubleshooting

- RichDEM build/import issues:
  - Ensure Python 3.10, a C++ toolchain, and `pybind11` are present.
  - If installing from PyPI, use `numpy<2` (e.g., `1.26.*`).
  - As a fallback, consider installing RichDEM from local source as shown above.
- Network access is required to fetch WBD/NHD and 3DEP DEMs.
