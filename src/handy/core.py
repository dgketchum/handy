import os
import sys
import logging

import numpy as np
import geopandas as gpd
import pandas as pd

import xarray as xr
import rioxarray as rxr
from shapely.geometry import box as shapely_box

# Data acquisition
import py3dep

import fiona
from scipy import ndimage as ndi
import richdem as rd
from pyproj import CRS as _CRS

# Pynhd imports (local version: NHD and WaterData are available; no WBD class)
from pynhd import NHD, WaterData

# Rasterization and transforms
from rasterio import features
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from affine import Affine

# Zonal statistics
from rasterstats import zonal_stats

LOGGER = logging.getLogger("handy.core")


def ensure_dir(path):
    """
    Create a directory if it doesn't exist.
    Uses os.path.* per constraints.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_huc10_boundary(huc10: str) -> gpd.GeoDataFrame:
    """Fetch HUC-10 boundary polygon using WaterData WBD10 via pynhd.

    Parameters
    ----------
    huc10 : str
        HUC-10 ID, e.g., "1002000207".

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame for the requested HUC-10.
    """
    LOGGER.info("Fetching WBD10 boundary for HUC-10 %s", huc10)
    wd = WaterData("wbd10")
    gdf = wd.byid("huc10", str(huc10))
    # Ensure exact match and canonicalize index
    if "huc10" in [c.lower() for c in gdf.columns]:
        # Find actual column name case
        col = next(c for c in gdf.columns if c.lower() == "huc10")
        gdf = gdf[gdf[col].astype(str) == str(huc10)]
    return gdf.reset_index(drop=True)


def get_flowlines_within_aoi(aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fetch NHDPlus flowlines intersecting the AOI using pynhd.

    Parameters
    ----------
    aoi_gdf : GeoDataFrame
        AOI polygon, any CRS.

    Returns
    -------
    GeoDataFrame
        Flowlines clipped to AOI (in the service/native CRS). Caller should reproject to target raster CRS.
    """
    LOGGER.info("Fetching NHDPlus flowlines within AOI")
    # Use NHD high-resolution flowlines and intersect by geometry in WGS84
    geom = aoi_gdf.to_crs(4326).geometry.unary_union
    nhd = NHD("flowline_hr")
    fl = nhd.bygeom(geom, geo_crs=4326)

    # Clip to AOI explicitly to be safe
    fl = gpd.clip(fl, aoi_gdf)

    fl = fl.reset_index(drop=True)
    return fl


def _filter_flowlines_nhd(fl, natural_perennial=False, exclude_artificial=False):
    """
    Optionally filter NHD flowlines to natural perennial streams and/or exclude artificial paths.
    Uses robust attribute lookups across common NHD schema variants.
    """
    if len(fl) == 0:
        return fl
    df = fl.copy()

    # Common attribute names
    fcode_col = None
    for c in ("FCODE", "FCode", "fcode"):
        if c in df.columns:
            fcode_col = c
            break
    ftype_col = None
    for c in ("FTYPE", "FType", "ftype"):
        if c in df.columns:
            ftype_col = c
            break

    # Perennial stream/river FCODE in NHD is typically 46006.
    if natural_perennial:
        if fcode_col is not None:
            df = df[df[fcode_col] == 46006]
        elif ftype_col is not None:
            df = df[df[ftype_col].astype(str).str.lower().isin(["streamriver"])]

    if exclude_artificial and ftype_col is not None:
        drop_types = {"artificialpath", "canalditch"}
        df = df[~df[ftype_col].astype(str).str.lower().isin(drop_types)]

    return df.reset_index(drop=True)


def get_dem_for_aoi(
    aoi_gdf,
    target_crs_epsg=5070,
    resolution=1,
    cache_path=None,
    overwrite=False,
    tile_max_px=4096,
):
    """
    Download USGS 3DEP DEM for the AOI using py3dep.

    Notes on coordinate systems and resolution:
    - To ensure the DEM cell size is in meters, we request the output DEM in a
      projected CRS (default EPSG:5070 - CONUS Albers Equal Area). When using a
      projected CRS, py3dep's `resolution` is interpreted in meters.
    - We request by bounding box in WGS84 or AOI-native depending on what API
      variant is available in the installed py3dep version, with robust fallbacks.

    Parameters
    ----------
    aoi_gdf : GeoDataFrame
        AOI polygon.
    target_crs_epsg : int
        EPSG of projected CRS for output DEM.
    resolution : int or float
        Desired DEM resolution in target CRS units (meters).

    Returns
    -------
    xarray.DataArray
        DEM with rioxarray CRS/transform set (in target_crs_epsg), masked to AOI extent.
    """
    LOGGER.info(
        "Requesting 3DEP DEM at ~%sm (LiDAR where available) in EPSG:%s",
        resolution,
        target_crs_epsg,
    )

    target_crs = f"EPSG:{int(target_crs_epsg)}"

    # Cached DEM
    if cache_path and (not overwrite) and os.path.exists(cache_path):
        LOGGER.info("Loading cached DEM: %s", cache_path)
        dem_cached = rxr.open_rasterio(cache_path)
        if "band" in dem_cached.dims:
            dem_cached = dem_cached.squeeze("band", drop=True)
        return dem_cached

    # Tiling in projected CRS to limit max pixels per request
    aoi_proj = aoi_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = aoi_proj.total_bounds.tolist()

    width_m = maxx - minx
    height_m = maxy - miny
    tile_px = int(tile_max_px)
    nx = max(1, int(np.ceil((width_m / float(resolution)) / float(tile_px))))
    ny = max(1, int(np.ceil((height_m / float(resolution)) / float(tile_px))))
    dx = width_m / float(nx)
    dy = height_m / float(ny)

    tiles = []
    for ix in range(nx):
        for iy in range(ny):
            x0 = minx + ix * dx
            x1 = minx + (ix + 1) * dx
            y0 = miny + iy * dy
            y1 = miny + (iy + 1) * dy
            tiles.append((x0, y0, x1, y1))

    parts = []
    for (x0, y0, x1, y1) in tiles:
        geom = (x0, y0, x1, y1)
        da = py3dep.get_dem(geom, int(resolution), crs=int(target_crs_epsg))
        # Squeeze band if present
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)
        # Ensure merge-friendly orientation (positive pixel height)
        tr = da.rio.transform()
        if tr.e < 0:
            h = da.sizes["y"]
            # flip along y dimension
            da = da.isel(y=slice(None, None, -1))
            new_tr = Affine(tr.a, tr.b, tr.c, tr.d, -tr.e, tr.f + tr.e * (h - 1))
            da = da.rio.write_transform(new_tr)
        parts.append(da)

    # Mosaic tiles using rasterio.merge with in-memory GTiffs to avoid
    # orientation issues during merge.

    srcs = []
    mems = []
    for da in parts:
        arr = np.asarray(da.data).astype("float32")
        tr = da.rio.transform()
        crs = da.rio.crs
        profile = {
            "driver": "GTiff",
            "height": arr.shape[-2],
            "width": arr.shape[-1],
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": tr,
            # Do not set nodata to NaN; leave undefined so merge treats NaNs as data mask
        }
        mem = MemoryFile()
        mems.append(mem)
        ds = mem.open(**profile)
        ds.write(arr, 1)
        srcs.append(ds)

    mosaic, mosaic_tr = rio_merge(srcs)
    # Close datasets but keep MemoryFiles alive until after DataArray creation
    for ds in srcs:
        try:
            ds.close()
        except Exception:
            pass

    dem = xr.DataArray(
        mosaic[0],
        dims=("y", "x"),
        name="elevation",
    )
    dem = dem.rio.write_crs(parts[0].rio.crs, inplace=False)
    dem = dem.rio.write_transform(mosaic_tr)
    if str(dem.rio.crs) != target_crs:
        dem = dem.rio.reproject(target_crs)

    # Optionally clip to AOI footprint to minimize downstream computation
    dem = dem.rio.clip(aoi_gdf.to_crs(target_crs).geometry, aoi_gdf.to_crs(target_crs).crs)

    # Save cache if requested
    if cache_path:
        ensure_dir(os.path.dirname(cache_path))
        LOGGER.info("Saving DEM cache: %s", cache_path)
        dem.rio.to_raster(cache_path)

    # Chunk for scalable computation; values are tuned for typical HUC-10 scale
    try:
        dem = dem.chunk({"y": 2048, "x": 2048})
    except Exception:
        pass

    return dem


def rasterize_lines_to_grid(lines_gdf, template_da, burn_value=1):
    """
    Rasterize line features to match a template DataArray grid.

    Parameters
    ----------
    lines_gdf : GeoDataFrame
        Line geometries in the same CRS as `template_da`.
    template_da : xarray.DataArray
        Raster providing transform, shape, and CRS.
    burn_value : numeric
        Value to burn for stream cells; others get 0.

    Returns
    -------
    xarray.DataArray
        Boolean/int mask DataArray with stream cells burned as `burn_value`.
    """
    transform = template_da.rio.transform()
    shape = template_da.shape

    # Prepare shapes iterable for rasterize: (geometry, value)
    shapes = [(geom, burn_value) for geom in lines_gdf.geometry if geom is not None]
    if len(shapes) == 0:
        raise ValueError("No valid geometries found to rasterize.")

    stream_arr = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    stream_da = xr.DataArray(
        stream_arr,
        dims=template_da.dims,
        coords=template_da.coords,
        name="streams",
    )
    stream_da = stream_da.rio.write_crs(template_da.rio.crs, inplace=False)
    return stream_da


def fill_sinks(dem_da):
    """
    Hydro-condition the DEM by filling depressions using richdem.

    This version uses the local RichDEM Python API directly with no fallbacks
    or exception handling so errors surface clearly.
    """
    LOGGER.info("Filling depressions (hydro-conditioning DEM via richdem)")

    # Materialize DEM to NumPy and cast to float32 (RichDEM supported type)
    data = dem_da.data
    dem_np = data.compute() if hasattr(data, "compute") else np.asarray(data)
    dem_np = dem_np.astype("float32", copy=False)

    # Build rdarray with an explicit no_data, and supply geotransform
    nd = dem_da.rio.nodata
    if nd is None or not np.isfinite(nd):
        nd = -9999.0
    # Replace NaNs with nd for RichDEM
    dem_in = np.where(np.isfinite(dem_np), dem_np, nd).astype("float32", copy=False)

    # Construct geotransform from rasterio Affine
    gt = dem_da.rio.transform()
    geotransform = [gt.c, gt.a, gt.b, gt.f, gt.d, gt.e]

    dem_rd = rd.rdarray(dem_in, no_data=float(nd), geotransform=geotransform)

    # Fill depressions with epsilon gradient to avoid flats
    filled_rd = rd.FillDepressions(dem_rd, epsilon=True, in_place=False, topology="D8")

    filled_np = np.asarray(filled_rd, dtype="float32")
    # Restore nodata to NaN for downstream processing
    filled_np = np.where(filled_np == float(nd), np.nan, filled_np)

    filled_da = xr.DataArray(
        filled_np,
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="DEM_filled",
        attrs={"description": "Hydro-conditioned DEM (depressions filled via richdem)"},
    )
    filled_da = filled_da.rio.write_crs(dem_da.rio.crs, inplace=False)
    return filled_da


def compute_rem_from_streams(dem_da, streams_da):
    """
    Compute Relative Elevation Model (REM, akin to HAND) relative to nearest stream.

    Approach:
    - Hydro-condition the DEM (fill depressions) to avoid artificial sinks.
    - Rasterize streams on the same grid; identify stream cells.
    - For each cell, find the nearest stream cell in Euclidean pixel space and
      subtract that stream cell's elevation from the cell's elevation.

    This produces a "detrended DEM" where values are height above the nearest
    stream centerline, which is a practical proxy for HAND when flowpath-based
    HAND is not available.

    Notes:
    - This method uses an Euclidean nearest-neighbor lookup. It is not identical
      to a flowpath HAND, but in floodplain/valley bottoms it tracks relative
      relief robustly for stratification tasks.
    - For large rasters, we convert to in-memory NumPy arrays for the nearest
      neighbor operation. This is typically manageable at HUC-10 scales at 10 m.
    """
    LOGGER.info("Computing REM relative to nearest stream")

    # Ensure matching grid/CRS
    if dem_da.rio.crs is None or streams_da.rio.crs is None:
        raise ValueError("Both DEM and streams rasters must have a valid CRS.")
    if str(dem_da.rio.crs) != str(streams_da.rio.crs) or dem_da.shape != streams_da.shape:
        raise ValueError("DEM and streams must share grid shape and CRS.")

    # Hydro-conditioning
    dem_filled = fill_sinks(dem_da)

    # Materialize as NumPy arrays for nearest-neighbor distance transform
    dem_np = np.asarray(dem_filled.data)
    streams_np = np.asarray(streams_da.data).astype(bool)

    if streams_np.sum() == 0:
        raise ValueError("Stream mask has no active cells after rasterization.")

    # Compute indices of nearest stream cell for each pixel
    # distance_transform_edt on ~stream cells returns indices to the nearest True
    # pixels in streams_np when using return_indices with ~streams_np as input.
    indices = ndi.distance_transform_edt(
        ~streams_np, return_distances=False, return_indices=True
    )
    row_ind, col_ind = indices
    base_elev = dem_np[row_ind, col_ind]

    rem_np = dem_np - base_elev
    # Height Above Nearest Drainage should be non-negative
    rem_np = np.where(rem_np < 0, 0, rem_np)

    rem_da = xr.DataArray(
        rem_np,
        dims=dem_filled.dims,
        coords=dem_filled.coords,
        name="REM",
        attrs={"description": "Relative Elevation Model (height above nearest stream)"},
    )
    rem_da = rem_da.rio.write_crs(dem_filled.rio.crs, inplace=False)
    return rem_da


def compute_rem_centerline(dem_da, flowlines_gdf, spacing_m=250.0, buffer_exclude_m=50.0, smooth_sigma_m=150.0):
    """
    Approximate the report's detrending approach using densified centerline points and kernel smoothing.

    Steps:
    - Fill sinks in DEM (hydro-conditioning).
    - Densify flowlines to points at `spacing_m`.
    - Burn point locations to a raster grid and assign DEM elevations at those cells.
    - Expand to full grid by nearest-point assignment via distance transform.
    - Apply Gaussian smoothing (sigma in meters -> pixels) to the water-surface raster.
    - Within `buffer_exclude_m` of streams, keep the unsmoothed base surface to avoid over-smoothing at channels.
    - Compute REM = filled DEM - smoothed water surface, clamped to [0, inf).
    """
    if dem_da.rio.crs is None or flowlines_gdf.crs is None:
        raise ValueError("DEM and flowlines must have a valid CRS.")
    if str(flowlines_gdf.crs) != str(dem_da.rio.crs):
        flowlines_gdf = flowlines_gdf.to_crs(dem_da.rio.crs)

    dem_filled = fill_sinks(dem_da)

    # Grid/transform
    transform = dem_filled.rio.transform()
    shape = dem_filled.shape
    try:
        resx, resy = dem_filled.rio.resolution()
    except Exception:
        # Fallback to coordinate diffs
        resx = float(abs(dem_filled.x[1] - dem_filled.x[0]))  # likely error: assumes regularly spaced coords
        resy = float(abs(dem_filled.y[1] - dem_filled.y[0]))
    cell_m = max(abs(resx), abs(resy))

    # Densify lines to points
    pts = []
    for geom in flowlines_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        length = float(geom.length)
        if length <= 0:
            continue
        # Sample along line at [0, spacing, 2*spacing, ...]
        dists = np.arange(0.0, max(length, 0.0), float(spacing_m))
        for d in dists:
            try:
                p = geom.interpolate(d)
                if p is not None and not p.is_empty:
                    pts.append(p)
            except Exception:
                continue
        # Include endpoint to ensure coverage
        try:
            p_end = geom.interpolate(length)
            if p_end is not None and not p_end.is_empty:
                pts.append(p_end)
        except Exception:
            pass

    if len(pts) == 0:
        raise ValueError("No densified points generated from flowlines.")

    # Rasterize densified points as a mask
    point_shapes = [(p, 1) for p in pts]
    point_mask = features.rasterize(
        shapes=point_shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    # DEM to NumPy
    dem_np = np.asarray(dem_filled.data)

    # Assign DEM elevation to densified point cells; others NaN
    points_elev = np.full(shape, np.nan, dtype="float32")
    if point_mask.sum() == 0:
        raise ValueError("Rasterized densified point mask is empty.")
    points_elev[point_mask] = dem_np[point_mask]

    # Nearest-point expansion via distance transform
    row_ind, col_ind = ndi.distance_transform_edt(
        ~point_mask, return_distances=False, return_indices=True
    )
    base_surface = points_elev[row_ind, col_ind]

    # Streams raster and distance map for buffer logic
    streams_da = rasterize_lines_to_grid(flowlines_gdf, dem_filled, burn_value=1)
    streams_mask = np.asarray(streams_da.data).astype(bool)
    dist_m = ndi.distance_transform_edt(~streams_mask) * float(cell_m)

    # Gaussian smoothing (meters -> pixels)
    sigma_px = max(1.0, float(smooth_sigma_m) / float(cell_m))
    water_smooth = ndi.gaussian_filter(base_surface.astype("float32"), sigma=sigma_px, mode="nearest")

    # Keep unsmoothed near channels
    water_surface = np.where(dist_m < float(buffer_exclude_m), base_surface, water_smooth)

    rem_np = dem_np - water_surface
    rem_np = np.where(rem_np < 0, 0, rem_np)

    rem_da = xr.DataArray(
        rem_np,
        dims=dem_filled.dims,
        coords=dem_filled.coords,
        name="REM",
        attrs={"description": "Relative Elevation Model (kernel-smoothed base surface from centerlines)"},
    )
    rem_da = rem_da.rio.write_crs(dem_filled.rio.crs, inplace=False)
    return rem_da


def load_and_clip_fields(fields_path, aoi_gdf, target_crs):
    """
    Load the statewide irrigation dataset and clip to AOI, reprojecting to target CRS.

    Parameters
    ----------
    fields_path : str
        Path to the statewide irrigation shapefile.
    aoi_gdf : GeoDataFrame
        AOI polygon.
    target_crs : str or dict
        CRS to project the output to (e.g., DEM CRS).

    Returns
    -------
    GeoDataFrame
        Clipped fields in target_crs.
    """
    LOGGER.info("Loading irrigation dataset: %s", fields_path)
    # Assume path exists; if wrong, let it fail upstream per instructions.
    fields = None
    try:
        # Discover file CRS and read with AOI bbox in that CRS to avoid full load
        with fiona.open(fields_path) as src:
            fields_crs = src.crs_wkt if src.crs_wkt else src.crs
        aoi_in_fields = aoi_gdf.to_crs(fields_crs)
        bounds = tuple(aoi_in_fields.total_bounds.tolist())
        fields = gpd.read_file(fields_path, bbox=bounds)
    except Exception:
        fields = gpd.read_file(fields_path)  # fallback to full read
    # Ensure valid geometries; drop empties
    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()].copy()

    LOGGER.info("Clipping irrigation dataset to AOI")
    try:
        clipped = gpd.clip(fields, aoi_gdf)
    except Exception:
        clipped = gpd.overlay(fields, aoi_gdf, how="intersection")

    clipped = clipped.to_crs(target_crs)
    clipped = clipped.reset_index(drop=True)
    return clipped


def compute_field_rem_stats(fields_gdf, rem_da, stats=("mean",)):
    """
    Compute zonal statistics of REM over irrigation polygons.

    Parameters
    ----------
    fields_gdf : GeoDataFrame
        Polygons reprojected to the same CRS as rem_da.
    rem_da : xarray.DataArray
        REM raster with CRS/transform.
    stats : tuple of str
        Statistics to compute via rasterstats (default: mean).

    Returns
    -------
    GeoDataFrame
        Input fields_gdf with added columns for each requested stat, prefixed by 'rem_'.
    """
    LOGGER.info("Computing zonal statistics over fields (stats: %s)", ",".join(stats))

    # Ensure CRS consistency; reproject fields to REM CRS if needed.
    rem_crs = rem_da.rio.crs
    if rem_crs is None:
        raise ValueError("REM raster has no CRS; cannot compute zonal stats.")
    if fields_gdf.crs is None:
        raise ValueError("Fields GeoDataFrame has no CRS; cannot compute zonal stats.")
    try:
        if not _CRS.from_user_input(fields_gdf.crs).equals(_CRS.from_user_input(rem_crs)):
            fields_gdf = fields_gdf.to_crs(rem_crs)
    except Exception:
        if str(fields_gdf.crs) != str(rem_crs):
            fields_gdf = fields_gdf.to_crs(rem_crs)

    affine = rem_da.rio.transform()
    raster = np.asarray(rem_da.data)
    # Mask NaNs so mean ignores nodata
    raster = np.ma.array(raster, mask=~np.isfinite(raster))

    zs = zonal_stats(
        vectors=fields_gdf.geometry,
        raster=raster,
        affine=affine,
        stats=list(stats),
        nodata=None,
        all_touched=True,
        geojson_out=False,
    )

    df_stats = pd.DataFrame(zs)
    # Prefix columns for clarity
    df_stats = df_stats.rename(columns={s: f"rem_{s}" for s in df_stats.columns})
    result = fields_gdf.reset_index(drop=True).join(df_stats)
    return result


def stratify_fields_by_rem(fields_with_stats_gdf, threshold_m=2.0):
    """
    Add a boolean field 'partitioned' where mean REM < threshold.

    Parameters
    ----------
    fields_with_stats_gdf : GeoDataFrame
        Fields with at least 'rem_mean' column.
    threshold_m : float
        Relative elevation threshold in meters for partitioned fields.

    Returns
    -------
    GeoDataFrame
        Input with added 'partitioned' bool flag.
    """
    if "rem_mean" not in fields_with_stats_gdf.columns:
        raise ValueError("'rem_mean' column not found; compute stats first.")
    out = fields_with_stats_gdf.copy()
    out["partitioned"] = out["rem_mean"] < float(threshold_m)
    return out


def run_hand_stratification(huc10, fields_path, out_dir,
                            dem_resolution=1, rem_threshold=2.0,
                            save_rem=True,
                            natural_perennial=True, exclude_artificial=False,
                            rem_method="nearest",
                            centerline_spacing_m=250.0,
                            buffer_exclude_m=50.0,
                            smooth_sigma_m=150.0,
                            save_intermediates=False,
                            overwrite_dem=False,
                            dem_tile_max_px=4096):
    """
    Orchestrate REM/HAND computation and field stratification over a HUC-10.

    Workflow:
    1) Fetch HUC-10 boundary.
    2) Fetch NHDPlus flowlines within boundary.
    3) Download 3DEP DEM in a projected CRS (EPSG:5070) at requested resolution.
    4) Reproject AOI and flowlines to DEM CRS; rasterize flowlines to DEM grid.
    5) Hydro-condition DEM and compute REM (elevation above nearest stream).
    6) Load statewide irrigation dataset, clip to AOI, reproject to DEM CRS.
    7) Zonal stats of REM over fields; stratify by threshold (< 2 m => partitioned).
    8) Save outputs to out_dir.

    Returns a dictionary of key artifacts.
    """
    ensure_dir(out_dir)

    # 1) AOI boundary (WBD)
    aoi = get_huc10_boundary(huc10)

    # 2) Flowlines
    flowlines = get_flowlines_within_aoi(aoi)
    if natural_perennial or exclude_artificial:
        flowlines = _filter_flowlines_nhd(flowlines, natural_perennial=natural_perennial,
                                          exclude_artificial=exclude_artificial)

    # 3) DEM
    dem_cache = os.path.join(out_dir, f"dem_huc10_{huc10}_{int(dem_resolution)}m.tif")
    dem = get_dem_for_aoi(
        aoi,
        target_crs_epsg=5070,
        resolution=dem_resolution,
        cache_path=dem_cache,
        overwrite=overwrite_dem,
        tile_max_px=int(dem_tile_max_px),
    )
    dem_crs = dem.rio.crs

    # 4) Reproject AOI + flowlines to DEM CRS and rasterize streams
    aoi_dem = aoi.to_crs(dem_crs)
    flowlines_dem = flowlines.to_crs(dem_crs)
    streams_da = rasterize_lines_to_grid(flowlines_dem, dem, burn_value=1)

    # 5) Compute REM
    if str(rem_method).lower() in ("nearest", "default"):
        rem = compute_rem_from_streams(dem, streams_da)
    elif str(rem_method).lower() in ("kernel", "centerline"):
        rem = compute_rem_centerline(
            dem, flowlines_dem,
            spacing_m=float(centerline_spacing_m),
            buffer_exclude_m=float(buffer_exclude_m),
            smooth_sigma_m=float(smooth_sigma_m),
        )
    else:
        raise ValueError("Unknown rem_method. Use 'nearest' or 'kernel'.")

    # Save REM for inspection if requested
    rem_path = None
    if save_rem:
        rem_path = os.path.join(out_dir, f"rem_huc10_{huc10}.tif")
        LOGGER.info("Saving REM raster: %s", rem_path)
        rem.rio.to_raster(rem_path)

    # Optional intermediates for QA
    aoi_path = None
    flowlines_path = None
    streams_path = None
    if save_intermediates:
        try:
            aoi_path = os.path.join(out_dir, f"aoi_huc10_{huc10}.gpkg")
            aoi_dem.to_file(aoi_path, driver="GPKG")
        except Exception:
            aoi_path = None
        try:
            flowlines_path = os.path.join(out_dir, f"flowlines_huc10_{huc10}.gpkg")
            flowlines_dem.to_file(flowlines_path, driver="GPKG")
        except Exception:
            flowlines_path = None
        try:
            streams_path = os.path.join(out_dir, f"streams_huc10_{huc10}.tif")
            streams_da.rio.to_raster(streams_path)
        except Exception:
            streams_path = None

    # 6) Fields (clip + reproject)
    fields = load_and_clip_fields(fields_path, aoi, dem_crs)

    # 7) Zonal stats and stratification
    fields_stats = compute_field_rem_stats(fields, rem, stats=("mean",))
    fields_strat = stratify_fields_by_rem(fields_stats, threshold_m=rem_threshold)

    # 8) Save outputs
    fields_out_gpkg = os.path.join(out_dir, f"fields_stratified_huc10_{huc10}.gpkg")
    LOGGER.info("Saving stratified fields: %s", fields_out_gpkg)
    fields_strat.to_file(fields_out_gpkg, driver="GPKG")

    # Shapefile optional (field name limits apply); keep concise names
    try:
        fields_out_shp = os.path.join(out_dir, f"fields_stratified_huc10_{huc10}.shp")
        fields_strat.to_file(fields_out_shp)
    except Exception:
        fields_out_shp = None

    # Quick summary
    total = len(fields_strat)
    part = int(fields_strat[fields_strat["partitioned"]].shape[0])
    LOGGER.info("Partitioned fields (< %.2fm): %s / %s", rem_threshold, part, total)

    return {
        "aoi": aoi,
        "flowlines": flowlines,
        "dem": dem,
        "streams": streams_da,
        "rem": rem,
        "fields": fields,
        "fields_stats": fields_stats,
        "fields_strat": fields_strat,
        "rem_path": rem_path,
        "aoi_path": aoi_path,
        "flowlines_path": flowlines_path,
        "streams_path": streams_path,
        "fields_out_gpkg": fields_out_gpkg,
        "fields_out_shp": fields_out_shp,
        "summary": {"total_fields": total, "partitioned": part, "threshold_m": rem_threshold},
    }
