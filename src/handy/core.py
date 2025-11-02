import os
import sys
import logging

import numpy as np
import geopandas as gpd
import pandas as pd

import xarray as xr
import rioxarray as rxr

# Data acquisition
import py3dep

import fiona
from scipy import ndimage as ndi

# Robust imports across pynhd versions
try:
    from pynhd import WBD
except Exception:  # Some versions don't export WBD
    WBD = None
try:
    from pynhd import NHD
except Exception:
    NHD = None
try:
    from pynhd import WaterData
except Exception:
    WaterData = None

# Rasterization and transforms
from rasterio import features

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


def get_huc10_boundary(huc10):
    """
    Fetch HUC-10 boundary polygon using USGS WBD via pynhd.

    Parameters
    ----------
    huc10 : str
        HUC-10 ID, e.g., "1002000207".

    Returns
    -------
    geopandas.GeoDataFrame
        Boundary as a single-row GeoDataFrame in its native CRS.
    """
    LOGGER.info("Fetching WBD boundary for HUC-10 %s", huc10)

    gdf = None
    err = None

    # Preferred: dedicated WBD class if available
    if WBD is not None:
        try:
            wbd = WBD("huc10")
            LOGGER.debug("Using pynhd.WBD('huc10') for HUC boundary")
            try:
                gdf = wbd.byids("huc10", [huc10])
            except Exception:
                gdf = wbd.byids(ids=[huc10])
        except Exception as e:
            err = e

    # Fallback A: WaterData("wbd") with layer name variants seen in servers
    if (gdf is None or len(gdf) == 0) and WaterData is not None:
        try:
            LOGGER.debug("Using pynhd.WaterData('wbd') with layer fallback for HUC boundary")
            wd = WaterData("wbd")
            # Valid options (from your environment): wbd10, wbd12, etc.
            for layer_name in [
                "wbd10",
                "wbdhu10",
                "huc10",
            ]:
                try:
                    gtmp = wd.byid(layer=layer_name, ids=[huc10])
                    if gtmp is not None and len(gtmp) > 0:
                        gdf = gtmp
                        break
                except Exception:
                    try:
                        gtmp = wd.byids(layer=layer_name, ids=[huc10])
                        if gtmp is not None and len(gtmp) > 0:
                            gdf = gtmp
                            break
                    except Exception:
                        pass
        except Exception as e:
            err = e

    # Fallback B: Direct dataset per-layer WaterData("wbd10")
    if (gdf is None or len(gdf) == 0) and WaterData is not None:
        try:
            LOGGER.debug("Using pynhd.WaterData('wbd10') direct for HUC boundary")
            wd10 = WaterData("wbd10")
            # Try a variety of signatures
            tried_err = None
            for call in (
                lambda: wd10.byid(id=huc10),
                lambda: wd10.byid(huc10),
                lambda: wd10.byid(ids=[huc10]),
                lambda: wd10.byids(ids=[huc10]),
                lambda: wd10.byids([huc10]),
            ):
                try:
                    gtmp = call()
                    if gtmp is not None and len(gtmp) > 0:
                        gdf = gtmp
                        break
                except Exception as ce:
                    tried_err = ce
            if gdf is None or len(gdf) == 0:
                err = tried_err
        except Exception as e:
            err = e

    if gdf is None or len(gdf) == 0:
        raise RuntimeError(
            "Failed to fetch HUC-10 boundary for %s via pynhd (WBD/WaterData); last error: %s"
            % (huc10, str(err))
        )

    # Some versions return multiple features; filter exact match if the field exists
    huc_col = None
    for col in gdf.columns:
        if str(col).lower() == "huc10":
            huc_col = col
            break
    if huc_col is not None:
        gdf = gdf[gdf[huc_col].astype(str) == str(huc10)].copy()
    gdf = gdf.reset_index(drop=True)
    if len(gdf) == 0:
        raise RuntimeError("WBD did not return the requested HUC-10 polygon.")

    return gdf


def get_flowlines_within_aoi(aoi_gdf):
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
    fl = None
    err = None

    # Preferred: NHD class if available
    if NHD is not None:
        try:
            nhd = NHD("flowline")
            geom = aoi_gdf.geometry.unary_union
            try:
                fl = nhd.bygeom(geom, spatial_rel="intersects")
            except Exception:
                fl = nhd.bygeom(geom)
        except Exception as e:
            err = e

    # Fallback: WaterData with flowline network dataset
    if (fl is None or len(fl) == 0) and WaterData is not None:
        try:
            wd = WaterData("nhdflowline_network")
            geom = aoi_gdf.geometry.unary_union
            try:
                fl = wd.bygeom(geom, spatial_rel="intersects")
            except Exception:
                fl = wd.bygeom(geom)
        except Exception as e:
            err = e
    if fl is None or len(fl) == 0:
        raise RuntimeError(
            "Failed to fetch NHDPlus flowlines; last error: %s" % str(err)
        )

    # Clip to AOI explicitly to be safe
    try:
        fl = gpd.clip(fl, aoi_gdf)
    except Exception:
        fl = gpd.overlay(fl, aoi_gdf, how="intersection")

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
            df = df[df[ftype_col].astype(str).str.lower().isin(["streamriver")]  # likely error: does not guarantee perennial specifically
        ]

    if exclude_artificial and ftype_col is not None:
        drop_types = {"artificialpath", "canalditch"}
        df = df[~df[ftype_col].astype(str).str.lower().isin(drop_types)]

    return df.reset_index(drop=True)


def get_dem_for_aoi(aoi_gdf, target_crs_epsg=5070, resolution=10):
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
        "Requesting 3DEP DEM at ~%sm resolution in EPSG:%s", resolution, target_crs_epsg
    )

    target_crs = f"EPSG:{int(target_crs_epsg)}"

    # Use a WGS84 bbox as a robust input, request output in target_crs
    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")
    bbox_wgs84 = tuple(aoi_wgs84.total_bounds.tolist())

    dem = None
    err = None
    # Try the newer get_dem signature first
    try:
        dem = py3dep.get_dem(
            bbox=bbox_wgs84, resolution=resolution, crs=target_crs, align=True
        )
    except Exception as e:
        err = e
        try:
            # Fallback via get_map API
            dem = py3dep.get_map(
                "elevation", bbox_wgs84, resolution=resolution, crs=target_crs, to_raster=True
            )
        except Exception as e2:
            err = e2

    if dem is None:
        raise RuntimeError(
            "Failed to fetch DEM from 3DEP via py3dep; last error: %s" % str(err)
        )

    # Ensure DataArray has a CRS and transform; py3dep provides them via rioxarray
    dem = dem.rio.write_crs(target_crs, inplace=False)

    # Optionally clip to AOI footprint to minimize downstream computation
    try:
        dem = dem.rio.clip(aoi_gdf.to_crs(target_crs).geometry, aoi_gdf.to_crs(target_crs).crs)
    except Exception:
        # If rioxarray clip fails (non-overlapping bounds tolerance), skip exact clip and rely on bbox
        pass

    # Chunk for scalable computation; values are tuned for typical HUC-10 scale
    try:
        dem = dem.chunk({"y": 2048, "x": 2048})
    except Exception:
        # If chunking unsupported, proceed un-chunked
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

    Notes:
    - richdem expects an in-memory array (RDArray). For typical HUC-10 areas at
      10 m resolution this is feasible; we compute the DataArray to NumPy.
    - Nodata handling: we propagate nodata as NaN after filling.
    """
    LOGGER.info("Filling depressions (hydro-conditioning DEM via richdem)")

    try:
        import richdem as rd
    except Exception as e:
        raise RuntimeError(
            "richdem is required for depression filling; please install richdem"
        ) from e

    data = dem_da.data
    # Compute to NumPy if dask-backed
    try:
        dem_np = data.compute() if hasattr(data, "compute") else np.asarray(data)
    except Exception:
        dem_np = np.asarray(dem_da.values)

    # Prepare nodata
    nd = dem_da.rio.nodata
    mask_valid = np.isfinite(dem_np) if nd is None else (dem_np != nd)
    rd_nd = -999999.0
    dem_in = np.where(mask_valid, dem_np, rd_nd)

    dem_rd = rd.rdarray(dem_in, no_data=rd_nd)

    filled_rd = None
    err = None
    # Try common richdem APIs defensively across versions
    try:
        filled_rd = rd.FillDepressions(dem_rd, epsilon=True)
    except Exception as e1:
        err = e1
        try:
            filled_rd = rd.FillDepressions(dem_rd, in_place=False)
        except Exception as e2:
            err = e2
            try:
                filled_rd = rd.fill_depressions(dem_rd)
            except Exception as e3:
                err = e3
    if filled_rd is None:
        raise RuntimeError(
            "Failed to fill depressions with richdem; last error: %s" % str(err)
        )

    filled_np = np.array(filled_rd)
    # Restore nodata as NaN for downstream xarray/rioxarray friendliness
    filled_np = np.where(mask_valid, filled_np, np.nan)

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

    try:
        from scipy import ndimage as ndi
    except Exception as e:
        raise RuntimeError(
            "scipy is required for nearest-neighbor elevation sampling; install scipy"
        ) from e

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

    if str(fields_gdf.crs) != str(rem_da.rio.crs):
        raise ValueError("Fields CRS and REM CRS must match before zonal stats.")

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
                            dem_resolution=10, rem_threshold=2.0,
                            save_rem=True,
                            natural_perennial=True, exclude_artificial=False,
                            rem_method="nearest",
                            centerline_spacing_m=250.0,
                            buffer_exclude_m=50.0,
                            smooth_sigma_m=150.0,
                            save_intermediates=False):
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
        flowlines = _filter_flowlines_nhd(flowlines, natural_perennial=natural_perennial, exclude_artificial=exclude_artificial)

    # 3) DEM
    dem = get_dem_for_aoi(aoi, target_crs_epsg=5070, resolution=dem_resolution)
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
