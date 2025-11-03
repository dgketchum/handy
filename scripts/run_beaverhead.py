#!/usr/bin/env python3
import os
import sys
import logging
import argparse


def configure_logging(verbosity):
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def add_src_to_path():
    # Ensure we can import handy from src/handy without packaging
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.abspath(os.path.join(here, "..", "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HAND/REM stratification for the Lower Beaverhead River HUC-10."
    )
    parser.add_argument(
        "--huc10",
        default="1002000207",
        help="Target HUC-10 ID (default: 1002000207)",
    )
    parser.add_argument(
        "--fields",
        default=os.path.expanduser(
            "~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp"
        ),
        help="Path to statewide irrigation shapefile",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.abspath(os.getcwd()), "outputs", "beaverhead"),
        help="Output directory (default: ./outputs/beaverhead)",
    )
    parser.add_argument(
        "--rem-threshold",
        type=float,
        default=2.0,
        help="REM threshold in meters for partitioned fields (default: 2.0)",
    )
    parser.add_argument(
        "--dem-resolution",
        type=float,
        default=1.0,
        help="Target DEM resolution in meters (LiDAR if available; default: 1)",
    )
    parser.add_argument(
        "--dem-tile-max-px",
        type=int,
        default=4096,
        help="Max tile dimension in pixels for DEM requests (default: 4096)",
    )
    parser.add_argument(
        "--overwrite-dem",
        action="store_true",
        help="Overwrite cached DEM if it exists",
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v for DEBUG)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.v)
    add_src_to_path()

    # Deferred import after path setup
    from src.handy.core import run_hand_stratification, ensure_dir, LOGGER as CORE_LOGGER
    from src.handy.viz import write_interactive_map

    # Make and log output directory
    ensure_dir(args.out_dir)
    CORE_LOGGER.info("Output directory: %s", args.out_dir)

    CORE_LOGGER.info("Starting REM/HAND stratification for HUC-10 %s", args.huc10)

    results = run_hand_stratification(
        huc10=str(args.huc10),
        fields_path=os.path.expanduser(args.fields),
        out_dir=args.out_dir,
        dem_resolution=float(args.dem_resolution),
        rem_threshold=float(args.rem_threshold),
        save_rem=True,
        overwrite_dem=bool(args.overwrite_dem),
        dem_tile_max_px=int(args.dem_tile_max_px),
    )

    # Write interactive debug map with basemap switch and REM/DEM overlays
    out_html = os.path.join(args.out_dir, "debug_map.html")
    write_interactive_map(results, out_html, initial_threshold=float(args.rem_threshold))

    # Minimal terminal summary
    summary = results.get("summary", {})
    total_fields = summary.get("total_fields")
    partitioned = summary.get("partitioned")
    threshold_m = summary.get("threshold_m")

    print("--- Stratification Summary ---")
    print(f"HUC-10: {args.huc10}")
    print(f"Fields total: {total_fields}")
    print(f"Partitioned (< {threshold_m} m): {partitioned}")
    print("Outputs:")
    if results.get("rem_path"):
        print(f"  REM: {results['rem_path']}")
    if results.get("fields_out_gpkg"):
        print(f"  Stratified fields (GPKG): {results['fields_out_gpkg']}")
    if results.get("fields_out_shp"):
        print(f"  Stratified fields (SHP): {results['fields_out_shp']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("run_beaverhead").exception("Unhandled error: %s", exc)
        sys.exit(1)
