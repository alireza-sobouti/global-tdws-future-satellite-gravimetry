#!/usr/bin/env python3
"""Extract basin runoff and gridded TWSA for simulated gravity missions.

Author
------
Alireza Sobouti
Institute of Geodesy, University of Stuttgart, Stuttgart, Germany
ESA SING project | 2026

Associated research
-------------------
"Toward Global Estimation of Total Drainable Water Storage with
Next-Generation Satellite Gravimetry"

Purpose
-------
Prepare basin-level hydrological inputs for comparing reference terrestrial
water storage anomalies (TWSA) with simulated GRACE-C-like, NGGM, and MAGIC
satellite-gravimetry observations. This script EXTRACTS input time series;
it does not estimate total drainable water storage (TDWS) or hydraulic time
constants.

Processing workflow
-------------------
1. Read basin outlet coordinates from a CSV and polygons from a shapefile.
2. Associate each outlet with a polygon, using point-in-polygon matching
   first and progressively less strict basin-name matching as a fallback.
3. Select the discharge grid cell with the highest temporal-mean flow in a
   5 x 5 neighbourhood around the nominal outlet. Reuse that cell for all
   discharge files, subject to consistent grids across files.
4. Convert daily mean discharge (m^3/s) to monthly runoff depth (mm/month):
       runoff_mm = sum(daily_discharge_m3_s * 86400) / area_m2 * 1000
5. Use a fractional basin mask to export the TWSA time series of individual
   grid cells and their in-basin areas for the four mission configurations.

Input data
----------
* Basin CSV: a name ("name" or "basin_name"), "lon", and "lat" columns.
* Basin shapefile: "NAME", "RASTAREA" (km^2), and polygon geometry.
* Daily discharge NetCDFs: "dis24" (m^3/s), "longitude", "latitude", and
  "valid_time". Default filename year selection is 1995-2006 (ESM 2.0).
  The one-day adjustment to "valid_time" is retained from the study script;
  verify it is appropriate for your particular discharge files.
* Simulated gravity NetCDF: read by the project's ``read_mag_l3`` helper.
  The helper must return a dictionary with "lat", "lon", "time", and "data"
  (TWSA array ordered time x latitude x longitude).
* Two project-specific functions are required but not distributed here:
  ``data_io.read_sing.read_mag_l3`` and
  ``utils.basin_mask.grid_basin_mask``. Supply the original implementations
  through --helpers-root. The latter must return a dictionary with a
  "centers" DataFrame containing "Lat", "Lon", and "Area_km2".

Outputs
-------
For each basin, creates only:
    OUT_DIR/BASIN/runoff.csv
    OUT_DIR/BASIN/Reference/filtered/twsa_pixels_timeseries.csv
    OUT_DIR/BASIN/GRACE-C/filtered/twsa_pixels_timeseries.csv
    OUT_DIR/BASIN/NGGM/filtered/twsa_pixels_timeseries.csv
    OUT_DIR/BASIN/MAGIC/filtered/twsa_pixels_timeseries.csv

Each TWSA CSV has one row per intersecting grid cell, with columns for
longitude, latitude, intersected basin area (km^2), and monthly TWSA dates.
The runoff CSV has monthly runoff depth and the number of available daily
records. Months with incomplete daily coverage are not gap-filled.

Example
-------
python basin_discharge_twsa_extraction.py \
    --basins-csv path/to/basins_meta.csv \
    --basins-shp path/to/basin.shp \
    --reference-nc path/to/reference_product.nc \
    --discharge-dir path/to/discharge \
    --helpers-root path/to/original/project \
    --out-dir outputs

To process only one basin, add: --basin "CONGO"
To print command-line options: python basin_discharge_twsa_extraction.py --help

Data record: https://doi.org/10.5281/zenodo.24404298
The Zenodo record contains derived TDWS and hydraulic time constants, not
necessarily the raw model inputs required to run this extraction script.

Note: This is a cleaned adaptation of the research script. Its numerical
results cannot be independently reproduced until the two project helpers
and original input files are supplied. No software license is asserted here.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from tqdm import tqdm


MISSIONS = ("Reference", "GRACE-C", "NGGM", "MAGIC")
DISCHARGE_VARIABLE = "dis24"
MAX_CONSECUTIVE_NODATA_FILES = 5
OUTLET_SEARCH_RADIUS_CELLS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--basins-csv", type=Path, required=True,
                        help="CSV with basin names and outlet coordinates")
    parser.add_argument("--basins-shp", type=Path, required=True,
                        help="Basin polygon shapefile")
    parser.add_argument("--reference-nc", type=Path, required=True,
                        help="Reference MAG-L3 NetCDF passed to read_mag_l3")
    parser.add_argument("--discharge-dir", type=Path, required=True,
                        help="Directory containing daily discharge NetCDF files")
    parser.add_argument("--helpers-root", type=Path,
                        help="Parent directory of data_io/ and utils/ helper modules")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--basin", type=str,
                        help="Optional case-insensitive substring filter for basin name")
    parser.add_argument("--start-year", type=int, default=1995)
    parser.add_argument("--end-year", type=int, default=2006)
    parser.add_argument("--filter", default="filtered",
                        help="Processing variant supplied to read_mag_l3")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate previously written CSV files")
    return parser.parse_args()


def load_project_helpers(root: Path | None):
    if root is not None:
        if not root.is_dir():
            raise FileNotFoundError(f"Helper root does not exist: {root}")
        sys.path.insert(0, str(root.resolve()))
    try:
        read_mag_l3 = importlib.import_module("data_io.read_sing").read_mag_l3
        grid_basin_mask = importlib.import_module("utils.basin_mask").grid_basin_mask
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Missing project helper modules: data_io.read_sing.read_mag_l3 and "
            "utils.basin_mask.grid_basin_mask. Supply their original code "
            "using --helpers-root; see the module documentation."
        ) from exc
    return read_mag_l3, grid_basin_mask


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).upper().replace("(", " (").replace(")", ") ").split())


def safe_directory_name(value: object) -> str:
    return re.sub(r"[^\w -]", "_", str(value), flags=re.UNICODE).strip()


def load_basins(csv_path: Path, shp_path: Path, basin_filter: str | None):
    basins_csv = pd.read_csv(csv_path)
    basins_csv.columns = basins_csv.columns.str.strip().str.lower()
    if "basin_name" not in basins_csv.columns and "name" in basins_csv.columns:
        basins_csv = basins_csv.rename(columns={"name": "basin_name"})
    required = {"basin_name", "lon", "lat"}
    missing = required - set(basins_csv.columns)
    if missing:
        raise ValueError(f"Missing basin CSV columns: {sorted(missing)}")

    basins_csv["basin_name"] = basins_csv["basin_name"].astype(str).str.strip().str.upper()
    for axis in ("lon", "lat"):
        basins_csv[axis] = pd.to_numeric(basins_csv[axis], errors="coerce")
    basins_csv = basins_csv.dropna(subset=["basin_name", "lon", "lat"])
    if basin_filter:
        basins_csv = basins_csv[
            basins_csv["basin_name"].str.contains(basin_filter.upper(), regex=False, na=False)
        ]
    if basins_csv.empty:
        raise ValueError("No basins remain after input validation and optional filtering")

    polygons = gpd.read_file(shp_path)
    for key in ("NAME", "RASTAREA"):
        if key not in polygons.columns:
            raise ValueError(f"Missing shapefile attribute: {key}")
    polygons["NAME"] = polygons["NAME"].astype(str).str.strip().str.upper()
    polygons["RASTAREA"] = pd.to_numeric(polygons["RASTAREA"], errors="coerce")
    if polygons.crs is None:
        warnings.warn("Basin shapefile lacks CRS; assuming EPSG:4326, as in original script")
        polygons = polygons.set_crs(epsg=4326)
    elif polygons.crs.to_epsg() != 4326:
        polygons = polygons.to_crs(epsg=4326)
    polygons["NAME_NORM"] = polygons["NAME"].map(normalize_name)
    return basins_csv, polygons


def find_basin(basin_name: str, lon: float, lat: float, polygons: gpd.GeoDataFrame):
    """Return (polygon geometry, area_km2), preferring the outlet location."""
    matches = polygons[polygons.contains(Point(lon, lat))]
    method = "outlet location"
    if matches.empty:
        key = normalize_name(basin_name)
        matches = polygons[polygons["NAME_NORM"] == key]
        method = "exact name"
        if matches.empty:
            matches = polygons[polygons["NAME_NORM"].map(
                lambda name: bool(name) and bool(key)
                and (name.startswith(key) or key.startswith(name))
            )]
            method = "name prefix"
        if matches.empty:
            matches = polygons[polygons["NAME_NORM"].str.contains(key, regex=False, na=False)]
            method = "name substring"
    if matches.empty:
        warnings.warn(f"No polygon match for basin: {basin_name}")
        return None
    if len(matches) > 1:
        warnings.warn(f"Ambiguous {method} match for {basin_name}; using first polygon")
    if method not in ("outlet location", "exact name"):
        warnings.warn(f"Matched {basin_name} by {method} to {matches.iloc[0]['NAME']}")
    result = matches.iloc[0]
    area_km2 = float(result["RASTAREA"])
    if not np.isfinite(area_km2) or area_km2 <= 0:
        warnings.warn(f"Missing/invalid area for basin {basin_name}")
        return None
    return result.geometry, area_km2


def list_discharge_files(directory: Path, start_year: int, end_year: int) -> list[Path]:
    """Find NetCDF files whose filenames contain a year in the study window."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Discharge directory does not exist: {directory}")
    matches = []
    for file in sorted(directory.rglob("*.nc")):
        year = re.search(r"(?:19|20)\d{2}", file.name)
        if year and start_year <= int(year.group(0)) <= end_year:
            matches.append(file)
    if not matches:
        raise FileNotFoundError(
            f"No discharge NetCDFs with years {start_year}-{end_year} "
            f"in filenames were found under {directory}"
        )
    return matches


def extract_discharge_timeseries(files: list[Path], lon_pt: float, lat_pt: float) -> pd.DataFrame:
    """Extract one daily discharge series from a high-flow outlet pixel."""
    times: list[pd.Timestamp] = []
    values: list[float] = []
    chosen_indices = None
    nodata_count = 0

    for file in tqdm(files, leave=False, desc="Discharge files"):
        try:
            with xr.open_dataset(file, decode_times=True) as ds:
                field = ds[DISCHARGE_VARIABLE]
                lon = np.asarray(ds["longitude"].values)
                lat = np.asarray(ds["latitude"].values)
                if lon.ndim != 1 or lat.ndim != 1:
                    raise ValueError("Expected one-dimensional longitude/latitude coordinates")

                if chosen_indices is None:
                    ix0 = int(np.abs(lon - lon_pt).argmin())
                    iy0 = int(np.abs(lat - lat_pt).argmin())
                    radius = OUTLET_SEARCH_RADIUS_CELLS
                    x_slice = slice(max(0, ix0 - radius), min(len(lon), ix0 + radius + 1))
                    y_slice = slice(max(0, iy0 - radius), min(len(lat), iy0 + radius + 1))
                    nearby = field.isel(longitude=x_slice, latitude=y_slice)
                    time_dims = [dim for dim in nearby.dims if dim not in ("latitude", "longitude")]
                    if not time_dims:
                        raise ValueError("No time dimension found for discharge variable")
                    mean_flow = nearby.mean(dim=time_dims, skipna=True).transpose(
                        "latitude", "longitude"
                    ).values
                    if not np.isfinite(mean_flow).any():
                        warnings.warn(f"No valid outlet pixels in {file.name}")
                        continue
                    local_y, local_x = np.unravel_index(np.nanargmax(mean_flow), mean_flow.shape)
                    chosen_indices = (y_slice.start + local_y, x_slice.start + local_x)

                iy, ix = chosen_indices
                raw = np.asarray(field.isel(latitude=iy, longitude=ix).values, dtype=float).ravel()
                dates = pd.to_datetime(np.atleast_1d(ds["valid_time"].values).ravel())
                # This convention is inherited from the original study workflow.
                dates = dates - pd.Timedelta(days=1)
                if len(raw) != len(dates):
                    raise ValueError(f"Discharge/time length mismatch in {file.name}")
                if not np.isfinite(raw).any():
                    nodata_count += 1
                    if nodata_count >= MAX_CONSECUTIVE_NODATA_FILES:
                        warnings.warn("Stopping after five consecutive files without discharge data")
                        break
                    continue
                nodata_count = 0
                times.extend(dates)
                values.extend(raw.tolist())
        except (OSError, ValueError, KeyError, IndexError) as exc:
            warnings.warn(f"Skipped {file.name}: {exc}")

    result = pd.DataFrame({"time": times, "discharge_m3_s": values})
    if result.empty:
        return result
    result = result.sort_values("time")
    duplicated = result["time"].duplicated()
    if duplicated.any():
        warnings.warn(f"Removed {duplicated.sum()} duplicate daily timestamps")
        result = result.drop_duplicates(subset="time", keep="first")
    return result


def daily_discharge_to_monthly_runoff(discharge: pd.DataFrame, area_km2: float) -> pd.DataFrame:
    """Aggregate daily mean flow (m^3/s) to monthly runoff depth (mm)."""
    if discharge.empty:
        return pd.DataFrame(columns=["runoff_mm", "days_available"]).rename_axis("time")
    series = discharge.set_index("time")["discharge_m3_s"].sort_index()
    daily_volume_m3 = series * 86400.0
    monthly = pd.DataFrame({
        "runoff_mm": daily_volume_m3.resample("MS").sum(min_count=1) / (area_km2 * 1e6) * 1000.0,
        "days_available": series.resample("MS").count(),
    })
    return monthly


def extract_twsa_pixels(grid: dict, mask: dict) -> pd.DataFrame:
    """Export valid basin grid cells and their full TWSA time series."""
    lat = np.asarray(grid["lat"])
    lon = np.asarray(grid["lon"])
    data = np.asarray(grid["data"])
    times = pd.to_datetime(grid["time"])
    if data.shape != (len(times), len(lat), len(lon)):
        raise ValueError("MAG-L3 data must have shape (time, latitude, longitude)")
    centers = mask["centers"]
    date_columns = [t.strftime("%Y-%m-%d") for t in times]
    rows = []
    lat_tol = max(1e-6, abs(float(lat[1] - lat[0])) * 1e-4) if len(lat) > 1 else 1e-6
    lon_tol = max(1e-6, abs(float(lon[1] - lon[0])) * 1e-4) if len(lon) > 1 else 1e-6
    for _, center in centers.iterrows():
        area = float(center["Area_km2"])
        if not np.isfinite(area) or area <= 0:
            continue
        center_lat = float(center["Lat"])
        center_lon = float(center["Lon"])
        iy = int(np.abs(lat - center_lat).argmin())
        ix = int(np.abs(lon - center_lon).argmin())
        if abs(lat[iy] - center_lat) > lat_tol or abs(lon[ix] - center_lon) > lon_tol:
            raise ValueError("Basin mask cell center does not match MAG-L3 grid")
        rows.append([center_lon, center_lat, area, *data[:, iy, ix].tolist()])
    return pd.DataFrame(rows, columns=["lon", "lat", "area_km2", *date_columns])


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    read_mag_l3, grid_basin_mask = load_project_helpers(args.helpers_root)
    csv_basins, polygon_basins = load_basins(args.basins_csv, args.basins_shp, args.basin)
    discharge_files = list_discharge_files(args.discharge_dir, args.start_year, args.end_year)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(csv_basins)} basins and {len(discharge_files)} discharge files")

    reference = read_mag_l3("Reference", args.filter, str(args.reference_nc))
    lat_grid = np.asarray(reference["lat"])
    lon_grid = np.asarray(reference["lon"])
    if len(lat_grid) < 2 or len(lon_grid) < 2:
        raise ValueError("MAG-L3 grid must contain at least two latitude and longitude cells")
    dlat = abs(float(lat_grid[1] - lat_grid[0]))
    dlon = abs(float(lon_grid[1] - lon_grid[0]))

    for _, basin in tqdm(csv_basins.iterrows(), total=len(csv_basins), desc="River basins"):
        name = basin["basin_name"]
        outlet_lon = float(basin["lon"])
        outlet_lat = float(basin["lat"])
        basin_match = find_basin(name, outlet_lon, outlet_lat, polygon_basins)
        if basin_match is None:
            continue
        polygon, area_km2 = basin_match
        target_dir = args.out_dir / safe_directory_name(name)
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{name}: basin area = {area_km2:,.1f} km^2")

        runoff_file = target_dir / "runoff.csv"
        if args.overwrite or not runoff_file.exists():
            daily = extract_discharge_timeseries(discharge_files, outlet_lon, outlet_lat)
            monthly = daily_discharge_to_monthly_runoff(daily, area_km2)
            monthly.to_csv(runoff_file)
            print(f"  Wrote {runoff_file} ({len(monthly)} months)")
        else:
            print(f"  Skipped existing {runoff_file}")

        basin_geometry = gpd.GeoSeries([polygon], crs="EPSG:4326")
        mask = grid_basin_mask(lat_grid, lon_grid, basin_geometry, dlat=dlat, dlon=dlon)

        for mission in MISSIONS:
            out_file = target_dir / mission / args.filter / "twsa_pixels_timeseries.csv"
            if out_file.exists() and not args.overwrite:
                print(f"  Skipped existing {mission} TWSA")
                continue
            grid = reference if mission == "Reference" else read_mag_l3(
                mission, args.filter, str(args.reference_nc)
            )
            extracted = extract_twsa_pixels(grid, mask)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            extracted.to_csv(out_file, index=False)
            print(f"  Wrote {mission} TWSA: {len(extracted)} pixels")

    print("\nExtraction completed. Note: TDWS estimation is not part of this script.")


if __name__ == "__main__":
    main()
