"""
dem_processing.py
-----------------

Copernicus DEM processing for the
NER Landslide Early Warning System.

Features extracted:
- Elevation
- Slope
- Aspect

Run with:
    python src/dem_processing.py
"""

import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol


# ==========================================
# FILE PATHS
# ==========================================

DEM_FILE = "data/raw/dem/ner_dem_90m.tiff"

LANDSLIDE_FILE = (
    "data/processed/"
    "ner_landslide_cleaned.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_terrain.csv"
)


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(DEM_FILE):
    print(f"❌ DEM file not found: {DEM_FILE}")
    raise SystemExit


if not os.path.exists(LANDSLIDE_FILE):
    print(f"❌ Landslide file not found: {LANDSLIDE_FILE}")
    raise SystemExit


print("\n✅ Required files found!")


# ==========================================
# LOAD LANDSLIDE DATA
# ==========================================

print("\nLoading landslide dataset...")

df = pd.read_csv(LANDSLIDE_FILE)

print("Landslide records:", len(df))


# ==========================================
# LOAD DEM
# ==========================================

print("\nLoading Copernicus DEM...")

with rasterio.open(DEM_FILE) as src:

    dem = src.read(1).astype("float64")

    transform = src.transform

    dem_crs = src.crs

    nodata = src.nodata

    bounds = src.bounds


print("\nDEM Information:")
print("CRS:", dem_crs)
print("Shape:", dem.shape)
print("Bounds:", bounds)
print("NoData:", nodata)


# ==========================================
# HANDLE NODATA
# ==========================================

if nodata is not None:
    dem[dem == nodata] = np.nan


# ==========================================
# CALCULATE PIXEL SIZE IN METERS
# ==========================================

# Approximate centre latitude of DEM
center_latitude = (
    bounds.top + bounds.bottom
) / 2


# If DEM uses geographic coordinates
if dem_crs is not None and dem_crs.is_geographic:

    latitude_radians = np.radians(
        center_latitude
    )

    # Approx metres per degree latitude
    meters_per_degree_lat = 111320

    # Approx metres per degree longitude
    meters_per_degree_lon = (
        111320
        *
        np.cos(latitude_radians)
    )

    pixel_width_m = (
        abs(transform.a)
        *
        meters_per_degree_lon
    )

    pixel_height_m = (
        abs(transform.e)
        *
        meters_per_degree_lat
    )

else:

    # Projected CRS usually already uses metres
    pixel_width_m = abs(transform.a)

    pixel_height_m = abs(transform.e)


print("\nApprox Pixel Size:")

print(
    "Width:",
    round(pixel_width_m, 2),
    "meters"
)

print(
    "Height:",
    round(pixel_height_m, 2),
    "meters"
)


# ==========================================
# CALCULATE TERRAIN GRADIENT
# ==========================================

print("\nCalculating terrain gradients...")

dz_dy, dz_dx = np.gradient(
    dem,
    pixel_height_m,
    pixel_width_m
)


# ==========================================
# CALCULATE SLOPE
# ==========================================

slope = np.degrees(
    np.arctan(
        np.sqrt(
            dz_dx ** 2
            +
            dz_dy ** 2
        )
    )
)


# ==========================================
# CALCULATE ASPECT
# ==========================================

aspect = np.degrees(
    np.arctan2(
        -dz_dx,
        dz_dy
    )
)

aspect = (
    aspect + 360
) % 360


print("✅ Slope calculated")
print("✅ Aspect calculated")


# ==========================================
# EXTRACT TERRAIN FEATURES
# ==========================================

print(
    "\nExtracting terrain features "
    "for landslide locations..."
)


elevation_values = []

slope_values = []

aspect_values = []


for _, record in df.iterrows():

    latitude = record["latitude"]

    longitude = record["longitude"]


    try:

        # Convert coordinates to raster row/column
        row, col = rowcol(
            transform,
            longitude,
            latitude
        )


        # Check raster boundaries
        if (
            0 <= row < dem.shape[0]
            and
            0 <= col < dem.shape[1]
        ):

            elevation = dem[row, col]

            slope_value = slope[row, col]

            aspect_value = aspect[row, col]


            if np.isnan(elevation):
                elevation = np.nan

            if np.isnan(slope_value):
                slope_value = np.nan

            if np.isnan(aspect_value):
                aspect_value = np.nan


        else:

            elevation = np.nan

            slope_value = np.nan

            aspect_value = np.nan


    except Exception:

        elevation = np.nan

        slope_value = np.nan

        aspect_value = np.nan


    elevation_values.append(
        elevation
    )

    slope_values.append(
        slope_value
    )

    aspect_values.append(
        aspect_value
    )


# ==========================================
# ADD TERRAIN FEATURES
# ==========================================

df["elevation_m"] = elevation_values

df["slope_degree"] = slope_values

df["aspect_degree"] = aspect_values


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# RESULTS
# ==========================================

print("\n======================================")
print("DEM PROCESSING COMPLETED")
print("======================================")


print("\nOutput Shape:")
print(df.shape)


print("\nTerrain Features Preview:")

print(
    df[
        [
            "ner_state",
            "latitude",
            "longitude",
            "elevation_m",
            "slope_degree",
            "aspect_degree"
        ]
    ].head(10)
)


print("\nMissing Terrain Values:")

print(
    df[
        [
            "elevation_m",
            "slope_degree",
            "aspect_degree"
        ]
    ].isnull().sum()
)


print("\nTerrain Statistics:")

print(
    df[
        [
            "elevation_m",
            "slope_degree",
            "aspect_degree"
        ]
    ].describe()
)


print("\n✅ Dataset saved at:")

print(OUTPUT_FILE)