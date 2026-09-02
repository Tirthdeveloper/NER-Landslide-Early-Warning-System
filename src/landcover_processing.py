"""
landcover_processing.py
-----------------------

ESA WorldCover land-cover feature extraction
for NER Landslide Early Warning System.

This script:
1. Loads cleaned landslide records
2. Finds the correct WorldCover tile
3. Extracts land-cover code
4. Converts code into readable class name
5. Saves the processed dataset

Run with:
    python src/landcover_processing.py
"""

import os
import glob

import pandas as pd
import rasterio


# ==========================================
# FILE PATHS
# ==========================================

LANDSLIDE_FILE = (
    "data/processed/"
    "ner_landslide_cleaned.csv"
)

LANDCOVER_FOLDER = (
    "data/raw/landcover"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_landcover.csv"
)


# ==========================================
# WORLDCOVER CLASSES
# ==========================================

LANDCOVER_CLASSES = {

    10: "Tree cover",

    20: "Shrubland",

    30: "Grassland",

    40: "Cropland",

    50: "Built-up",

    60: "Bare / sparse vegetation",

    70: "Snow and ice",

    80: "Permanent water bodies",

    90: "Herbaceous wetland",

    95: "Mangroves",

    100: "Moss and lichen"
}


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(LANDSLIDE_FILE):

    print(
        f"❌ Landslide file not found: "
        f"{LANDSLIDE_FILE}"
    )

    raise SystemExit


if not os.path.exists(LANDCOVER_FOLDER):

    print(
        f"❌ Landcover folder not found: "
        f"{LANDCOVER_FOLDER}"
    )

    raise SystemExit


# ==========================================
# FIND ALL WORLDCOVER TILES
# ==========================================

tif_files = glob.glob(
    os.path.join(
        LANDCOVER_FOLDER,
        "*.tif"
    )
)


print("\n======================================")
print("LAND COVER PROCESSING")
print("======================================")


print(
    f"\nWorldCover tiles found: "
    f"{len(tif_files)}"
)


if len(tif_files) == 0:

    print(
        "\n❌ No .tif files found."
    )

    raise SystemExit


# ==========================================
# LOAD LANDSLIDE DATA
# ==========================================

print(
    "\nLoading cleaned landslide dataset..."
)

df = pd.read_csv(
    LANDSLIDE_FILE
)


print(
    "Landslide records:",
    len(df)
)


# ==========================================
# OPEN WORLDCOVER TILES
# ==========================================

print(
    "\nOpening WorldCover tiles..."
)


raster_tiles = []


for tif_file in tif_files:

    src = rasterio.open(
        tif_file
    )

    raster_tiles.append(
        src
    )


print(
    f"✅ {len(raster_tiles)} tiles opened"
)


# ==========================================
# EXTRACT LAND COVER
# ==========================================

def get_landcover(
    longitude,
    latitude
):

    """
    Find the WorldCover tile containing
    the given coordinate and return
    its land-cover code.
    """

    for src in raster_tiles:

        bounds = src.bounds


        # Check whether coordinate
        # lies inside this tile
        if (
            bounds.left
            <= longitude
            <= bounds.right
            and
            bounds.bottom
            <= latitude
            <= bounds.top
        ):

            try:

                # Sample raster value
                value = next(
                    src.sample(
                        [
                            (
                                longitude,
                                latitude
                            )
                        ]
                    )
                )[0]


                # Convert numpy value
                # into normal integer
                value = int(value)


                return value


            except Exception:

                return None


    # No tile found
    return None


# ==========================================
# PROCESS LANDSLIDE LOCATIONS
# ==========================================

print(
    "\nExtracting land-cover features..."
)


landcover_codes = []

landcover_names = []


total_records = len(df)


for number, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    latitude = row["latitude"]

    longitude = row["longitude"]


    code = get_landcover(
        longitude,
        latitude
    )


    if code is not None:

        name = LANDCOVER_CLASSES.get(
            code,
            "Unknown"
        )

    else:

        name = "No Data"


    landcover_codes.append(
        code
    )

    landcover_names.append(
        name
    )


    # Show progress every 50 records
    if (
        number % 50 == 0
        or
        number == total_records
    ):

        print(
            f"Processed "
            f"{number}/{total_records}"
        )


# ==========================================
# ADD NEW COLUMNS
# ==========================================

df["landcover_code"] = (
    landcover_codes
)

df["landcover_class"] = (
    landcover_names
)


# ==========================================
# CLOSE RASTER FILES
# ==========================================

for src in raster_tiles:

    src.close()


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# SAVE OUTPUT
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# RESULTS
# ==========================================

print(
    "\n======================================"
)

print(
    "LAND COVER PROCESSING COMPLETED"
)

print(
    "======================================"
)


print(
    "\nOutput Shape:"
)

print(
    df.shape
)


print(
    "\nLand Cover Preview:"
)

print(
    df[
        [
            "ner_state",
            "latitude",
            "longitude",
            "landcover_code",
            "landcover_class"
        ]
    ].head(10)
)


print(
    "\nLand Cover Distribution:"
)

print(
    df[
        "landcover_class"
    ].value_counts(
        dropna=False
    )
)


print(
    "\nMissing Land Cover Codes:"
)

print(
    df[
        "landcover_code"
    ].isnull().sum()
)


print(
    "\nUnknown Land Cover:"
)

print(
    (
        df["landcover_class"]
        ==
        "Unknown"
    ).sum()
)


print(
    "\nNo Data Locations:"
)

print(
    (
        df["landcover_class"]
        ==
        "No Data"
    ).sum()
)


print(
    "\n✅ Dataset saved at:"
)

print(
    OUTPUT_FILE
)