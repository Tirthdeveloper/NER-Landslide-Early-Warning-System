"""
download_era5_historical.py
---------------------------

Downloads ERA5-Land historical weather data
for landslide events in North-East India.

Strategy:
- Read cleaned NER landslide dataset
- For each unique event date
- Download previous 7 days + event day
- Save monthly ERA5 files only once
- Skip files already downloaded

Run with:
    python src/download_era5_historical.py
"""

import os
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
import cdsapi


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

CDS_API_URL = os.getenv("CDS_API_URL")
CDS_API_KEY = os.getenv("CDS_API_KEY")


if not CDS_API_URL:
    print("❌ CDS_API_URL missing in .env")
    raise SystemExit

if not CDS_API_KEY:
    print("❌ CDS_API_KEY missing in .env")
    raise SystemExit


# ==========================================
# CREATE CDS CLIENT
# ==========================================

client = cdsapi.Client(
    url=CDS_API_URL,
    key=CDS_API_KEY
)


# ==========================================
# FILE PATHS
# ==========================================

LANDSLIDE_FILE = (
    "data/processed/"
    "ner_landslide_cleaned.csv"
)

OUTPUT_FOLDER = (
    "data/raw/"
    "era5_historical"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================
# LOAD LANDSLIDE DATASET
# ==========================================

print("\nLoading landslide dataset...")

df = pd.read_csv(
    LANDSLIDE_FILE
)

df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["event_date"]
)


print(
    "Total landslide records:",
    len(df)
)


# ==========================================
# GET REQUIRED DATES
# ==========================================

required_dates = set()


for event_date in df["event_date"]:

    # Event day + previous 7 days
    for day_offset in range(0, 8):

        date = (
            event_date
            -
            timedelta(
                days=day_offset
            )
        )

        required_dates.add(
            date.date()
        )


print(
    "\nTotal unique weather dates required:",
    len(required_dates)
)


# ==========================================
# GROUP BY YEAR-MONTH
# ==========================================

month_groups = {}


for date in required_dates:

    key = (
        date.year,
        date.month
    )

    if key not in month_groups:
        month_groups[key] = set()

    month_groups[key].add(
        date.day
    )


print(
    "\nUnique year-month groups:",
    len(month_groups)
)


# ==========================================
# ERA5 VARIABLES
# ==========================================

variables = [
    "total_precipitation",
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "surface_pressure"
]


# ==========================================
# ALL HOURS
# ==========================================

hours = [
    f"{hour:02d}:00"
    for hour in range(24)
]


# ==========================================
# DOWNLOAD ERA5 FILES
# ==========================================

total_files = len(month_groups)


for number, ((year, month), days) in enumerate(
    sorted(month_groups.items()),
    start=1
):

    filename = (
        f"era5_"
        f"{year}_"
        f"{month:02d}.nc"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    print(
        "\n======================================"
    )

    print(
        f"Processing {number}/{total_files}"
    )

    print(
        f"Year-Month: {year}-{month:02d}"
    )

    print(
        f"Days required: "
        f"{sorted(days)}"
    )


    # ======================================
    # SKIP EXISTING FILE
    # ======================================

    if os.path.exists(output_path):

        print(
            f"✅ Already exists: "
            f"{filename}"
        )

        continue


    # ======================================
    # FORMAT DAYS
    # ======================================

    day_list = [
        f"{day:02d}"
        for day in sorted(days)
    ]


    # ======================================
    # REQUEST
    # ======================================

    request = {

        "variable": variables,

        "year": str(year),

        "month": (
            f"{month:02d}"
        ),

        "day": day_list,

        "time": hours,

        "data_format": "netcdf",

        "download_format": "unarchived",

        # North, West, South, East
        "area": [
            30,
            88,
            21,
            98
        ]
    }


    # ======================================
    # DOWNLOAD
    # ======================================

    try:

        print(
            f"Downloading: "
            f"{filename}"
        )

        client.retrieve(
            "reanalysis-era5-land",
            request,
            output_path
        )

        print(
            f"✅ Downloaded: "
            f"{filename}"
        )


    except Exception as error:

        print(
            f"❌ Failed: "
            f"{filename}"
        )

        print(error)


# ==========================================
# COMPLETED
# ==========================================

print(
    "\n======================================"
)

print(
    "HISTORICAL ERA5 DOWNLOAD COMPLETED"
)

print(
    "======================================"
)

print(
    f"\nFiles saved inside:"
)

print(
    OUTPUT_FOLDER
)