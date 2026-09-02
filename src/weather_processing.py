"""
weather_processing.py
---------------------

ERA5-Land weather data integration for
NER Landslide Early Warning System.

This script:
1. Loads cleaned NER landslide data
2. Loads ERA5 temperature/soil/pressure data
3. Loads ERA5 rainfall data
4. Filters landslide records for June 2016
5. Finds nearest weather grid point
6. Extracts weather features
7. Saves weather-enriched landslide dataset

Run with:
    python src/weather_processing.py
"""

import os

import pandas as pd
import xarray as xr


# ==========================================
# FILE PATHS
# ==========================================

LANDSLIDE_FILE = (
    "data/processed/"
    "ner_landslide_cleaned.csv"
)

WEATHER_FILE = (
    "data/raw/"
    "era5_land_2016_june.nc"
)

RAINFALL_FILE = (
    "data/raw/"
    "era5_rainfall_2016_june.nc"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_weather_2016_june.csv"
)


# ==========================================
# CHECK FILES
# ==========================================

files_to_check = [
    LANDSLIDE_FILE,
    WEATHER_FILE,
    RAINFALL_FILE
]

for file in files_to_check:

    if not os.path.exists(file):

        print(f"\n❌ File not found: {file}")
        print(
            "Please check the filename and location."
        )

        raise SystemExit


print("\n✅ All required files found!")


# ==========================================
# LOAD LANDSLIDE DATA
# ==========================================

print("\nLoading landslide dataset...")

landslides = pd.read_csv(
    LANDSLIDE_FILE
)

landslides["event_date"] = pd.to_datetime(
    landslides["event_date"],
    errors="coerce"
)


# ==========================================
# FILTER JUNE 2016 EVENTS
# ==========================================

june_2016 = landslides[
    (landslides["event_date"].dt.year == 2016)
    &
    (landslides["event_date"].dt.month == 6)
].copy()


print(
    "\nJune 2016 Landslide Records:",
    len(june_2016)
)


if june_2016.empty:

    print(
        "\n⚠️ No June 2016 landslide events found."
    )

    raise SystemExit


# ==========================================
# LOAD ERA5 DATA
# ==========================================

print("\nLoading ERA5 weather data...")

weather_ds = xr.open_dataset(
    WEATHER_FILE
)

print(
    "\nWeather Dataset Variables:"
)

print(
    list(weather_ds.data_vars)
)


print("\nLoading ERA5 rainfall data...")

rainfall_ds = xr.open_dataset(
    RAINFALL_FILE
)

print(
    "\nRainfall Dataset Variables:"
)

print(
    list(rainfall_ds.data_vars)
)


# ==========================================
# FIND TIME COORDINATE
# ==========================================

def get_time_name(dataset):

    possible_names = [
        "valid_time",
        "time"
    ]

    for name in possible_names:

        if name in dataset.coords:
            return name

        if name in dataset.dims:
            return name

    raise ValueError(
        "Time coordinate not found."
    )


weather_time = get_time_name(
    weather_ds
)

rainfall_time = get_time_name(
    rainfall_ds
)


print(
    "\nWeather time coordinate:",
    weather_time
)

print(
    "Rainfall time coordinate:",
    rainfall_time
)


# ==========================================
# PREPARE RESULT LIST
# ==========================================

results = []


# ==========================================
# PROCESS LANDSLIDE EVENTS
# ==========================================

print(
    "\nExtracting weather data "
    "for landslide events..."
)


for index, row in june_2016.iterrows():

    latitude = row["latitude"]
    longitude = row["longitude"]

    event_date = row["event_date"]


    # ======================================
    # SELECT NEAREST WEATHER GRID POINT
    # ======================================

    weather_point = weather_ds.sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest"
    )


    rainfall_point = rainfall_ds.sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest"
    )


    # ======================================
    # EVENT DAY
    # ======================================

    start_time = event_date.normalize()

    end_time = (
        start_time
        +
        pd.Timedelta(
            hours=23
        )
    )


    # ======================================
    # WEATHER FOR EVENT DAY
    # ======================================

    weather_day = weather_point.sel(
        {
            weather_time:
            slice(
                start_time,
                end_time
            )
        }
    )


    rainfall_day = rainfall_point.sel(
        {
            rainfall_time:
            slice(
                start_time,
                end_time
            )
        }
    )


    # ======================================
    # TEMPERATURE
    # ======================================

    temperature_c = None

    if "t2m" in weather_day:

        temperature_k = (
            weather_day["t2m"]
            .mean()
            .item()
        )

        temperature_c = (
            temperature_k
            -
            273.15
        )


    # ======================================
    # SOIL MOISTURE
    # ======================================

    soil_water_1 = None

    if "swvl1" in weather_day:

        soil_water_1 = (
            weather_day["swvl1"]
            .mean()
            .item()
        )


    soil_water_2 = None

    if "swvl2" in weather_day:

        soil_water_2 = (
            weather_day["swvl2"]
            .mean()
            .item()
        )


    # ======================================
    # SURFACE PRESSURE
    # ======================================

    surface_pressure = None

    if "sp" in weather_day:

        pressure_pa = (
            weather_day["sp"]
            .mean()
            .item()
        )

        # Pascal → hPa
        surface_pressure = (
            pressure_pa
            /
            100
        )


    # ======================================
    # RAINFALL
    # ======================================

    rainfall_mm = None

    if "tp" in rainfall_day:

        # ERA5 total precipitation
        # usually stored in metres.
        #
        # Sum hourly accumulated values
        # and convert metres → mm.

        rainfall_m = (
            rainfall_day["tp"]
            .sum()
            .item()
        )

        rainfall_mm = (
            rainfall_m
            *
            1000
        )


    # ======================================
    # SAVE RESULT
    # ======================================

    result = row.to_dict()

    result["temperature_c"] = (
        temperature_c
    )

    result["soil_water_layer_1"] = (
        soil_water_1
    )

    result["soil_water_layer_2"] = (
        soil_water_2
    )

    result["surface_pressure_hpa"] = (
        surface_pressure
    )

    result["rainfall_24h_mm"] = (
        rainfall_mm
    )


    results.append(
        result
    )


# ==========================================
# CREATE DATAFRAME
# ==========================================

weather_df = pd.DataFrame(
    results
)


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

weather_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# FINAL RESULTS
# ==========================================

print(
    "\n======================================"
)

print(
    "WEATHER INTEGRATION COMPLETED"
)

print(
    "======================================"
)


print(
    "\nOutput Shape:"
)

print(
    weather_df.shape
)


print(
    "\nWeather Columns:"
)

weather_columns = [
    "temperature_c",
    "soil_water_layer_1",
    "soil_water_layer_2",
    "surface_pressure_hpa",
    "rainfall_24h_mm"
]

print(
    weather_df[
        weather_columns
    ].head()
)


print(
    "\nMissing Weather Values:"
)

print(
    weather_df[
        weather_columns
    ].isnull().sum()
)


print(
    "\n✅ Dataset saved at:"
)

print(
    OUTPUT_FILE
)