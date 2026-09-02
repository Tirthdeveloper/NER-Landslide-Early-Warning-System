"""
historical_weather_processing.py
--------------------------------

Historical ERA5-Land weather feature extraction
for NER Landslide Early Warning System.

Features:
- Rainfall 24 hours
- Rainfall previous 3 days
- Rainfall previous 7 days
- Average temperature
- Soil moisture layer 1
- Soil moisture layer 2
- Surface pressure

Run with:
    python src/historical_weather_processing.py
"""

import os
import glob

import pandas as pd
import xarray as xr


# ==========================================
# FILE PATHS
# ==========================================

LANDSLIDE_FILE = (
    "data/processed/"
    "ner_landslide_cleaned.csv"
)

ERA5_FOLDER = (
    "data/raw/"
    "era5_historical"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_historical_weather.csv"
)


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(LANDSLIDE_FILE):
    print(f"❌ Landslide file not found: {LANDSLIDE_FILE}")
    raise SystemExit


if not os.path.exists(ERA5_FOLDER):
    print(f"❌ ERA5 folder not found: {ERA5_FOLDER}")
    raise SystemExit


era5_files = sorted(
    glob.glob(
        os.path.join(
            ERA5_FOLDER,
            "era5_*.nc"
        )
    )
)


print("\n======================================")
print("HISTORICAL WEATHER PROCESSING")
print("======================================")


print(
    f"\nERA5 files found: {len(era5_files)}"
)


if len(era5_files) == 0:
    print("❌ No ERA5 historical files found.")
    raise SystemExit


# ==========================================
# LOAD LANDSLIDE DATA
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
    subset=[
        "event_date",
        "latitude",
        "longitude"
    ]
).copy()


print(
    "Landslide records:",
    len(df)
)


# ==========================================
# CREATE ERA5 FILE MAP
# ==========================================

era5_map = {}


for file_path in era5_files:

    file_name = os.path.basename(
        file_path
    )

    # Example:
    # era5_2013_07.nc

    name = (
        file_name
        .replace("era5_", "")
        .replace(".nc", "")
    )

    parts = name.split("_")


    if len(parts) != 2:
        continue


    year = int(parts[0])

    month = int(parts[1])


    era5_map[
        (year, month)
    ] = file_path


print(
    "\nERA5 month files mapped:",
    len(era5_map)
)


# ==========================================
# DATASET CACHE
# ==========================================

dataset_cache = {}


def load_era5_dataset(
    year,
    month
):

    """
    Load one ERA5 month only once.
    """

    key = (
        year,
        month
    )


    if key in dataset_cache:

        return dataset_cache[key]


    file_path = era5_map.get(
        key
    )


    if file_path is None:

        return None


    try:

        dataset = xr.open_dataset(
            file_path
        )

        dataset_cache[key] = (
            dataset
        )

        return dataset


    except Exception as error:

        print(
            f"❌ Error opening "
            f"{file_path}"
        )

        print(error)

        return None


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


    return None


# ==========================================
# EXTRACT VARIABLE VALUE
# ==========================================

def get_daily_weather(
    date,
    latitude,
    longitude
):

    """
    Extract weather values
    for one date/location.
    """

    dataset = load_era5_dataset(
        date.year,
        date.month
    )


    if dataset is None:

        return None


    time_name = get_time_name(
        dataset
    )


    if time_name is None:

        return None


    try:

        point = dataset.sel(
            latitude=latitude,
            longitude=longitude,
            method="nearest"
        )


        start_time = pd.Timestamp(
            date
        ).normalize()


        end_time = (
            start_time
            +
            pd.Timedelta(
                hours=23
            )
        )


        day_data = point.sel(
            {
                time_name:
                slice(
                    start_time,
                    end_time
                )
            }
        )


        # ==============================
        # RAINFALL
        # ==============================

        rainfall_mm = None


        if "tp" in day_data:

            rainfall_m = (
                day_data["tp"]
                .sum(
                    skipna=True
                )
                .item()
            )


            rainfall_mm = (
                rainfall_m
                *
                1000
            )


        # ==============================
        # TEMPERATURE
        # ==============================

        temperature_c = None


        if "t2m" in day_data:

            temperature_k = (
                day_data["t2m"]
                .mean(
                    skipna=True
                )
                .item()
            )


            temperature_c = (
                temperature_k
                -
                273.15
            )


        # ==============================
        # SOIL WATER LAYER 1
        # ==============================

        soil_water_1 = None


        if "swvl1" in day_data:

            soil_water_1 = (
                day_data["swvl1"]
                .mean(
                    skipna=True
                )
                .item()
            )


        # ==============================
        # SOIL WATER LAYER 2
        # ==============================

        soil_water_2 = None


        if "swvl2" in day_data:

            soil_water_2 = (
                day_data["swvl2"]
                .mean(
                    skipna=True
                )
                .item()
            )


        # ==============================
        # SURFACE PRESSURE
        # ==============================

        pressure_hpa = None


        if "sp" in day_data:

            pressure_pa = (
                day_data["sp"]
                .mean(
                    skipna=True
                )
                .item()
            )


            pressure_hpa = (
                pressure_pa
                /
                100
            )


        return {
            "rainfall_mm":
                rainfall_mm,

            "temperature_c":
                temperature_c,

            "soil_water_1":
                soil_water_1,

            "soil_water_2":
                soil_water_2,

            "pressure_hpa":
                pressure_hpa
        }


    except Exception:

        return None


# ==========================================
# PROCESS LANDSLIDE EVENTS
# ==========================================

results = []


total_records = len(df)


print(
    "\nExtracting historical weather..."
)


for number, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    event_date = (
        row["event_date"]
    )

    latitude = (
        row["latitude"]
    )

    longitude = (
        row["longitude"]
    )


    # ======================================
    # EVENT DAY WEATHER
    # ======================================

    event_weather = (
        get_daily_weather(
            event_date,
            latitude,
            longitude
        )
    )


    # ======================================
    # RAINFALL WINDOWS
    # ======================================

    rainfall_values = []


    for offset in range(0, 7):

        date = (
            event_date
            -
            pd.Timedelta(
                days=offset
            )
        )


        weather = get_daily_weather(
            date,
            latitude,
            longitude
        )


        if (
            weather is not None
            and
            weather["rainfall_mm"]
            is not None
        ):

            rainfall_values.append(
                weather["rainfall_mm"]
            )

        else:

            rainfall_values.append(
                None
            )


    # ======================================
    # RAINFALL 24 HOURS
    # ======================================

    rainfall_24h = (
        rainfall_values[0]
    )


    # ======================================
    # RAINFALL 3 DAYS
    # ======================================

    rainfall_3d_values = [
        value
        for value
        in rainfall_values[:3]
        if value is not None
    ]


    if rainfall_3d_values:

        rainfall_3d = sum(
            rainfall_3d_values
        )

    else:

        rainfall_3d = None


    # ======================================
    # RAINFALL 7 DAYS
    # ======================================

    rainfall_7d_values = [
        value
        for value
        in rainfall_values
        if value is not None
    ]


    if rainfall_7d_values:

        rainfall_7d = sum(
            rainfall_7d_values
        )

    else:

        rainfall_7d = None


    # ======================================
    # EVENT WEATHER FEATURES
    # ======================================

    if event_weather is not None:

        temperature_c = (
            event_weather[
                "temperature_c"
            ]
        )

        soil_water_1 = (
            event_weather[
                "soil_water_1"
            ]
        )

        soil_water_2 = (
            event_weather[
                "soil_water_2"
            ]
        )

        pressure_hpa = (
            event_weather[
                "pressure_hpa"
            ]
        )


    else:

        temperature_c = None

        soil_water_1 = None

        soil_water_2 = None

        pressure_hpa = None


    # ======================================
    # SAVE RESULT
    # ======================================

    result = row.to_dict()


    result[
        "rainfall_24h_mm"
    ] = rainfall_24h


    result[
        "rainfall_3d_mm"
    ] = rainfall_3d


    result[
        "rainfall_7d_mm"
    ] = rainfall_7d


    result[
        "temperature_c"
    ] = temperature_c


    result[
        "soil_water_layer_1"
    ] = soil_water_1


    result[
        "soil_water_layer_2"
    ] = soil_water_2


    result[
        "surface_pressure_hpa"
    ] = pressure_hpa


    results.append(
        result
    )


    # ======================================
    # PROGRESS
    # ======================================

    if (
        number % 25 == 0
        or
        number == total_records
    ):

        print(
            f"Processed "
            f"{number}/{total_records}"
        )


# ==========================================
# CREATE FINAL DATAFRAME
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
# SAVE OUTPUT
# ==========================================

weather_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# WEATHER COLUMNS
# ==========================================

weather_columns = [
    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "temperature_c",
    "soil_water_layer_1",
    "soil_water_layer_2",
    "surface_pressure_hpa"
]


# ==========================================
# RESULTS
# ==========================================

print(
    "\n======================================"
)

print(
    "HISTORICAL WEATHER PROCESSING COMPLETED"
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
    "\nWeather Preview:"
)

print(
    weather_df[
        weather_columns
    ].head(10)
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
    "\nWeather Statistics:"
)

print(
    weather_df[
        weather_columns
    ].describe()
)


print(
    "\n✅ Dataset saved at:"
)

print(
    OUTPUT_FILE
)


# ==========================================
# CLOSE ERA5 DATASETS
# ==========================================

for dataset in dataset_cache.values():

    dataset.close()