"""
generate_negative_samples.py
----------------------------

Creates pseudo-negative/control samples for the
NER Landslide Early Warning System.

Strategy:
- For every historical landslide event
- Generate a nearby random location
- Keep the same event date
- Ensure candidate is sufficiently far from
  known landslide locations
- Extract ERA5 weather
- Extract DEM terrain
- Extract ESA WorldCover
- Label sample as landslide_occurred = 0

Run:
    python src/generate_negative_samples.py
"""

import os
import glob
import math
import random

import numpy as np
import pandas as pd
import rasterio
import xarray as xr


# ==========================================
# RANDOM SEED
# ==========================================

random.seed(42)
np.random.seed(42)


# ==========================================
# FILE PATHS
# ==========================================

POSITIVE_FILE = (
    "data/processed/"
    "ner_landslide_master.csv"
)

ERA5_FOLDER = (
    "data/raw/"
    "era5_historical"
)

DEM_FILE = (
    "data/raw/dem/"
    "ner_dem_90m.tiff"
)

LANDCOVER_FOLDER = (
    "data/raw/"
    "landcover"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_negative_samples.csv"
)


# ==========================================
# SAMPLE SETTINGS
# ==========================================

MIN_DISTANCE_KM = 20

MAX_DISTANCE_KM = 50

MAX_ATTEMPTS = 100


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

if not os.path.exists(POSITIVE_FILE):
    print(f"❌ Missing: {POSITIVE_FILE}")
    raise SystemExit

if not os.path.exists(DEM_FILE):
    print(f"❌ Missing: {DEM_FILE}")
    raise SystemExit

if not os.path.exists(ERA5_FOLDER):
    print(f"❌ Missing: {ERA5_FOLDER}")
    raise SystemExit

if not os.path.exists(LANDCOVER_FOLDER):
    print(f"❌ Missing: {LANDCOVER_FOLDER}")
    raise SystemExit


# ==========================================
# LOAD POSITIVE DATA
# ==========================================

print("\nLoading positive landslide samples...")

positive_df = pd.read_csv(
    POSITIVE_FILE
)

positive_df["event_date"] = pd.to_datetime(
    positive_df["event_date"],
    errors="coerce"
)

positive_df = positive_df.dropna(
    subset=[
        "event_date",
        "latitude",
        "longitude"
    ]
).copy()


print(
    "Positive samples:",
    len(positive_df)
)


# ==========================================
# HAVERSINE DISTANCE
# ==========================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lat = (
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            delta_lat / 2
        ) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(
            delta_lon / 2
        ) ** 2
    )

    c = (
        2
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )

    return radius * c


# ==========================================
# GENERATE NEARBY POINT
# ==========================================

def generate_nearby_point(
    latitude,
    longitude
):

    distance_km = random.uniform(
        MIN_DISTANCE_KM,
        MAX_DISTANCE_KM
    )

    bearing = random.uniform(
        0,
        360
    )

    bearing_rad = math.radians(
        bearing
    )

    # Approx conversion
    latitude_change = (
        distance_km
        *
        math.cos(bearing_rad)
        /
        111.0
    )

    longitude_change = (
        distance_km
        *
        math.sin(bearing_rad)
        /
        (
            111.0
            *
            math.cos(
                math.radians(latitude)
            )
        )
    )

    new_latitude = (
        latitude
        +
        latitude_change
    )

    new_longitude = (
        longitude
        +
        longitude_change
    )

    return (
        new_latitude,
        new_longitude
    )


# ==========================================
# CHECK DISTANCE FROM KNOWN LANDSLIDES
# ==========================================

positive_coordinates = positive_df[
    [
        "latitude",
        "longitude"
    ]
].values


def far_from_known_landslides(
    latitude,
    longitude
):

    for (
        known_lat,
        known_lon
    ) in positive_coordinates:

        distance = haversine_distance(
            latitude,
            longitude,
            known_lat,
            known_lon
        )

        if distance < MIN_DISTANCE_KM:
            return False

    return True


# ==========================================
# LOAD DEM
# ==========================================

print("\nLoading DEM...")

dem_src = rasterio.open(
    DEM_FILE
)

dem = dem_src.read(
    1
).astype(
    "float64"
)

dem_transform = (
    dem_src.transform
)

dem_bounds = (
    dem_src.bounds
)

dem_crs = (
    dem_src.crs
)

dem_nodata = (
    dem_src.nodata
)


if dem_nodata is not None:

    dem[
        dem == dem_nodata
    ] = np.nan


# ==========================================
# DEM PIXEL SIZE
# ==========================================

center_latitude = (
    dem_bounds.top
    +
    dem_bounds.bottom
) / 2


if (
    dem_crs is not None
    and
    dem_crs.is_geographic
):

    latitude_radians = (
        np.radians(
            center_latitude
        )
    )

    meters_per_degree_lat = (
        111320
    )

    meters_per_degree_lon = (
        111320
        *
        np.cos(
            latitude_radians
        )
    )

    pixel_width_m = (
        abs(
            dem_transform.a
        )
        *
        meters_per_degree_lon
    )

    pixel_height_m = (
        abs(
            dem_transform.e
        )
        *
        meters_per_degree_lat
    )

else:

    pixel_width_m = abs(
        dem_transform.a
    )

    pixel_height_m = abs(
        dem_transform.e
    )


# ==========================================
# DEM SLOPE + ASPECT
# ==========================================

print(
    "Calculating slope and aspect..."
)

dz_dy, dz_dx = np.gradient(
    dem,
    pixel_height_m,
    pixel_width_m
)


slope = np.degrees(
    np.arctan(
        np.sqrt(
            dz_dx ** 2
            +
            dz_dy ** 2
        )
    )
)


aspect = np.degrees(
    np.arctan2(
        -dz_dx,
        dz_dy
    )
)

aspect = (
    aspect + 360
) % 360


# ==========================================
# TERRAIN FUNCTION
# ==========================================

def get_terrain(
    latitude,
    longitude
):

    try:

        row, col = (
            dem_src.index(
                longitude,
                latitude
            )
        )


        if (
            0 <= row < dem.shape[0]
            and
            0 <= col < dem.shape[1]
        ):

            elevation = (
                dem[row, col]
            )

            slope_value = (
                slope[row, col]
            )

            aspect_value = (
                aspect[row, col]
            )


            if np.isnan(
                elevation
            ):
                return None


            return {
                "elevation_m":
                    float(elevation),

                "slope_degree":
                    float(slope_value),

                "aspect_degree":
                    float(aspect_value)
            }


    except Exception:
        pass


    return None


# ==========================================
# LOAD WORLDCOVER
# ==========================================

print(
    "\nLoading WorldCover tiles..."
)

landcover_files = glob.glob(
    os.path.join(
        LANDCOVER_FOLDER,
        "*.tif"
    )
)


landcover_tiles = [
    rasterio.open(
        file_path
    )
    for file_path
    in landcover_files
]


print(
    "WorldCover tiles:",
    len(landcover_tiles)
)


# ==========================================
# LAND COVER FUNCTION
# ==========================================

def get_landcover(
    latitude,
    longitude
):

    for src in landcover_tiles:

        bounds = src.bounds


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


                code = int(value)


                if code == 0:
                    return None


                return {
                    "landcover_code":
                        code,

                    "landcover_class":
                        LANDCOVER_CLASSES.get(
                            code,
                            "Unknown"
                        )
                }


            except Exception:
                return None


    return None


# ==========================================
# ERA5 FILE MAP
# ==========================================

print(
    "\nLoading ERA5 file index..."
)


era5_files = glob.glob(
    os.path.join(
        ERA5_FOLDER,
        "era5_*.nc"
    )
)


era5_map = {}


for file_path in era5_files:

    file_name = os.path.basename(
        file_path
    )

    name = (
        file_name
        .replace(
            "era5_",
            ""
        )
        .replace(
            ".nc",
            ""
        )
    )

    parts = name.split("_")


    if len(parts) != 2:
        continue


    try:

        year = int(
            parts[0]
        )

        month = int(
            parts[1]
        )


        era5_map[
            (year, month)
        ] = file_path


    except ValueError:
        continue


print(
    "ERA5 monthly files:",
    len(era5_map)
)


# ==========================================
# ERA5 CACHE
# ==========================================

era5_cache = {}


def load_era5(
    year,
    month
):

    key = (
        year,
        month
    )


    if key in era5_cache:

        return era5_cache[key]


    file_path = era5_map.get(
        key
    )


    if file_path is None:

        return None


    try:

        dataset = xr.open_dataset(
            file_path
        )

        era5_cache[key] = (
            dataset
        )

        return dataset


    except Exception:

        return None


# ==========================================
# TIME COORDINATE
# ==========================================

def get_time_name(
    dataset
):

    for name in [
        "valid_time",
        "time"
    ]:

        if name in dataset.coords:
            return name


        if name in dataset.dims:
            return name


    return None


# ==========================================
# DAILY WEATHER
# ==========================================

def get_daily_weather(
    date,
    latitude,
    longitude
):

    dataset = load_era5(
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


        start = (
            pd.Timestamp(date)
            .normalize()
        )


        end = (
            start
            +
            pd.Timedelta(
                hours=23
            )
        )


        day_data = point.sel(
            {
                time_name:
                slice(
                    start,
                    end
                )
            }
        )


        rainfall_mm = None

        if "tp" in day_data:

            rainfall_mm = (
                day_data["tp"]
                .sum(
                    skipna=True
                )
                .item()
                *
                1000
            )


        temperature_c = None

        if "t2m" in day_data:

            temperature_c = (
                day_data["t2m"]
                .mean(
                    skipna=True
                )
                .item()
                -
                273.15
            )


        soil_1 = None

        if "swvl1" in day_data:

            soil_1 = (
                day_data["swvl1"]
                .mean(
                    skipna=True
                )
                .item()
            )


        soil_2 = None

        if "swvl2" in day_data:

            soil_2 = (
                day_data["swvl2"]
                .mean(
                    skipna=True
                )
                .item()
            )


        pressure = None

        if "sp" in day_data:

            pressure = (
                day_data["sp"]
                .mean(
                    skipna=True
                )
                .item()
                /
                100
            )


        return {
            "rainfall_mm":
                rainfall_mm,

            "temperature_c":
                temperature_c,

            "soil_water_layer_1":
                soil_1,

            "soil_water_layer_2":
                soil_2,

            "surface_pressure_hpa":
                pressure
        }


    except Exception:

        return None


# ==========================================
# WEATHER FEATURES
# ==========================================

def get_weather_features(
    event_date,
    latitude,
    longitude
):

    rainfall_values = []


    for offset in range(
        0,
        7
    ):

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
            weather is None
            or
            weather[
                "rainfall_mm"
            ] is None
        ):

            rainfall_values.append(
                None
            )

        else:

            rainfall_values.append(
                weather[
                    "rainfall_mm"
                ]
            )


    event_weather = (
        get_daily_weather(
            event_date,
            latitude,
            longitude
        )
    )


    if event_weather is None:

        return None


    if rainfall_values[0] is None:

        return None


    rainfall_24h = (
        rainfall_values[0]
    )


    rainfall_3d = sum(
        value
        for value
        in rainfall_values[:3]
        if value is not None
    )


    rainfall_7d = sum(
        value
        for value
        in rainfall_values
        if value is not None
    )


    return {

        "rainfall_24h_mm":
            rainfall_24h,

        "rainfall_3d_mm":
            rainfall_3d,

        "rainfall_7d_mm":
            rainfall_7d,

        "temperature_c":
            event_weather[
                "temperature_c"
            ],

        "soil_water_layer_1":
            event_weather[
                "soil_water_layer_1"
            ],

        "soil_water_layer_2":
            event_weather[
                "soil_water_layer_2"
            ],

        "surface_pressure_hpa":
            event_weather[
                "surface_pressure_hpa"
            ]
    }


# ==========================================
# GENERATE NEGATIVE SAMPLES
# ==========================================

print(
    "\nGenerating control samples..."
)


negative_samples = []


for number, (_, event) in enumerate(
    positive_df.iterrows(),
    start=1
):

    original_lat = (
        event["latitude"]
    )

    original_lon = (
        event["longitude"]
    )

    event_date = (
        event["event_date"]
    )


    sample_created = False


    for attempt in range(
        MAX_ATTEMPTS
    ):

        candidate_lat, candidate_lon = (
            generate_nearby_point(
                original_lat,
                original_lon
            )
        )


        # ------------------------------
        # Basic NER bounding area
        # ------------------------------

        if not (
            21
            <= candidate_lat
            <= 30
            and
            88
            <= candidate_lon
            <= 98
        ):

            continue


        # ------------------------------
        # Avoid known landslide points
        # ------------------------------

        if not far_from_known_landslides(
            candidate_lat,
            candidate_lon
        ):

            continue


        # ------------------------------
        # Terrain
        # ------------------------------

        terrain = get_terrain(
            candidate_lat,
            candidate_lon
        )


        if terrain is None:

            continue


        # ------------------------------
        # Land Cover
        # ------------------------------

        landcover = get_landcover(
            candidate_lat,
            candidate_lon
        )


        if landcover is None:

            continue


        # Avoid open water
        if (
            landcover[
                "landcover_code"
            ]
            ==
            80
        ):

            continue


        # ------------------------------
        # Weather
        # ------------------------------

        weather = get_weather_features(
            event_date,
            candidate_lat,
            candidate_lon
        )


        if weather is None:

            continue


        # ------------------------------
        # Create sample
        # ------------------------------

        sample = {

            "event_id":
                f"NEG_{event['event_id']}",

            "event_date":
                event_date,

            "event_title":
                "Control Sample",

            "location_description":
                "Pseudo-negative nearby control",

            "ner_state":
                event[
                    "ner_state"
                ],

            "landslide_category":
                "none",

            "landslide_trigger":
                "none",

            "landslide_size":
                "none",

            "landslide_setting":
                "control",

            "fatality_count":
                0,

            "injury_count":
                0,

            "longitude":
                candidate_lon,

            "latitude":
                candidate_lat,

            **weather,

            **terrain,

            **landcover,

            "landslide_occurred":
                0
        }


        negative_samples.append(
            sample
        )


        sample_created = True

        break


    if not sample_created:

        print(
            f"⚠️ Could not create control "
            f"for event {event['event_id']}"
        )


    if (
        number % 25 == 0
        or
        number == len(
            positive_df
        )
    ):

        print(
            f"Processed "
            f"{number}/"
            f"{len(positive_df)}"
        )


# ==========================================
# CREATE DATAFRAME
# ==========================================

negative_df = pd.DataFrame(
    negative_samples
)


# ==========================================
# SAVE
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


negative_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# CLOSE FILES
# ==========================================

dem_src.close()


for src in landcover_tiles:

    src.close()


for dataset in era5_cache.values():

    dataset.close()


# ==========================================
# RESULTS
# ==========================================

print(
    "\n======================================"
)

print(
    "NEGATIVE SAMPLE GENERATION COMPLETED"
)

print(
    "======================================"
)


print(
    "\nPositive Samples:",
    len(positive_df)
)


print(
    "Negative Samples Created:",
    len(negative_df)
)


if len(negative_df) > 0:

    print(
        "\nNegative Sample Preview:"
    )

    print(
        negative_df[
            [
                "event_date",
                "latitude",
                "longitude",
                "rainfall_24h_mm",
                "slope_degree",
                "landcover_class",
                "landslide_occurred"
            ]
        ].head()
    )


print(
    "\n✅ Saved at:"
)

print(
    OUTPUT_FILE
)