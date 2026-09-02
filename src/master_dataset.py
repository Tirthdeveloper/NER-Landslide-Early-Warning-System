"""
master_dataset.py
-----------------

Creates the final landslide feature dataset
for the NER Landslide Early Warning System.

This script merges:

1. Historical ERA5 weather features
2. DEM terrain features
3. ESA WorldCover land-cover features

Run with:
    python src/master_dataset.py
"""

import os

import pandas as pd


# ==========================================
# FILE PATHS
# ==========================================

WEATHER_FILE = (
    "data/processed/"
    "ner_landslide_historical_weather.csv"
)

TERRAIN_FILE = (
    "data/processed/"
    "ner_landslide_terrain.csv"
)

LANDCOVER_FILE = (
    "data/processed/"
    "ner_landslide_landcover.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_master.csv"
)


# ==========================================
# CHECK FILES
# ==========================================

required_files = [
    WEATHER_FILE,
    TERRAIN_FILE,
    LANDCOVER_FILE
]


for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            f"❌ File not found: "
            f"{file_path}"
        )

        raise SystemExit


print("\n✅ All required files found!")


# ==========================================
# LOAD DATASETS
# ==========================================

print("\nLoading datasets...")


weather_df = pd.read_csv(
    WEATHER_FILE
)

terrain_df = pd.read_csv(
    TERRAIN_FILE
)

landcover_df = pd.read_csv(
    LANDCOVER_FILE
)


print(
    "\nWeather Shape:",
    weather_df.shape
)

print(
    "Terrain Shape:",
    terrain_df.shape
)

print(
    "Land Cover Shape:",
    landcover_df.shape
)


# ==========================================
# CHECK EVENT_ID
# ==========================================

datasets = {
    "Weather": weather_df,
    "Terrain": terrain_df,
    "Land Cover": landcover_df
}


for name, dataset in datasets.items():

    if "event_id" not in dataset.columns:

        print(
            f"\n❌ event_id missing "
            f"from {name} dataset"
        )

        raise SystemExit


# ==========================================
# CHECK DUPLICATE EVENT IDs
# ==========================================

print(
    "\nDuplicate Event IDs:"
)


for name, dataset in datasets.items():

    duplicates = (
        dataset[
            "event_id"
        ].duplicated().sum()
    )

    print(
        f"{name}: {duplicates}"
    )


# ==========================================
# WEATHER DATA
# ==========================================

weather_columns = [
    "event_id",
    "event_date",
    "event_title",
    "location_description",
    "ner_state",
    "landslide_category",
    "landslide_trigger",
    "landslide_size",
    "landslide_setting",
    "fatality_count",
    "injury_count",
    "longitude",
    "latitude",

    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "temperature_c",
    "soil_water_layer_1",
    "soil_water_layer_2",
    "surface_pressure_hpa"
]


weather_df = weather_df[
    [
        column
        for column in weather_columns
        if column in weather_df.columns
    ]
].copy()


# ==========================================
# TERRAIN FEATURES
# ==========================================

terrain_columns = [
    "event_id",
    "elevation_m",
    "slope_degree",
    "aspect_degree"
]


terrain_features = terrain_df[
    [
        column
        for column in terrain_columns
        if column in terrain_df.columns
    ]
].copy()


# ==========================================
# LAND COVER FEATURES
# ==========================================

landcover_columns = [
    "event_id",
    "landcover_code",
    "landcover_class"
]


landcover_features = landcover_df[
    [
        column
        for column in landcover_columns
        if column in landcover_df.columns
    ]
].copy()


# ==========================================
# MERGE WEATHER + TERRAIN
# ==========================================

print(
    "\nMerging weather + terrain..."
)


master_df = weather_df.merge(
    terrain_features,
    on="event_id",
    how="left"
)


print(
    "Shape after terrain merge:",
    master_df.shape
)


# ==========================================
# MERGE LAND COVER
# ==========================================

print(
    "\nMerging land-cover features..."
)


master_df = master_df.merge(
    landcover_features,
    on="event_id",
    how="left"
)


print(
    "Shape after land-cover merge:",
    master_df.shape
)


# ==========================================
# ADD TARGET COLUMN
# ==========================================

# These are historical landslide events,
# therefore target = 1.

master_df[
    "landslide_occurred"
] = 1


# ==========================================
# SORT DATA
# ==========================================

master_df[
    "event_date"
] = pd.to_datetime(
    master_df["event_date"],
    errors="coerce"
)


master_df = master_df.sort_values(
    by="event_date"
)


# ==========================================
# CHECK IMPORTANT FEATURES
# ==========================================

feature_columns = [
    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "temperature_c",
    "soil_water_layer_1",
    "soil_water_layer_2",
    "surface_pressure_hpa",
    "elevation_m",
    "slope_degree",
    "aspect_degree",
    "landcover_code"
]


print(
    "\n======================================"
)

print(
    "MISSING VALUES BEFORE FINAL SAVE"
)

print(
    "======================================"
)


print(
    master_df[
        feature_columns
    ].isnull().sum()
)


# ==========================================
# REMOVE FULLY INVALID ROWS
# ==========================================

# Remove records where core location
# or date information is missing.

master_df = master_df.dropna(
    subset=[
        "event_date",
        "latitude",
        "longitude"
    ]
)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# SAVE MASTER DATASET
# ==========================================

master_df.to_csv(
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
    "MASTER DATASET CREATED SUCCESSFULLY"
)

print(
    "======================================"
)


print(
    "\nFinal Dataset Shape:"
)

print(
    master_df.shape
)


print(
    "\nFinal Columns:"
)

print(
    master_df.columns.tolist()
)


print(
    "\nFeature Preview:"
)

preview_columns = [
    "ner_state",
    "event_date",
    "latitude",
    "longitude",
    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "temperature_c",
    "elevation_m",
    "slope_degree",
    "landcover_class",
    "landslide_occurred"
]


print(
    master_df[
        preview_columns
    ].head(10)
)


print(
    "\nFinal Missing Values:"
)

print(
    master_df[
        feature_columns
    ].isnull().sum()
)


print(
    "\n✅ Master dataset saved at:"
)

print(
    OUTPUT_FILE
)