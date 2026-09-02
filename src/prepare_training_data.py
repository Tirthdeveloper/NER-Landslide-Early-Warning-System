"""
prepare_training_data.py
------------------------

Combines positive landslide samples and
pseudo-negative control samples into the
final ML training dataset.

Run with:
    python src/prepare_training_data.py
"""

import os

import pandas as pd


# ==========================================
# FILE PATHS
# ==========================================

POSITIVE_FILE = (
    "data/processed/"
    "ner_landslide_master.csv"
)

NEGATIVE_FILE = (
    "data/processed/"
    "ner_landslide_negative_samples.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "ner_landslide_training.csv"
)


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(POSITIVE_FILE):
    print(f"❌ Positive dataset missing: {POSITIVE_FILE}")
    raise SystemExit


if not os.path.exists(NEGATIVE_FILE):
    print(f"❌ Negative dataset missing: {NEGATIVE_FILE}")
    raise SystemExit


print("\n✅ Required datasets found!")


# ==========================================
# LOAD DATASETS
# ==========================================

positive_df = pd.read_csv(
    POSITIVE_FILE
)

negative_df = pd.read_csv(
    NEGATIVE_FILE
)


print(
    "\nPositive Samples:",
    len(positive_df)
)

print(
    "Negative Samples:",
    len(negative_df)
)


# ==========================================
# TARGET COLUMN
# ==========================================

positive_df[
    "landslide_occurred"
] = 1


negative_df[
    "landslide_occurred"
] = 0


# ==========================================
# IMPORTANT ML FEATURES
# ==========================================

features = [
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


required_columns = [
    "event_id",
    "event_date",
    "ner_state",
    "latitude",
    "longitude"
] + features + [
    "landslide_occurred"
]


# ==========================================
# CHECK COLUMNS
# ==========================================

for column in required_columns:

    if column not in positive_df.columns:

        print(
            f"❌ Positive dataset missing column: "
            f"{column}"
        )

        raise SystemExit


    if column not in negative_df.columns:

        print(
            f"❌ Negative dataset missing column: "
            f"{column}"
        )

        raise SystemExit


# ==========================================
# KEEP REQUIRED COLUMNS
# ==========================================

positive_df = positive_df[
    required_columns
].copy()


negative_df = negative_df[
    required_columns
].copy()


# ==========================================
# COMBINE DATASETS
# ==========================================

training_df = pd.concat(
    [
        positive_df,
        negative_df
    ],
    ignore_index=True
)


print(
    "\nCombined Dataset Shape:",
    training_df.shape
)


# ==========================================
# REMOVE DUPLICATES
# ==========================================

before_duplicates = len(
    training_df
)


training_df = training_df.drop_duplicates()


after_duplicates = len(
    training_df
)


print(
    "\nDuplicates Removed:",
    before_duplicates
    -
    after_duplicates
)


# ==========================================
# MISSING VALUES
# ==========================================

print(
    "\nMissing Values Before Cleaning:"
)

print(
    training_df[
        features
    ].isnull().sum()
)


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

for column in features:

    if training_df[column].isnull().any():

        median_value = (
            training_df[column]
            .median()
        )

        training_df[column] = (
            training_df[column]
            .fillna(
                median_value
            )
        )


# ==========================================
# REMOVE INVALID VALUES
# ==========================================

training_df = training_df[
    training_df["rainfall_24h_mm"] >= 0
]

training_df = training_df[
    training_df["rainfall_3d_mm"] >= 0
]

training_df = training_df[
    training_df["rainfall_7d_mm"] >= 0
]


training_df = training_df[
    training_df["slope_degree"]
    .between(
        0,
        90
    )
]


# ==========================================
# SHUFFLE DATASET
# ==========================================

training_df = training_df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print(
    "\nClass Distribution:"
)

print(
    training_df[
        "landslide_occurred"
    ].value_counts()
)


print(
    "\nClass Percentage:"
)

print(
    training_df[
        "landslide_occurred"
    ].value_counts(
        normalize=True
    )
    *
    100
)


# ==========================================
# FINAL STATISTICS
# ==========================================

print(
    "\nFeature Statistics:"
)

print(
    training_df[
        features
    ].describe()
)


# ==========================================
# SAVE DATASET
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# FINAL RESULT
# ==========================================

print(
    "\n======================================"
)

print(
    "FINAL TRAINING DATASET READY"
)

print(
    "======================================"
)


print(
    "\nFinal Shape:",
    training_df.shape
)


print(
    "\nFinal Missing Values:"
)

print(
    training_df[
        features
    ].isnull().sum()
)


print(
    "\n✅ Training dataset saved at:"
)

print(
    OUTPUT_FILE
)