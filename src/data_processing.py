import os
import pandas as pd
import unicodedata


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(r"C:\Gen_AI_series11\app1\Data\raw\Global_Landslide_Catalog_Export_rows.csv")

print("Original Dataset Shape:")
print(df.shape)


# ==========================================
# FILTER INDIA
# ==========================================

india_df = df[df["country_name"] == "India"].copy()

print("\nIndia Records:")
print(india_df.shape)


# ==========================================
# NORMALIZE STATE NAMES
# ==========================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    # Remove accents
    # Example:
    # Nāgāland -> Nagaland
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )

    return value.strip().lower()


india_df["state_normalized"] = (
    india_df["admin_division_name"]
    .apply(normalize_text)
)


# ==========================================
# NER STATES
# ==========================================

ner_states = {
    "assam": "Assam",
    "arunachal pradesh": "Arunachal Pradesh",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura"
}


india_df["ner_state"] = (
    india_df["state_normalized"]
    .map(ner_states)
)


# Keep only NER records
ner_df = india_df[
    india_df["ner_state"].notna()
].copy()


print("\nNER Dataset Shape:")
print(ner_df.shape)


print("\nNER State Distribution:")
print(
    ner_df["ner_state"]
    .value_counts()
)


# ==========================================
# CLEAN EVENT DATE
# ==========================================

ner_df["event_date"] = pd.to_datetime(
    ner_df["event_date"],
    format="mixed",
    errors="coerce"
)


# ==========================================
# SELECT IMPORTANT COLUMNS
# ==========================================

columns = [
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
    "latitude"
]


ner_clean = ner_df[columns].copy()


# ==========================================
# REMOVE INVALID COORDINATES / DATES
# ==========================================

ner_clean = ner_clean.dropna(
    subset=[
        "event_date",
        "latitude",
        "longitude"
    ]
)


# ==========================================
# REMOVE DUPLICATES
# ==========================================

ner_clean = ner_clean.drop_duplicates()


# ==========================================
# SORT BY DATE
# ==========================================

ner_clean = ner_clean.sort_values(
    by="event_date"
)


# ==========================================
# CREATE PROCESSED FOLDER
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# SAVE CLEAN DATASET
# ==========================================

output_path = (
    "data/processed/"
    "ner_landslide_cleaned.csv"
)

ner_clean.to_csv(
    output_path,
    index=False
)


# ==========================================
# FINAL INFORMATION
# ==========================================

print("\nFinal Clean Dataset Shape:")
print(ner_clean.shape)


print("\nMissing Values:")
print(ner_clean.isnull().sum())


print("\nDate Range:")
print(
    ner_clean["event_date"].min(),
    "to",
    ner_clean["event_date"].max()
)


print("\nNER State Distribution After Cleaning:")
print(
    ner_clean["ner_state"]
    .value_counts()
)


print("\n✅ NER dataset cleaned successfully!")

print(
    f"\nSaved at: {output_path}"
)