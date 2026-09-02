"""
eda.py
------

Exploratory Data Analysis for the
AI-Based Landslide Early Warning System - NER

This script analyses the cleaned landslide dataset
and creates useful graphs for understanding
landslide patterns in North-East India.

Run with:
    python src/eda.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# FILE PATHS
# ==========================================

OUTPUT_FOLDER = "outputs/eda"


# Create output folder automatically
os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================
# LOAD CLEAN DATASET
# ==========================================

print("\nLoading cleaned NER landslide dataset...")

df = pd.read_csv(r"C:\Gen_AI_series11\app1\Data\Processed\ner_landslide_cleaned.csv")

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# CONVERT DATE
# ==========================================

df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)


# Create Year and Month columns
df["year"] = df["event_date"].dt.year

df["month"] = df["event_date"].dt.month_name()


# ==========================================
# BASIC DATASET INFORMATION
# ==========================================

print("\n======================================")
print("BASIC DATASET INFORMATION")
print("======================================")

print("\nTotal Landslide Events:")
print(len(df))

print("\nDate Range:")
print(
    df["event_date"].min(),
    "to",
    df["event_date"].max()
)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 1. LANDSLIDES BY STATE
# ==========================================

print("\nCreating Landslides by State graph...")

state_counts = (
    df["ner_state"]
    .value_counts()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=state_counts.values,
    y=state_counts.index
)

plt.title(
    "Landslide Events by State in North-East India"
)

plt.xlabel("Number of Landslide Events")

plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/landslides_by_state.png",
    dpi=300
)

plt.close()


# ==========================================
# 2. LANDSLIDES BY YEAR
# ==========================================

print("Creating Landslides by Year graph...")

year_counts = (
    df["year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

plt.plot(
    year_counts.index,
    year_counts.values,
    marker="o"
)

plt.title(
    "Landslide Events by Year"
)

plt.xlabel("Year")

plt.ylabel("Number of Landslide Events")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/landslides_by_year.png",
    dpi=300
)

plt.close()


# ==========================================
# 3. LANDSLIDES BY MONTH
# ==========================================

print("Creating Landslides by Month graph...")

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


month_counts = (
    df["month"]
    .value_counts()
    .reindex(
        month_order,
        fill_value=0
    )
)


plt.figure(figsize=(12, 6))

sns.barplot(
    x=month_counts.index,
    y=month_counts.values
)

plt.title(
    "Monthly Distribution of Landslide Events"
)

plt.xlabel("Month")

plt.ylabel("Number of Landslide Events")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/landslides_by_month.png",
    dpi=300
)

plt.close()


# ==========================================
# 4. LANDSLIDE TRIGGER ANALYSIS
# ==========================================

print("Creating Landslide Trigger graph...")

trigger_counts = (
    df["landslide_trigger"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=trigger_counts.values,
    y=trigger_counts.index
)

plt.title(
    "Top Landslide Triggers"
)

plt.xlabel("Number of Events")

plt.ylabel("Trigger")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/landslide_triggers.png",
    dpi=300
)

plt.close()


# ==========================================
# 5. LANDSLIDE SIZE ANALYSIS
# ==========================================

print("Creating Landslide Size graph...")

size_counts = (
    df["landslide_size"]
    .fillna("Unknown")
    .value_counts()
)


plt.figure(figsize=(9, 6))

sns.barplot(
    x=size_counts.index,
    y=size_counts.values
)

plt.title(
    "Distribution of Landslide Size"
)

plt.xlabel("Landslide Size")

plt.ylabel("Number of Events")

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/landslide_size.png",
    dpi=300
)

plt.close()


# ==========================================
# 6. FATALITY ANALYSIS
# ==========================================

print("Creating Fatality Analysis graph...")

fatalities_by_state = (
    df.groupby("ner_state")["fatality_count"]
    .sum()
    .sort_values(
        ascending=False
    )
)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=fatalities_by_state.values,
    y=fatalities_by_state.index
)

plt.title(
    "Reported Landslide Fatalities by State"
)

plt.xlabel("Total Reported Fatalities")

plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/fatalities_by_state.png",
    dpi=300
)

plt.close()


# ==========================================
# PRINT IMPORTANT RESULTS
# ==========================================

print("\n======================================")
print("NER LANDSLIDE EDA RESULTS")
print("======================================")


print("\nLandslides by State:")
print(state_counts)


print("\nLandslides by Year:")
print(year_counts)


print("\nLandslides by Month:")
print(month_counts)


print("\nTop Landslide Triggers:")
print(trigger_counts)


print("\nLandslide Sizes:")
print(size_counts)


print("\nFatalities by State:")
print(fatalities_by_state)


# ==========================================
# COMPLETED
# ==========================================

print("\n======================================")

print("EDA COMPLETED SUCCESSFULLY!")

print(
    f"Graphs saved inside: {OUTPUT_FOLDER}"
)

print("======================================")