"""
rebalance_and_train.py
----------------------
Augments the NER Landslide training dataset with realistic negative control cases:
- Low-rainfall / dry-season days across all 8 NER states
- Gentle slopes and river valleys (slope < 8 deg)
- Urban built-up areas under stable non-failure conditions
- Altitude-adjusted barometric surface pressure

Then retrains the optimized XGBoost model and saves model artifacts to Models/ and models/.
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Data" / "Processed" / "ner_landslide_training.csv"
BACKUP_FILE = BASE_DIR / "Data" / "Processed" / "ner_landslide_training_orig.csv"

# Load original dataset
df_orig = pd.read_csv(DATA_FILE)
if not BACKUP_FILE.exists():
    df_orig.to_csv(BACKUP_FILE, index=False)
    print(f"Backed up original dataset to {BACKUP_FILE}")

print(f"Original dataset: {df_orig.shape}, classes: {dict(df_orig['landslide_occurred'].value_counts())}")

# State topography profiles
STATE_PROFILES = {
    "Assam": {"elev_range": (30, 700), "slope_range": (0.2, 18.0), "base_temp": 28.0},
    "Arunachal Pradesh": {"elev_range": (300, 3200), "slope_range": (5.0, 38.0), "base_temp": 19.0},
    "Manipur": {"elev_range": (600, 1800), "slope_range": (2.0, 24.0), "base_temp": 22.0},
    "Meghalaya": {"elev_range": (200, 1800), "slope_range": (3.0, 28.0), "base_temp": 20.0},
    "Mizoram": {"elev_range": (300, 1600), "slope_range": (4.0, 28.0), "base_temp": 22.0},
    "Nagaland": {"elev_range": (400, 2200), "slope_range": (5.0, 30.0), "base_temp": 20.0},
    "Sikkim": {"elev_range": (600, 3500), "slope_range": (8.0, 42.0), "base_temp": 16.0},
    "Tripura": {"elev_range": (15, 350), "slope_range": (0.5, 12.0), "base_temp": 27.0},
}

def calc_pressure(elev):
    return round(1013.25 * (1.0 - 0.0000225577 * elev) ** 5.25588, 2)

negative_samples = []

# 1. Dry / Light weather non-events across all states (320 samples)
for i in range(320):
    state = random.choice(list(STATE_PROFILES.keys()))
    prof = STATE_PROFILES[state]
    elev = round(random.uniform(*prof["elev_range"]), 1)
    slope = round(random.uniform(*prof["slope_range"]), 2)
    aspect = round(random.uniform(0.0, 360.0), 1)
    r24 = round(random.expovariate(1.0 / 4.0), 2)  # Mean ~4mm, mostly 0-15mm
    r24 = min(r24, 25.0)
    r3d = round(r24 + random.uniform(0.0, 15.0), 2)
    r7d = round(r3d + random.uniform(0.0, 35.0), 2)
    soil1 = round(random.uniform(0.12, 0.28), 3)
    soil2 = round(random.uniform(0.14, 0.30), 3)
    temp = round(prof["base_temp"] + random.uniform(-4.0, 4.0), 1)
    press = calc_pressure(elev)
    # Balanced landcovers including Built-up (50)
    lc = random.choice([10, 10, 20, 30, 40, 50, 50, 60])
    
    negative_samples.append({
        "event_id": f"CTRL_DRY_{i+1}",
        "event_date": "2021-02-15",
        "ner_state": state,
        "latitude": 26.0 + random.uniform(-2, 2),
        "longitude": 92.0 + random.uniform(-2, 2),
        "rainfall_24h_mm": r24,
        "rainfall_3d_mm": r3d,
        "rainfall_7d_mm": r7d,
        "temperature_c": temp,
        "soil_water_layer_1": soil1,
        "soil_water_layer_2": soil2,
        "surface_pressure_hpa": press,
        "elevation_m": elev,
        "slope_degree": slope,
        "aspect_degree": aspect,
        "landcover_code": lc,
        "landslide_occurred": 0
    })

# 2. Flat / Low Slope Valley & Plain Samples (even during rain, plains don't landslide) (160 samples)
for i in range(160):
    state = random.choice(["Assam", "Tripura", "Manipur", "Meghalaya"])
    elev = round(random.uniform(20.0, 400.0), 1)
    slope = round(random.uniform(0.1, 7.5), 2)  # Strict flat/gentle
    aspect = round(random.uniform(0.0, 360.0), 1)
    r24 = round(random.uniform(10.0, 120.0), 2)  # Can be moderate or heavy rain
    r3d = round(r24 * random.uniform(1.5, 2.5), 2)
    r7d = round(r3d * random.uniform(1.5, 2.8), 2)
    soil1 = round(random.uniform(0.25, 0.45), 3)
    soil2 = round(random.uniform(0.28, 0.46), 3)
    temp = round(26.0 + random.uniform(-3, 3), 1)
    press = calc_pressure(elev)
    lc = random.choice([40, 50, 50, 80, 10, 30])  # Urban, croplands, water
    
    negative_samples.append({
        "event_id": f"CTRL_PLAIN_{i+1}",
        "event_date": "2021-07-20",
        "ner_state": state,
        "latitude": 26.0 + random.uniform(-2, 2),
        "longitude": 92.0 + random.uniform(-2, 2),
        "rainfall_24h_mm": r24,
        "rainfall_3d_mm": r3d,
        "rainfall_7d_mm": r7d,
        "temperature_c": temp,
        "soil_water_layer_1": soil1,
        "soil_water_layer_2": soil2,
        "surface_pressure_hpa": press,
        "elevation_m": elev,
        "slope_degree": slope,
        "aspect_degree": aspect,
        "landcover_code": lc,
        "landslide_occurred": 0
    })

# 3. Urban built-up stable controls in hilly cities under typical moderate rain (120 samples)
for i in range(120):
    state = random.choice(["Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Arunachal Pradesh"])
    prof = STATE_PROFILES[state]
    elev = round(random.uniform(600.0, 1600.0), 1)
    slope = round(random.uniform(5.0, 20.0), 2)
    aspect = round(random.uniform(0.0, 360.0), 1)
    r24 = round(random.uniform(5.0, 45.0), 2)  # Moderate routine rain
    r3d = round(r24 * 2.0, 2)
    r7d = round(r24 * 3.5, 2)
    soil1 = round(random.uniform(0.20, 0.35), 3)
    soil2 = round(random.uniform(0.22, 0.36), 3)
    temp = round(prof["base_temp"] + random.uniform(-3, 3), 1)
    press = calc_pressure(elev)
    
    negative_samples.append({
        "event_id": f"CTRL_URBAN_{i+1}",
        "event_date": "2021-06-10",
        "ner_state": state,
        "latitude": 25.5 + random.uniform(-1, 1),
        "longitude": 93.0 + random.uniform(-1, 1),
        "rainfall_24h_mm": r24,
        "rainfall_3d_mm": r3d,
        "rainfall_7d_mm": r7d,
        "temperature_c": temp,
        "soil_water_layer_1": soil1,
        "soil_water_layer_2": soil2,
        "surface_pressure_hpa": press,
        "elevation_m": elev,
        "slope_degree": slope,
        "aspect_degree": aspect,
        "landcover_code": 50,  # Explicitly Built-up without failure
        "landslide_occurred": 0
    })

df_neg = pd.DataFrame(negative_samples)
df_combined = pd.concat([df_orig, df_neg], ignore_index=True)
df_combined.to_csv(DATA_FILE, index=False)
print(f"Augmented dataset saved to {DATA_FILE}: shape {df_combined.shape}")
print(f"New class distribution:\n{df_combined['landslide_occurred'].value_counts()}")

# Feature engineering
def build_features(df):
    f_df = df.copy()
    f_df["rainfall_3d_avg"] = f_df["rainfall_3d_mm"] / 3.0
    f_df["rainfall_7d_avg"] = f_df["rainfall_7d_mm"] / 7.0
    f_df["rainfall_24h_7d_ratio"] = f_df["rainfall_24h_mm"] / (f_df["rainfall_7d_mm"] + 0.001)
    f_df["soil_moisture_avg"] = (f_df["soil_water_layer_1"] + f_df["soil_water_layer_2"]) / 2.0
    f_df["slope_rainfall_interaction"] = f_df["slope_degree"] * f_df["rainfall_7d_mm"]
    f_df["soil_rainfall_interaction"] = f_df["soil_moisture_avg"] * f_df["rainfall_7d_mm"]
    f_df["elevation_slope_interaction"] = f_df["elevation_m"] * f_df["slope_degree"]
    f_df["terrain_risk_index"] = f_df["slope_degree"] * (f_df["elevation_m"] / 1000.0)
    return f_df

df_feats = build_features(df_combined)

base_features = [
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

engineered_features = [
    "rainfall_3d_avg",
    "rainfall_7d_avg",
    "rainfall_24h_7d_ratio",
    "soil_moisture_avg",
    "slope_rainfall_interaction",
    "soil_rainfall_interaction",
    "elevation_slope_interaction",
    "terrain_risk_index"
]

feature_cols = base_features + engineered_features
target_col = "landslide_occurred"

X = df_feats[feature_cols].fillna(df_feats[feature_cols].median())
y = df_feats[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

model = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.04,
    max_leaf_nodes=31,
    min_samples_leaf=15,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

# Save model to both Models/ and models/
for target_dir in [BASE_DIR / "Models", BASE_DIR / "models"]:
    os.makedirs(target_dir, exist_ok=True)
    joblib.dump(model, target_dir / "landslide_model_optimized.pkl")
    joblib.dump(feature_cols, target_dir / "model_features_optimized.pkl")
    print(f"Saved optimized model and features to {target_dir}")

print("Training script finished successfully.")
