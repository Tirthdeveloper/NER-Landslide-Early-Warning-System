"""
train_xgboost.py
----------------

Train XGBoost model for
NER Landslide Early Warning System.

Run:
    python src/train_xgboost.py
"""

import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# FILE PATHS
# ==========================================

DATA_FILE = (
    "data/processed/"
    "ner_landslide_training.csv"
)

MODEL_FOLDER = "models"

OUTPUT_FOLDER = (
    "outputs/xgboost"
)

MODEL_FILE = (
    "models/"
    "landslide_xgboost_model.pkl"
)

FEATURE_FILE = (
    "models/"
    "model_features.pkl"
)


# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATA_FILE):

    print(
        f"❌ Dataset not found: "
        f"{DATA_FILE}"
    )

    raise SystemExit


# ==========================================
# LOAD DATASET
# ==========================================

print(
    "\nLoading training dataset..."
)

df = pd.read_csv(
    DATA_FILE
)

print(
    "Dataset Shape:",
    df.shape
)


# ==========================================
# FEATURES
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


target = (
    "landslide_occurred"
)


# ==========================================
# CHECK COLUMNS
# ==========================================

for column in (
    features + [target]
):

    if column not in df.columns:

        print(
            f"❌ Missing column: "
            f"{column}"
        )

        raise SystemExit


# ==========================================
# INPUT AND TARGET
# ==========================================

X = df[
    features
].copy()

y = df[
    target
].copy()


print(
    "\nClass Distribution:"
)

print(
    y.value_counts()
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )
)


print(
    "\nTraining Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=5,

    min_child_weight=1,

    subsample=0.8,

    colsample_bytree=0.8,

    gamma=0,

    reg_alpha=0,

    reg_lambda=1,

    eval_metric="logloss",

    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

print(
    "\nTraining XGBoost model..."
)

model.fit(
    X_train,
    y_train
)

print(
    "✅ XGBoost training completed"
)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print(
    "\n======================================"
)

print(
    "XGBOOST MODEL RESULTS"
)

print(
    "======================================"
)


print(
    f"\nAccuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall: "
    f"{recall:.4f}"
)

print(
    f"F1 Score: "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)


display = (
    ConfusionMatrixDisplay(
        confusion_matrix=cm
    )
)


display.plot()

plt.title(
    "XGBoost Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/"
    f"xgboost_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame(
    {
        "feature":
            features,

        "importance":
            model.feature_importances_
    }
)


importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)


print(
    "\nFeature Importance:"
)

print(
    importance_df
)


importance_df.to_csv(
    f"{OUTPUT_FOLDER}/"
    f"feature_importance.csv",
    index=False
)


plt.figure(
    figsize=(10, 6)
)

plt.barh(
    importance_df[
        "feature"
    ],
    importance_df[
        "importance"
    ]
)

plt.gca().invert_yaxis()

plt.title(
    "XGBoost Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/"
    f"xgboost_feature_importance.png",
    dpi=300
)

plt.close()


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_FILE
)


joblib.dump(
    features,
    FEATURE_FILE
)


# ==========================================
# SAVE METRICS
# ==========================================

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "XGBoost",

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "roc_auc":
                roc_auc
        }
    ]
)


metrics_df.to_csv(
    f"{OUTPUT_FOLDER}/"
    f"xgboost_metrics.csv",
    index=False
)


# ==========================================
# COMPLETED
# ==========================================

print(
    "\n======================================"
)

print(
    "XGBOOST MODEL SAVED SUCCESSFULLY"
)

print(
    "======================================"
)


print(
    "\n✅ Model saved at:"
)

print(
    MODEL_FILE
)


print(
    "\n✅ Feature list saved at:"
)

print(
    FEATURE_FILE
)