"""
train_xgboost_optimized.py
--------------------------

Optimized XGBoost training for
NER Landslide Early Warning System.

Features:
- Feature engineering
- Train/Test split
- RandomizedSearchCV
- 5-fold cross-validation
- Accuracy / Precision / Recall / F1 / ROC-AUC
- Confusion Matrix
- ROC Curve
- Feature Importance
- Best model save

Run with:
    python src/train_xgboost_optimized.py
"""

import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
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
    "outputs/xgboost_optimized"
)

MODEL_FILE = (
    "models/"
    "landslide_model_optimized.pkl"
)

FEATURE_FILE = (
    "models/"
    "model_features_optimized.pkl"
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
# FEATURE ENGINEERING
# ==========================================

print(
    "\nCreating engineered features..."
)


# Rainfall averages
df[
    "rainfall_3d_avg"
] = (
    df[
        "rainfall_3d_mm"
    ]
    /
    3
)


df[
    "rainfall_7d_avg"
] = (
    df[
        "rainfall_7d_mm"
    ]
    /
    7
)


# Rainfall ratio
df[
    "rainfall_24h_7d_ratio"
] = (
    df[
        "rainfall_24h_mm"
    ]
    /
    (
        df[
            "rainfall_7d_mm"
        ]
        +
        0.001
    )
)


# Soil moisture average
df[
    "soil_moisture_avg"
] = (

    df[
        "soil_water_layer_1"
    ]

    +

    df[
        "soil_water_layer_2"
    ]

) / 2


# Rainfall + slope interaction
df[
    "slope_rainfall_interaction"
] = (

    df[
        "slope_degree"
    ]

    *

    df[
        "rainfall_7d_mm"
    ]

)


# Soil moisture + rainfall interaction
df[
    "soil_rainfall_interaction"
] = (

    df[
        "soil_moisture_avg"
    ]

    *

    df[
        "rainfall_7d_mm"
    ]

)


# Elevation + slope interaction
df[
    "elevation_slope_interaction"
] = (

    df[
        "elevation_m"
    ]

    *

    df[
        "slope_degree"
    ]

)


# Terrain rough-risk proxy
df[
    "terrain_risk_index"
] = (

    df[
        "slope_degree"
    ]

    *

    (
        df[
            "elevation_m"
        ]
        /
        1000
    )

)


print(
    "✅ Feature engineering completed"
)


# ==========================================
# ORIGINAL FEATURES
# ==========================================

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


# ==========================================
# ENGINEERED FEATURES
# ==========================================

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


# ==========================================
# FINAL FEATURES
# ==========================================

features = (
    base_features
    +
    engineered_features
)


target = (
    "landslide_occurred"
)


# ==========================================
# CHECK COLUMNS
# ==========================================

for column in (
    features
    +
    [target]
):

    if column not in df.columns:

        print(
            f"❌ Missing column: "
            f"{column}"
        )

        raise SystemExit


# ==========================================
# HANDLE MISSING / INVALID VALUES
# ==========================================

print(
    "\nChecking missing values..."
)


for column in features:

    if df[column].isnull().any():

        median_value = (
            df[column]
            .median()
        )

        df[column] = (
            df[column]
            .fillna(
                median_value
            )
        )


print(
    "✅ Missing values handled"
)


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
# BASE XGBOOST MODEL
# ==========================================

base_model = XGBClassifier(

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ==========================================
# PARAMETER SEARCH SPACE
# ==========================================

parameter_grid = {

    "n_estimators": [
        200,
        300,
        400,
        500,
        700
    ],

    "max_depth": [
        3,
        4,
        5,
        6,
        7
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.08,
        0.1
    ],

    "min_child_weight": [
        1,
        2,
        3,
        5,
        7
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "gamma": [
        0,
        0.1,
        0.2,
        0.3,
        0.5
    ],

    "reg_alpha": [
        0,
        0.01,
        0.05,
        0.1,
        0.5
    ],

    "reg_lambda": [
        1,
        2,
        3,
        5,
        10
    ]
}


# ==========================================
# RANDOMIZED SEARCH
# ==========================================

print(
    "\n======================================"
)

print(
    "STARTING XGBOOST OPTIMIZATION"
)

print(
    "======================================"
)

print(
    "\nThis may take a few minutes..."
)


search = RandomizedSearchCV(

    estimator=
        base_model,

    param_distributions=
        parameter_grid,

    n_iter=
        60,

    scoring=
        "f1",

    cv=
        5,

    verbose=
        1,

    random_state=
        42,

    n_jobs=
        -1,

    return_train_score=
        True
)


search.fit(
    X_train,
    y_train
)


# ==========================================
# BEST PARAMETERS
# ==========================================

print(
    "\n======================================"
)

print(
    "BEST PARAMETERS"
)

print(
    "======================================"
)


print(
    search.best_params_
)


print(
    "\nBest Cross-Validation F1:"
)

print(
    round(
        search.best_score_,
        4
    )
)


# ==========================================
# BEST MODEL
# ==========================================

best_model = (
    search.best_estimator_
)


# ==========================================
# TEST PREDICTION
# ==========================================

y_pred = best_model.predict(
    X_test
)

y_prob = best_model.predict_proba(
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


# ==========================================
# RESULTS
# ==========================================

print(
    "\n======================================"
)

print(
    "OPTIMIZED XGBOOST RESULTS"
)

print(
    "======================================"
)


print(
    f"\nAccuracy:"
    f" {accuracy:.4f}"
)

print(
    f"Precision:"
    f" {precision:.4f}"
)

print(
    f"Recall:"
    f" {recall:.4f}"
)

print(
    f"F1 Score:"
    f" {f1:.4f}"
)

print(
    f"ROC-AUC:"
    f" {roc_auc:.4f}"
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
    "Optimized XGBoost Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    f"{OUTPUT_FOLDER}/"
    f"confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = (
    roc_curve(
        y_test,
        y_prob
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=(
        f"XGBoost "
        f"(AUC={roc_auc:.3f})"
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "XGBoost ROC Curve"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    f"{OUTPUT_FOLDER}/"
    f"roc_curve.png",
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
            best_model
            .feature_importances_
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
    "\n======================================"
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "======================================"
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
    figsize=(10, 8)
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


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Optimized XGBoost Feature Importance"
)

plt.tight_layout()


plt.savefig(
    f"{OUTPUT_FOLDER}/"
    f"feature_importance.png",
    dpi=300
)

plt.close()


# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
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
                "Optimized XGBoost",

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "roc_auc":
                roc_auc,

            "best_cv_f1":
                search.best_score_
        }
    ]
)


metrics_df.to_csv(
    f"{OUTPUT_FOLDER}/"
    f"metrics.csv",
    index=False
)


# ==========================================
# SAVE BEST PARAMETERS
# ==========================================

best_parameters_df = (
    pd.DataFrame(
        [
            search.best_params_
        ]
    )
)


best_parameters_df.to_csv(
    f"{OUTPUT_FOLDER}/"
    f"best_parameters.csv",
    index=False
)


# ==========================================
# COMPLETED
# ==========================================

print(
    "\n======================================"
)

print(
    "OPTIMIZED XGBOOST TRAINING COMPLETED"
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
    "\n✅ Features saved at:"
)

print(
    FEATURE_FILE
)


print(
    "\n✅ Results saved inside:"
)

print(
    OUTPUT_FOLDER
)