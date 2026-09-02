"""
train_model.py
--------------

Train and compare ML models for
NER Landslide Early Warning System.

Models:
- Random Forest
- XGBoost
- LightGBM

Run with:
    python src/train_model.py
"""

import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================
# FILE PATHS
# ==========================================

DATA_FILE = (
    "data/processed/"
    "ner_landslide_training.csv"
)

MODEL_FOLDER = "models"

OUTPUT_FOLDER = "outputs/model"

BEST_MODEL_FILE = (
    "models/"
    "landslide_model.pkl"
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
        f"❌ Training dataset not found: "
        f"{DATA_FILE}"
    )

    raise SystemExit


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading training dataset...")

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


target = "landslide_occurred"


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

for column in features + [target]:

    if column not in df.columns:

        print(
            f"❌ Missing column: "
            f"{column}"
        )

        raise SystemExit


# ==========================================
# INPUT / TARGET
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
# TRAIN / TEST SPLIT
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
# MODELS
# ==========================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        )
}


# ==========================================
# STORE RESULTS
# ==========================================

results = []

trained_models = {}


# ==========================================
# TRAIN MODELS
# ==========================================

for model_name, model in models.items():

    print(
        "\n======================================"
    )

    print(
        f"Training: {model_name}"
    )

    print(
        "======================================"
    )


    # Train model
    model.fit(
        X_train,
        y_train
    )


    # Prediction
    y_pred = model.predict(
        X_test
    )


    # Probability
    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # ======================================
    # METRICS
    # ======================================

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
        "\nAccuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Precision:",
        round(
            precision,
            4
        )
    )

    print(
        "Recall:",
        round(
            recall,
            4
        )
    )

    print(
        "F1 Score:",
        round(
            f1,
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc,
            4
        )
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


    # ======================================
    # CONFUSION MATRIX
    # ======================================

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
        f"{model_name} "
        f"Confusion Matrix"
    )


    safe_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    plt.tight_layout()


    plt.savefig(
        f"{OUTPUT_FOLDER}/"
        f"{safe_name}_confusion_matrix.png",
        dpi=300
    )


    plt.close()


    # ======================================
    # SAVE METRICS
    # ======================================

    results.append(
        {
            "model":
                model_name,

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
    )


    trained_models[
        model_name
    ] = model


# ==========================================
# RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(
    results
)


print(
    "\n======================================"
)

print(
    "MODEL COMPARISON"
)

print(
    "======================================"
)


print(
    results_df
    .sort_values(
        by="f1_score",
        ascending=False
    )
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_df.to_csv(
    f"{OUTPUT_FOLDER}/"
    f"model_comparison.csv",
    index=False
)


# ==========================================
# SELECT BEST MODEL
# ==========================================

best_row = (
    results_df
    .sort_values(
        by=[
            "f1_score",
            "recall",
            "roc_auc"
        ],
        ascending=False
    )
    .iloc[0]
)


best_model_name = (
    best_row[
        "model"
    ]
)


best_model = (
    trained_models[
        best_model_name
    ]
)


print(
    "\n🏆 Best Model:"
)

print(
    best_model_name
)


print(
    "\nBest Model Metrics:"
)

print(
    best_row
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    BEST_MODEL_FILE
)


# ==========================================
# SAVE FEATURE LIST
# ==========================================

joblib.dump(
    features,
    "models/"
    "model_features.pkl"
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    features,

                "importance":
                    best_model
                    .feature_importances_
            }
        )
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
        "Landslide Model "
        "Feature Importance"
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
        f"feature_importance.png",
        dpi=300
    )


    plt.close()


# ==========================================
# COMPLETED
# ==========================================

print(
    "\n======================================"
)

print(
    "MODEL TRAINING COMPLETED"
)

print(
    "======================================"
)


print(
    "\n✅ Best model saved at:"
)

print(
    BEST_MODEL_FILE
)


print(
    "\n✅ Model features saved at:"
)

print(
    "models/model_features.pkl"
)


print(
    "\n✅ Model comparison saved at:"
)

print(
    "outputs/model/model_comparison.csv"
)