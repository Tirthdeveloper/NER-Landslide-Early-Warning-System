"""
risk_prediction.py
------------------

Optimized Landslide Risk Prediction System
for NER Landslide Early Warning System.

Uses:
- Optimized XGBoost model
- Base environmental inputs
- Automatically generated engineered features

Run:
    python src/risk_prediction.py
"""

import os
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

candidate_models = [
    BASE_DIR / "Models" / "landslide_model_optimized.pkl",
    BASE_DIR / "models" / "landslide_model_optimized.pkl",
    Path("Models/landslide_model_optimized.pkl"),
    Path("models/landslide_model_optimized.pkl")
]
MODEL_FILE = next((p for p in candidate_models if p.exists()), candidate_models[0])

candidate_features = [
    BASE_DIR / "Models" / "model_features_optimized.pkl",
    BASE_DIR / "models" / "model_features_optimized.pkl",
    Path("Models/model_features_optimized.pkl"),
    Path("models/model_features_optimized.pkl")
]
FEATURE_FILE = next((p for p in candidate_features if p.exists()), candidate_features[0])

# ==========================================
# LOAD MODEL
# ==========================================

model = None
features = []

if MODEL_FILE.exists() and FEATURE_FILE.exists():
    try:
        model = joblib.load(str(MODEL_FILE))
        features = joblib.load(str(FEATURE_FILE))
        print("[INFO] Optimized model loaded successfully")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
else:
    print(f"[WARN] Model or feature file not found at {MODEL_FILE}")


def _heuristic_probability(
    rainfall_24h_mm: float,
    rainfall_3d_mm: float,
    rainfall_7d_mm: float,
    soil_water_1: float,
    soil_water_2: float,
    slope: float,
    elevation: float
) -> float:
    """Resilient heuristic risk probability when ML model is unavailable."""
    score = 0.0

    if rainfall_24h_mm >= 70:
        score += 25
    elif rainfall_24h_mm >= 35:
        score += 15
    elif rainfall_24h_mm >= 15:
        score += 8

    if rainfall_7d_mm >= 250:
        score += 20
    elif rainfall_7d_mm >= 120:
        score += 12
    elif rainfall_7d_mm >= 60:
        score += 6

    if slope >= 38:
        score += 30
    elif slope >= 25:
        score += 20
    elif slope >= 15:
        score += 10

    avg_soil = (soil_water_1 + soil_water_2) / 2.0
    if avg_soil >= 0.40:
        score += 15
    elif avg_soil >= 0.28:
        score += 10
    elif avg_soil >= 0.20:
        score += 5

    if elevation >= 1500:
        score += 10
    elif elevation >= 800:
        score += 5

    prob = min(max(score / 100.0, 0.05), 0.95)
    return round(float(prob), 4)


# ==========================================
# RISK LEVEL FUNCTION
# ==========================================

def get_risk_level(
    probability
):

    """
    Convert model probability into
    operational risk level.
    """

    if probability < 0.30:

        return "LOW"

    elif probability < 0.60:

        return "MODERATE"

    elif probability < 0.80:

        return "HIGH"

    else:

        return "CRITICAL"


# ==========================================
# RISK MESSAGE
# ==========================================

def get_risk_message(
    risk_level
):

    if risk_level == "LOW":

        return (
            "Low model-estimated landslide risk. "
            "Continue routine monitoring."
        )

    elif risk_level == "MODERATE":

        return (
            "Moderate landslide risk. "
            "Monitor rainfall, soil moisture "
            "and vulnerable slopes."
        )

    elif risk_level == "HIGH":

        return (
            "High landslide risk. "
            "Increase field monitoring and "
            "prepare local response teams."
        )

    else:

        return (
            "Critical model-estimated landslide risk. "
            "Immediate field verification, "
            "road monitoring and emergency "
            "preparedness are recommended."
        )


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_engineered_features(
    rainfall_24h_mm,
    rainfall_3d_mm,
    rainfall_7d_mm,
    soil_water_layer_1,
    soil_water_layer_2,
    elevation_m,
    slope_degree
):

    """
    Create the same engineered features
    that were used during optimized model training.
    """

    # ======================================
    # RAINFALL AVERAGES
    # ======================================

    rainfall_3d_avg = (
        rainfall_3d_mm
        /
        3
    )


    rainfall_7d_avg = (
        rainfall_7d_mm
        /
        7
    )


    # ======================================
    # RAINFALL RATIO
    # ======================================

    rainfall_24h_7d_ratio = (
        rainfall_24h_mm
        /
        (
            rainfall_7d_mm
            +
            0.001
        )
    )


    # ======================================
    # SOIL MOISTURE AVERAGE
    # ======================================

    soil_moisture_avg = (

        soil_water_layer_1
        +
        soil_water_layer_2

    ) / 2


    # ======================================
    # SLOPE + RAINFALL
    # ======================================

    slope_rainfall_interaction = (

        slope_degree
        *
        rainfall_7d_mm

    )


    # ======================================
    # SOIL + RAINFALL
    # ======================================

    soil_rainfall_interaction = (

        soil_moisture_avg
        *
        rainfall_7d_mm

    )


    # ======================================
    # ELEVATION + SLOPE
    # ======================================

    elevation_slope_interaction = (

        elevation_m
        *
        slope_degree

    )


    # ======================================
    # TERRAIN RISK INDEX
    # ======================================

    terrain_risk_index = (

        slope_degree
        *
        (
            elevation_m
            /
            1000
        )

    )


    return {

        "rainfall_3d_avg":
            rainfall_3d_avg,

        "rainfall_7d_avg":
            rainfall_7d_avg,

        "rainfall_24h_7d_ratio":
            rainfall_24h_7d_ratio,

        "soil_moisture_avg":
            soil_moisture_avg,

        "slope_rainfall_interaction":
            slope_rainfall_interaction,

        "soil_rainfall_interaction":
            soil_rainfall_interaction,

        "elevation_slope_interaction":
            elevation_slope_interaction,

        "terrain_risk_index":
            terrain_risk_index
    }


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_landslide_risk(
    rainfall_24h_mm,
    rainfall_3d_mm,
    rainfall_7d_mm,
    temperature_c,
    soil_water_layer_1,
    soil_water_layer_2,
    surface_pressure_hpa,
    elevation_m,
    slope_degree,
    aspect_degree,
    landcover_code
):

    """
    Predict landslide susceptibility
    using optimized XGBoost model.
    """

    # ======================================
    # BASE FEATURES
    # ======================================

    input_data = {

        "rainfall_24h_mm":
            rainfall_24h_mm,

        "rainfall_3d_mm":
            rainfall_3d_mm,

        "rainfall_7d_mm":
            rainfall_7d_mm,

        "temperature_c":
            temperature_c,

        "soil_water_layer_1":
            soil_water_layer_1,

        "soil_water_layer_2":
            soil_water_layer_2,

        "surface_pressure_hpa":
            surface_pressure_hpa,

        "elevation_m":
            elevation_m,

        "slope_degree":
            slope_degree,

        "aspect_degree":
            aspect_degree,

        "landcover_code":
            landcover_code
    }


    # ======================================
    # ENGINEERED FEATURES
    # ======================================

    engineered = create_engineered_features(

        rainfall_24h_mm=
            rainfall_24h_mm,

        rainfall_3d_mm=
            rainfall_3d_mm,

        rainfall_7d_mm=
            rainfall_7d_mm,

        soil_water_layer_1=
            soil_water_layer_1,

        soil_water_layer_2=
            soil_water_layer_2,

        elevation_m=
            elevation_m,

        slope_degree=
            slope_degree
    )


    # Add engineered features
    input_data.update(
        engineered
    )


    # ======================================
    # DATAFRAME
    # ======================================

    input_df = pd.DataFrame(
        [
            input_data
        ]
    )


    # ======================================
    # FEATURE ORDER & PREDICTION
    # ======================================

    probability = None

    if model is not None and len(features) > 0:
        try:
            # Reindex to ensure all training features exist and in exact order
            aligned_df = input_df.reindex(columns=features, fill_value=0)
            raw_prob = model.predict_proba(aligned_df)[0][1]
            probability = float(raw_prob)
        except Exception as pred_err:
            print(f"[WARN] Model inference failed: {pred_err}. Using heuristic fallback.")

    if probability is None:
        probability = _heuristic_probability(
            rainfall_24h_mm=rainfall_24h_mm,
            rainfall_3d_mm=rainfall_3d_mm,
            rainfall_7d_mm=rainfall_7d_mm,
            soil_water_1=soil_water_layer_1,
            soil_water_2=soil_water_layer_2,
            slope=slope_degree,
            elevation=elevation_m
        )


    # ======================================
    # RISK SCORE
    # ======================================

    risk_score = (
        probability
        *
        100
    )


    # ======================================
    # RISK LEVEL
    # ======================================

    risk_level = get_risk_level(
        probability
    )


    # ======================================
    # MESSAGE
    # ======================================

    message = get_risk_message(
        risk_level
    )


    # ======================================
    # RESULT
    # ======================================

    return {

        "probability":
            probability,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "message":
            message,

        "engineered_features":
            engineered
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "OPTIMIZED LANDSLIDE RISK TEST"
    )

    print(
        "======================================"
    )


    result = predict_landslide_risk(

        rainfall_24h_mm=
            80,

        rainfall_3d_mm=
            180,

        rainfall_7d_mm=
            350,

        temperature_c=
            24,

        soil_water_layer_1=
            0.40,

        soil_water_layer_2=
            0.38,

        surface_pressure_hpa=
            900,

        elevation_m=
            1500,

        slope_degree=
            35,

        aspect_degree=
            180,

        landcover_code=
            10
    )


    print(
        "\nModel Probability:"
    )

    print(
        f"{result['probability']:.4f}"
    )


    print(
        "\nRisk Score:"
    )

    print(
        f"{result['risk_score']:.2f}%"
    )


    print(
        "\nRisk Level:"
    )

    print(
        result[
            "risk_level"
        ]
    )


    print(
        "\nRecommendation:"
    )

    print(
        result[
            "message"
        ]
    )


    print(
        "\nEngineered Features:"
    )

    for name, value in (
        result[
            "engineered_features"
        ].items()
    ):

        print(
            f"{name}: "
            f"{value:.4f}"
        )


    print(
        "\n======================================"
    )