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


def _calculate_physics_probability(
    rainfall_24h_mm: float,
    rainfall_3d_mm: float,
    rainfall_7d_mm: float,
    soil_water_1: float,
    soil_water_2: float,
    slope: float,
    elevation: float
) -> float:
    """
    Physics-informed geotechnical risk assessment based on Geological Survey of India (GSI)
    and National Disaster Management Authority (NDMA) landslide guidelines.
    Formula: Risk = Terrain Susceptibility x Hydro-meteorological Trigger.
    """
    avg_soil = (float(soil_water_1) + float(soil_water_2)) / 2.0
    r24 = float(rainfall_24h_mm)
    r7 = float(rainfall_7d_mm)
    s = float(slope)
    elev = float(elevation)

    # 1. Rainfall Trigger Index (0.0 to 1.0)
    rain_score = 0.0
    if r24 >= 140:
        rain_score += 40
    elif r24 >= 100:
        rain_score += 34
    elif r24 >= 70:
        rain_score += 26
    elif r24 >= 40:
        rain_score += 18
    elif r24 >= 20:
        rain_score += 10
    elif r24 >= 10:
        rain_score += 4

    # Cumulative antecedent rain (0 to 30)
    cum_score = 0.0
    if r7 >= 350:
        cum_score += 30
    elif r7 >= 250:
        cum_score += 24
    elif r7 >= 150:
        cum_score += 16
    elif r7 >= 80:
        cum_score += 10
    elif r7 >= 40:
        cum_score += 4

    # Soil moisture saturation (0 to 20)
    soil_score = 0.0
    if avg_soil >= 0.44:
        soil_score += 20
    elif avg_soil >= 0.38:
        soil_score += 15
    elif avg_soil >= 0.30:
        soil_score += 10
    elif avg_soil >= 0.22:
        soil_score += 4

    trigger_index = (rain_score + cum_score + soil_score) / 90.0

    # 2. Terrain Susceptibility Factor (0.0 to 1.0) based on GSI NLSM classes
    if s >= 30.0:
        slope_factor = 1.00  # Very High
    elif s >= 20.0:
        slope_factor = 0.82  # High
    elif s >= 12.0:
        slope_factor = 0.62  # Moderate to High
    elif s >= 6.0:
        slope_factor = 0.38  # Moderate
    elif s >= 3.0:
        slope_factor = 0.12  # Low
    else:
        slope_factor = 0.02  # Extremely Low / Flat Plain

    elev_factor = min(max(elev / 2000.0, 0.1), 1.0)
    terrain_susceptibility = min(slope_factor * (0.7 + 0.3 * elev_factor), 1.0)

    # Dynamic risk calculation: without triggering precipitation, steep slopes don't slide.
    # On flat ground, heavy rain causes surface waterlogging / floods, NOT landslides.
    physics_prob = terrain_susceptibility * (0.15 + 0.85 * trigger_index)

    # Hard physical boundary constraints
    if s < 3.0:
        physics_prob = min(physics_prob, 0.12)
    elif s < 6.0:
        physics_prob = min(physics_prob, 0.28)

    if r24 < 12.0 and r7 < 40.0 and avg_soil < 0.28:
        physics_prob = min(physics_prob, 0.20)

    return round(float(min(max(physics_prob, 0.02), 0.98)), 4)


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
    return _calculate_physics_probability(
        rainfall_24h_mm=rainfall_24h_mm,
        rainfall_3d_mm=rainfall_3d_mm,
        rainfall_7d_mm=rainfall_7d_mm,
        soil_water_1=soil_water_1,
        soil_water_2=soil_water_2,
        slope=slope,
        elevation=elevation
    )


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

    # Adjust OpenWeather sea-level normalized pressure to local ground station pressure
    adj_pressure = float(surface_pressure_hpa)
    if adj_pressure > 980.0 and float(elevation_m) > 80.0:
        adj_pressure = round(adj_pressure * ((1.0 - 0.0000225577 * float(elevation_m)) ** 5.25588), 1)

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
            adj_pressure,

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

    raw_prob = None

    if model is not None and len(features) > 0:
        try:
            # Reindex to ensure all training features exist and in exact order
            aligned_df = input_df.reindex(columns=features, fill_value=0)
            raw_val = model.predict_proba(aligned_df)[0][1]
            raw_prob = float(raw_val)
        except Exception as pred_err:
            print(f"[WARN] Model inference failed: {pred_err}. Using heuristic fallback.")

    # Calculate physics-grounded geotechnical probability
    physics_prob = _calculate_physics_probability(
        rainfall_24h_mm=rainfall_24h_mm,
        rainfall_3d_mm=rainfall_3d_mm,
        rainfall_7d_mm=rainfall_7d_mm,
        soil_water_1=soil_water_layer_1,
        soil_water_2=soil_water_layer_2,
        slope=slope_degree,
        elevation=elevation_m
    )

    if raw_prob is not None:
        # Geotechnical blending: validated ML model weighted with physical bounds
        combined = 0.55 * raw_prob + 0.45 * physics_prob

        # Geotechnical safety caps
        s = float(slope_degree)
        r24 = float(rainfall_24h_mm)
        r7 = float(rainfall_7d_mm)
        avg_soil = (float(soil_water_layer_1) + float(soil_water_layer_2)) / 2.0

        if s < 3.0:
            combined = min(combined, 0.12)
        elif s < 6.0:
            combined = min(combined, 0.28)

        if r24 < 12.0 and r7 < 40.0 and avg_soil < 0.28:
            combined = min(combined, 0.20)

        # Operational hazard level alignments:
        # Extreme deluge on steep mountain slopes -> CRITICAL
        if (r24 >= 100.0 or r7 >= 280.0) and s >= 14.0 and avg_soil >= 0.35:
            combined = max(combined, 0.82)
        # Heavy rain on vulnerable mountain slopes -> HIGH
        elif (r24 >= 60.0 or r7 >= 180.0) and s >= 8.0 and avg_soil >= 0.30:
            combined = max(combined, 0.65)
        # Moderate rain on hills -> MODERATE
        elif (r24 >= 25.0 or r7 >= 75.0) and s >= 6.0 and avg_soil >= 0.25:
            combined = max(combined, 0.38)

        probability = float(min(max(combined, 0.02), 0.98))
    else:
        probability = physics_prob


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