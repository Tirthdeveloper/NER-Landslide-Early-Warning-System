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

import joblib
import pandas as pd


# ==========================================
# FILE PATHS
# ==========================================

MODEL_FILE = (
    "models/"
    "landslide_model_optimized.pkl"
)

FEATURE_FILE = (
    "models/"
    "model_features_optimized.pkl"
)


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(MODEL_FILE):

    print(
        f"❌ Optimized model not found: "
        f"{MODEL_FILE}"
    )

    raise SystemExit


if not os.path.exists(FEATURE_FILE):

    print(
        f"❌ Optimized feature file not found: "
        f"{FEATURE_FILE}"
    )

    raise SystemExit


# ==========================================
# LOAD MODEL
# ==========================================

print(
    "\nLoading optimized landslide model..."
)

model = joblib.load(
    MODEL_FILE
)

features = joblib.load(
    FEATURE_FILE
)

print(
    "✅ Optimized model loaded successfully"
)


print(
    "\nModel Features:"
)

for feature in features:

    print(
        f"- {feature}"
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
    # FEATURE ORDER
    # ======================================

    # Ensure exactly the same feature
    # order used during model training.

    input_df = input_df[
        features
    ]


    # ======================================
    # PREDICT PROBABILITY
    # ======================================

    probability = (
        model.predict_proba(
            input_df
        )[0][1]
    )


    probability = float(
        probability
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