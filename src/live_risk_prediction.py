"""
live_risk_prediction.py
-----------------------

Live landslide risk prediction for NER.

Automatically gets:
- Location coordinates
- Temperature
- Pressure
- Elevation
- Slope
- Aspect
- Land-cover

Recent accumulated rainfall and soil-water
values are supplied manually for now.

Run:
    python src/live_risk_prediction.py
"""

import os
import glob

import numpy as np
import rasterio

from src.weather_api import get_current_weather
from src.risk_prediction import predict_landslide_risk


# ==========================================
# FILE PATHS
# ==========================================

DEM_FILE = (
    "data/raw/dem/"
    "ner_dem_90m.tiff"
)

LANDCOVER_FOLDER = (
    "data/raw/"
    "landcover"
)


# ==========================================
# WORLDCOVER CLASSES
# ==========================================

LANDCOVER_CLASSES = {

    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}


# ==========================================
# TERRAIN EXTRACTION
# ==========================================

def get_terrain_features(
    latitude,
    longitude
):

    if not os.path.exists(DEM_FILE):

        print(
            f"❌ DEM file missing: "
            f"{DEM_FILE}"
        )

        return None


    with rasterio.open(
        DEM_FILE
    ) as src:

        dem = (
            src.read(1)
            .astype("float64")
        )

        transform = src.transform

        bounds = src.bounds

        crs = src.crs

        nodata = src.nodata


        if nodata is not None:

            dem[
                dem == nodata
            ] = np.nan


        # ==================================
        # PIXEL SIZE
        # ==================================

        center_latitude = (
            bounds.top
            +
            bounds.bottom
        ) / 2


        if (
            crs is not None
            and
            crs.is_geographic
        ):

            lat_rad = np.radians(
                center_latitude
            )

            meters_per_degree_lat = (
                111320
            )

            meters_per_degree_lon = (
                111320
                *
                np.cos(lat_rad)
            )

            pixel_width_m = (
                abs(transform.a)
                *
                meters_per_degree_lon
            )

            pixel_height_m = (
                abs(transform.e)
                *
                meters_per_degree_lat
            )

        else:

            pixel_width_m = abs(
                transform.a
            )

            pixel_height_m = abs(
                transform.e
            )


        # ==================================
        # SLOPE + ASPECT
        # ==================================

        dz_dy, dz_dx = np.gradient(
            dem,
            pixel_height_m,
            pixel_width_m
        )


        slope = np.degrees(
            np.arctan(
                np.sqrt(
                    dz_dx ** 2
                    +
                    dz_dy ** 2
                )
            )
        )


        aspect = np.degrees(
            np.arctan2(
                -dz_dx,
                dz_dy
            )
        )

        aspect = (
            aspect + 360
        ) % 360


        # ==================================
        # PIXEL LOCATION
        # ==================================

        try:

            row, col = src.index(
                longitude,
                latitude
            )


            if not (
                0 <= row < dem.shape[0]
                and
                0 <= col < dem.shape[1]
            ):

                return None


            elevation = (
                dem[row, col]
            )

            slope_value = (
                slope[row, col]
            )

            aspect_value = (
                aspect[row, col]
            )


            if np.isnan(elevation):

                return None


            return {

                "elevation_m":
                    float(elevation),

                "slope_degree":
                    float(slope_value),

                "aspect_degree":
                    float(aspect_value)
            }


        except Exception:

            return None


# ==========================================
# LAND COVER EXTRACTION
# ==========================================

def get_landcover_features(
    latitude,
    longitude
):

    tif_files = glob.glob(
        os.path.join(
            LANDCOVER_FOLDER,
            "*.tif"
        )
    )


    for tif_file in tif_files:

        with rasterio.open(
            tif_file
        ) as src:

            bounds = src.bounds


            if (
                bounds.left
                <= longitude
                <= bounds.right
                and
                bounds.bottom
                <= latitude
                <= bounds.top
            ):

                try:

                    value = next(
                        src.sample(
                            [
                                (
                                    longitude,
                                    latitude
                                )
                            ]
                        )
                    )[0]


                    code = int(value)


                    if code == 0:
                        return None


                    return {

                        "landcover_code":
                            code,

                        "landcover_class":
                            LANDCOVER_CLASSES.get(
                                code,
                                "Unknown"
                            )
                    }


                except Exception:

                    return None


    return None


# ==========================================
# LIVE RISK FUNCTION
# ==========================================

def get_live_risk(
    city,
    rainfall_24h_mm,
    rainfall_3d_mm,
    rainfall_7d_mm,
    soil_water_layer_1,
    soil_water_layer_2
):

    # ======================================
    # LIVE WEATHER
    # ======================================

    weather = get_current_weather(
        city
    )


    if not weather["success"]:

        return {
            "success": False,
            "message":
                "Weather API failed."
        }


    latitude = (
        weather["latitude"]
    )

    longitude = (
        weather["longitude"]
    )


    # ======================================
    # TERRAIN
    # ======================================

    terrain = get_terrain_features(
        latitude,
        longitude
    )


    if terrain is None:

        return {
            "success": False,
            "message":
                "Terrain data unavailable "
                "for this location."
        }


    # ======================================
    # LAND COVER
    # ======================================

    landcover = get_landcover_features(
        latitude,
        longitude
    )


    if landcover is None:

        return {
            "success": False,
            "message":
                "Land-cover data unavailable "
                "for this location."
        }


    # ======================================
    # ML PREDICTION
    # ======================================

    prediction = predict_landslide_risk(

        rainfall_24h_mm=
            rainfall_24h_mm,

        rainfall_3d_mm=
            rainfall_3d_mm,

        rainfall_7d_mm=
            rainfall_7d_mm,

        temperature_c=
            weather[
                "temperature_c"
            ],

        soil_water_layer_1=
            soil_water_layer_1,

        soil_water_layer_2=
            soil_water_layer_2,

        surface_pressure_hpa=
            weather[
                "surface_pressure_hpa"
            ],

        elevation_m=
            terrain[
                "elevation_m"
            ],

        slope_degree=
            terrain[
                "slope_degree"
            ],

        aspect_degree=
            terrain[
                "aspect_degree"
            ],

        landcover_code=
            landcover[
                "landcover_code"
            ]
    )


    return {

        "success":
            True,

        "city":
            weather["city"],

        "latitude":
            latitude,

        "longitude":
            longitude,

        "temperature_c":
            weather[
                "temperature_c"
            ],

        "humidity":
            weather[
                "humidity"
            ],

        "pressure_hpa":
            weather[
                "surface_pressure_hpa"
            ],

        "weather":
            weather[
                "weather"
            ],

        "elevation_m":
            terrain[
                "elevation_m"
            ],

        "slope_degree":
            terrain[
                "slope_degree"
            ],

        "aspect_degree":
            terrain[
                "aspect_degree"
            ],

        "landcover_code":
            landcover[
                "landcover_code"
            ],

        "landcover_class":
            landcover[
                "landcover_class"
            ],

        "risk_score":
            prediction[
                "risk_score"
            ],

        "risk_level":
            prediction[
                "risk_level"
            ],

        "recommendation":
            prediction[
                "message"
            ]
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "NER LIVE LANDSLIDE RISK TEST"
    )

    print(
        "======================================"
    )


    # ======================================
    # CITY
    # ======================================

    city = input(
        "\nEnter city: "
    )


    # ======================================
    # RECENT WEATHER INPUTS
    # ======================================

    rainfall_24h = float(
        input(
            "Rainfall last 24h (mm): "
        )
    )


    rainfall_3d = float(
        input(
            "Rainfall last 3 days (mm): "
        )
    )


    rainfall_7d = float(
        input(
            "Rainfall last 7 days (mm): "
        )
    )


    soil_water_1 = float(
        input(
            "Soil water layer 1: "
        )
    )


    soil_water_2 = float(
        input(
            "Soil water layer 2: "
        )
    )


    # ======================================
    # PREDICT
    # ======================================

    result = get_live_risk(

        city=city,

        rainfall_24h_mm=
            rainfall_24h,

        rainfall_3d_mm=
            rainfall_3d,

        rainfall_7d_mm=
            rainfall_7d,

        soil_water_layer_1=
            soil_water_1,

        soil_water_layer_2=
            soil_water_2
    )


    # ======================================
    # OUTPUT
    # ======================================

    if not result["success"]:

        print(
            "\n❌",
            result["message"]
        )

        raise SystemExit


    print(
        "\n======================================"
    )

    print(
        "LIVE RISK RESULT"
    )

    print(
        "======================================"
    )


    print(
        "\nLocation:",
        result["city"]
    )


    print(
        "Coordinates:",
        result["latitude"],
        result["longitude"]
    )


    print(
        "\nTemperature:",
        round(
            result[
                "temperature_c"
            ],
            2
        ),
        "°C"
    )


    print(
        "Humidity:",
        result[
            "humidity"
        ],
        "%"
    )


    print(
        "Pressure:",
        result[
            "pressure_hpa"
        ],
        "hPa"
    )


    print(
        "Weather:",
        result[
            "weather"
        ]
    )


    print(
        "\nElevation:",
        round(
            result[
                "elevation_m"
            ],
            2
        ),
        "m"
    )


    print(
        "Slope:",
        round(
            result[
                "slope_degree"
            ],
            2
        ),
        "degrees"
    )


    print(
        "Aspect:",
        round(
            result[
                "aspect_degree"
            ],
            2
        ),
        "degrees"
    )


    print(
        "Land Cover:",
        result[
            "landcover_class"
        ]
    )


    print(
        "\n🚨 Risk Score:",
        f"{result['risk_score']:.2f}%"
    )


    print(
        "Risk Level:",
        result[
            "risk_level"
        ]
    )


    print(
        "\nRecommendation:"
    )

    print(
        result[
            "recommendation"
        ]
    )