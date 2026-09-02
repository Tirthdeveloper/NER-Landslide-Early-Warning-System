"""
gis_map.py
----------

Interactive GIS risk map for
NER Landslide Early Warning System.

Run:
    python src/gis_map.py
"""

import os

import pandas as pd
import folium


# ==========================================
# FILE PATHS
# ==========================================

DATA_FILE = (
    "data/processed/"
    "ner_landslide_training.csv"
)

OUTPUT_FOLDER = (
    "outputs/gis"
)

OUTPUT_FILE = (
    "outputs/gis/"
    "ner_landslide_risk_map.html"
)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

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
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE
)


print(
    "Dataset Shape:",
    df.shape
)


# ==========================================
# RISK LEVEL FUNCTION
# ==========================================

def get_risk_level(
    row
):

    # For GIS prototype,
    # use landslide/control label
    # plus terrain/rainfall indicators.

    score = 0


    # Rainfall contribution
    if row["rainfall_7d_mm"] >= 300:
        score += 35

    elif row["rainfall_7d_mm"] >= 150:
        score += 25

    elif row["rainfall_7d_mm"] >= 75:
        score += 15


    # Slope contribution
    if row["slope_degree"] >= 40:
        score += 30

    elif row["slope_degree"] >= 25:
        score += 20

    elif row["slope_degree"] >= 15:
        score += 10


    # Elevation contribution
    if row["elevation_m"] >= 1500:
        score += 15

    elif row["elevation_m"] >= 700:
        score += 10


    # Soil moisture contribution
    if row["soil_water_layer_1"] >= 0.40:
        score += 20

    elif row["soil_water_layer_1"] >= 0.25:
        score += 10


    # ======================================
    # CLASSIFY RISK
    # ======================================

    if score < 30:
        return "LOW", score

    elif score < 60:
        return "MODERATE", score

    elif score < 80:
        return "HIGH", score

    else:
        return "CRITICAL", score


# ==========================================
# MAP COLORS
# ==========================================

risk_colors = {

    "LOW":
        "green",

    "MODERATE":
        "orange",

    "HIGH":
        "darkorange",

    "CRITICAL":
        "red"
}


# ==========================================
# CREATE BASE MAP
# ==========================================

print(
    "\nCreating NER GIS map..."
)


ner_map = folium.Map(

    location=[
        26.0,
        93.0
    ],

    zoom_start=6,

    tiles="OpenStreetMap"
)


# ==========================================
# ADD TITLE
# ==========================================

title_html = """
<div style="
position: fixed;
top: 10px;
left: 50%;
transform: translateX(-50%);
z-index: 9999;
background-color: white;
padding: 10px 20px;
border-radius: 10px;
box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
font-size: 20px;
font-weight: bold;
">
🏔️ NER Landslide Risk Monitoring Map
</div>
"""


ner_map.get_root().html.add_child(
    folium.Element(
        title_html
    )
)


# ==========================================
# ADD RISK POINTS
# ==========================================

print(
    "Adding risk locations..."
)


for _, row in df.iterrows():

    latitude = row["latitude"]

    longitude = row["longitude"]


    risk_level, risk_score = (
        get_risk_level(
            row
        )
    )


    color = risk_colors[
        risk_level
    ]


    popup_html = f"""
    <b>NER Landslide Risk</b><br><br>

    <b>State:</b>
    {row['ner_state']}<br>

    <b>Risk Level:</b>
    {risk_level}<br>

    <b>Risk Score:</b>
    {risk_score}/100<br><br>

    <b>Rainfall 24h:</b>
    {row['rainfall_24h_mm']:.2f} mm<br>

    <b>Rainfall 3d:</b>
    {row['rainfall_3d_mm']:.2f} mm<br>

    <b>Rainfall 7d:</b>
    {row['rainfall_7d_mm']:.2f} mm<br><br>

    <b>Elevation:</b>
    {row['elevation_m']:.2f} m<br>

    <b>Slope:</b>
    {row['slope_degree']:.2f}°<br>

    <b>Soil Water:</b>
    {row['soil_water_layer_1']:.3f}<br>

    <b>Land Cover Code:</b>
    {row['landcover_code']}
    """


    folium.CircleMarker(

        location=[
            latitude,
            longitude
        ],

        radius=6,

        popup=folium.Popup(
            popup_html,
            max_width=350
        ),

        color=color,

        fill=True,

        fill_color=color,

        fill_opacity=0.7

    ).add_to(
        ner_map
    )


# ==========================================
# ADD LEGEND
# ==========================================

legend_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
z-index: 9999;
background-color: white;
padding: 15px;
border-radius: 10px;
box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
font-size: 14px;
">

<b>Risk Severity</b><br><br>

<span style="color:green;">●</span>
LOW<br>

<span style="color:orange;">●</span>
MODERATE<br>

<span style="color:darkorange;">●</span>
HIGH<br>

<span style="color:red;">●</span>
CRITICAL

</div>
"""


ner_map.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ==========================================
# SAVE MAP
# ==========================================

ner_map.save(
    OUTPUT_FILE
)


# ==========================================
# COMPLETED
# ==========================================

print(
    "\n======================================"
)

print(
    "GIS MAP CREATED SUCCESSFULLY"
)

print(
    "======================================"
)


print(
    "\n✅ Map saved at:"
)

print(
    OUTPUT_FILE
)