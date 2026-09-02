"""
app.py
------

NER AI-Based Landslide Early Warning System

Features:
- North Eastern Region overview
- State-wise AI landslide risk prediction
- Live weather monitoring
- GIS risk map
- Citizen / Field Officer reporting
- Emergency email alerts
- Historical analytics

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
import folium

from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

load_dotenv()

from streamlit_folium import st_folium

from src.weather_api import get_current_weather
from src.live_risk_prediction import get_live_risk

from src.citizen_reporting import (
    save_citizen_report,
    load_reports
)

from src.alerts import send_email_alert
from src.sms_alert import send_sms_alert


from src.road_connectivity import (
    assess_road_connectivity
)

from src.emergency_prioritisation import (
    calculate_emergency_priority
)

from src.multilingual_alerts import (
    LANGUAGES,
    generate_multilingual_alert
)

from src.offline_support import (
    get_offline_status,
    queue_alert
)



# ==========================================
# UI / SYSTEM HELPERS
# ==========================================

def get_current_timestamp():
    return datetime.now().strftime("%d %b %Y, %I:%M:%S %p")


def render_system_health():
    health_items = [
        ("🧠 ML Risk Engine", True, "Ready"),
        ("🗺️ GIS Module", True, "Ready"),
        ("🌦️ Weather Service", bool(os.getenv("OPENWEATHER_API_KEY")), "Configured" if os.getenv("OPENWEATHER_API_KEY") else "API key missing"),
        ("📱 Twilio SMS", bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN")), "Configured" if (os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN")) else "Credentials missing"),
        ("📧 Email Service", bool(os.getenv("EMAIL_SENDER") or os.getenv("EMAIL_ADDRESS") or os.getenv("GMAIL_USER")), "Configured" if (os.getenv("EMAIL_SENDER") or os.getenv("EMAIL_ADDRESS") or os.getenv("GMAIL_USER")) else "Check email credentials"),
        ("🧠 GenAI / Groq", bool(os.getenv("GROQ_API_KEY")), "Configured" if os.getenv("GROQ_API_KEY") else "API key missing"),
    ]
    cols = st.columns(3)
    for index, item in enumerate(health_items):
        name, ok, status = item
        with cols[index % 3]:
            if ok:
                st.success(f"{name}\n\n{status}")
            else:
                st.warning(f"{name}\n\n{status}")


def get_risk_driver_summary(result, risk_inputs):
    drivers = []
    rainfall_24h = float(risk_inputs.get("rainfall_24h", 0))
    rainfall_7d = float(risk_inputs.get("rainfall_7d", 0))
    soil_1 = float(risk_inputs.get("soil_water_1", 0))
    soil_2 = float(risk_inputs.get("soil_water_2", 0))
    slope = float(result.get("slope_degree", 0))
    if rainfall_7d >= 300:
        drivers.append(("🌧️ Heavy 7-day rainfall", f"{rainfall_7d:.1f} mm accumulated rainfall"))
    elif rainfall_24h >= 100:
        drivers.append(("🌧️ Intense recent rainfall", f"{rainfall_24h:.1f} mm in the last 24 hours"))
    avg_soil = (soil_1 + soil_2) / 2
    if avg_soil >= 0.55:
        drivers.append(("💧 High soil saturation", f"Average soil water level: {avg_soil:.2f}"))
    if slope >= 30:
        drivers.append(("⛰️ Steep terrain", f"Slope angle: {slope:.2f}°"))
    if not drivers:
        drivers.append(("🧠 Combined environmental conditions", "Risk is based on multiple weather, terrain and land-cover features."))
    return drivers[:3]

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NER Landslide Early Warning System",
    page_icon="🏔️",
    layout="wide"
)


# ==========================================
# NER STATES
# ==========================================

NER_STATES = [
    "Assam",
    "Arunachal Pradesh",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]


STATE_DEFAULT_CITIES = {

    "Assam":
        "Haflong",

    "Arunachal Pradesh":
        "Itanagar",

    "Manipur":
        "Imphal",

    "Meghalaya":
        "Shillong",

    "Mizoram":
        "Aizawl",

    "Nagaland":
        "Kohima",

    "Sikkim":
        "Gangtok",

    "Tripura":
        "Agartala"
}


STATE_DEFAULT_COORDINATES = {

    "Assam":
        (25.1648, 93.0176),

    "Arunachal Pradesh":
        (27.0844, 93.6053),

    "Manipur":
        (24.8170, 93.9368),

    "Meghalaya":
        (25.5788, 91.8933),

    "Mizoram":
        (23.7271, 92.7176),

    "Nagaland":
        (25.6751, 94.1086),

    "Sikkim":
        (27.3314, 88.6138),

    "Tripura":
        (23.8315, 91.2868)
}


# ==========================================
# CUSTOM CSS - ANIMATED FUTURE COMMAND UI
# ==========================================

st.markdown(
    """
    <style>

    /* =========================================
       GLOBAL BACKGROUND
    ========================================= */

    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(0, 255, 170, 0.10), transparent 25%),
            radial-gradient(circle at 85% 10%, rgba(0, 150, 255, 0.10), transparent 28%),
            linear-gradient(135deg, #06101d 0%, #0b1422 45%, #07111a 100%);
        background-size: 180% 180%;
        animation: bgShift 14s ease-in-out infinite alternate;
        color: #eaf6ff;
    }

    @keyframes bgShift {
        0% {
            background-position: 0% 50%;
        }
        100% {
            background-position: 100% 50%;
        }
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        animation: fadePage 0.7s ease;
    }

    @keyframes fadePage {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: rgba(5, 14, 25, 0.72);
        backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(94, 234, 212, 0.08);
    }

    /* =========================================
       HERO
    ========================================= */

    .hero-shell {
        position: relative;
        overflow: hidden;
        padding: 30px 32px;
        margin-bottom: 24px;
        border-radius: 26px;
        border: 1px solid rgba(94, 234, 212, 0.18);
        background:
            linear-gradient(135deg, rgba(6, 78, 59, 0.28), rgba(30, 64, 175, 0.18)),
            rgba(9, 18, 31, 0.80);
        box-shadow:
            0 24px 70px rgba(0, 0, 0, 0.34),
            0 0 40px rgba(45, 212, 191, 0.05);
        backdrop-filter: blur(18px);
        animation: heroGlow 4s ease-in-out infinite alternate;
    }

    @keyframes heroGlow {
        from {
            box-shadow:
                0 24px 70px rgba(0, 0, 0, 0.34),
                0 0 26px rgba(45, 212, 191, 0.04);
        }
        to {
            box-shadow:
                0 28px 80px rgba(0, 0, 0, 0.38),
                0 0 54px rgba(45, 212, 191, 0.12);
        }
    }

    .hero-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(
                110deg,
                transparent 0%,
                transparent 35%,
                rgba(255,255,255,0.06) 50%,
                transparent 65%,
                transparent 100%
            );
        transform: translateX(-100%);
        animation: scanGlow 5.5s linear infinite;
        pointer-events: none;
    }

    @keyframes scanGlow {
        0% {
            transform: translateX(-120%);
        }
        100% {
            transform: translateX(120%);
        }
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 850;
        color: #5eead4;
        margin-bottom: 10px;
    }

    .hero-kicker::before {
        content: "";
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 16px #22c55e;
        animation: livePulse 1.5s ease-in-out infinite;
    }

    @keyframes livePulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.45;
            transform: scale(1.5);
        }
    }

    .main-title {
        margin: 0;
        font-size: clamp(30px, 4vw, 48px);
        line-height: 1.05;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.035em;
        text-shadow: 0 0 28px rgba(45, 212, 191, 0.12);
    }

    .subtitle {
        margin-top: 12px;
        max-width: 980px;
        font-size: 16px;
        line-height: 1.65;
        color: #b9cbe0;
    }

    .hero-pills {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 8px 13px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 750;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.62);
        color: #dbeafe;
        transition: all 0.25s ease;
    }

    .hero-pill:hover {
        transform: translateY(-2px);
        border-color: rgba(94, 234, 212, 0.5);
        box-shadow: 0 8px 20px rgba(45, 212, 191, 0.12);
    }

    .hero-pill.live {
        color: #bbf7d0;
        border-color: rgba(34, 197, 94, 0.30);
        background: rgba(22, 101, 52, 0.22);
    }

    /* =========================================
       TYPOGRAPHY
    ========================================= */

    h1, h2, h3 {
        color: #f8fbff !important;
        letter-spacing: -0.02em;
    }

    p, label, .stMarkdown, .stCaption {
        color: #c9d6e6;
    }

    /* =========================================
       METRIC CARDS
    ========================================= */

    div[data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(180deg, rgba(17, 31, 50, 0.96), rgba(8, 19, 33, 0.96));
        border: 1px solid rgba(94, 234, 212, 0.14);
        border-radius: 20px;
        padding: 17px 19px;
        min-height: 114px;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
        transition: all 0.28s ease;
        animation: cardRise 0.65s ease both;
    }

    div[data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -30px;
        bottom: -40px;
        border-radius: 50%;
        background: rgba(45, 212, 191, 0.07);
        filter: blur(2px);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(94, 234, 212, 0.36);
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.30);
    }

    @keyframes cardRise {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    div[data-testid="stMetricLabel"] {
        color: #90a6be !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fbff !important;
        font-weight: 900;
        text-shadow: 0 0 16px rgba(96, 165, 250, 0.10);
    }

    /* =========================================
       BUTTONS
    ========================================= */

    .stButton > button {
        position: relative;
        overflow: hidden;
        width: 100%;
        min-height: 47px;
        border-radius: 13px;
        font-weight: 850;
        color: #ffffff;
        border: 1px solid rgba(45, 212, 191, 0.36);
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 50%, #0891b2 100%);
        background-size: 180% 180%;
        box-shadow: 0 11px 28px rgba(13, 148, 136, 0.22);
        transition: all 0.20s ease;
        animation: btnGradient 5s ease infinite alternate;
    }

    @keyframes btnGradient {
        from {
            background-position: 0% 50%;
        }
        to {
            background-position: 100% 50%;
        }
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(94, 234, 212, 0.78);
        box-shadow: 0 16px 34px rgba(13, 148, 136, 0.34);
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.99);
    }

    /* =========================================
       INPUTS
    ========================================= */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: rgba(12, 22, 38, 0.86) !important;
        border-color: rgba(148, 163, 184, 0.18) !important;
        border-radius: 13px !important;
        transition: all 0.20s ease;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color: rgba(94, 234, 212, 0.55) !important;
        box-shadow: 0 0 0 2px rgba(45, 212, 191, 0.08);
    }

    input, textarea,
    div[data-baseweb="select"] span {
        color: #f8fbff !important;
    }

    /* =========================================
       SIDEBAR
    ========================================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(5, 14, 24, 0.98), rgba(7, 21, 35, 0.98));
        border-right: 1px solid rgba(94, 234, 212, 0.10);
    }

    .sidebar-brand {
        position: relative;
        overflow: hidden;
        padding: 17px 14px;
        margin-bottom: 10px;
        border-radius: 17px;
        background:
            linear-gradient(135deg, rgba(13, 148, 136, 0.24), rgba(30, 64, 175, 0.16));
        border: 1px solid rgba(94, 234, 212, 0.16);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .sidebar-brand::after {
        content: "";
        position: absolute;
        height: 2px;
        left: 8%;
        right: 8%;
        bottom: 0;
        background: linear-gradient(90deg, transparent, #2dd4bf, transparent);
        animation: linePulse 2.2s ease-in-out infinite;
    }

    @keyframes linePulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    .sidebar-brand-title {
        font-size: 17px;
        font-weight: 900;
        color: #f8fbff;
    }

    .sidebar-brand-sub {
        margin-top: 4px;
        font-size: 11px;
        color: #8ea4bd;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 9px 11px;
        border-radius: 12px;
        transition: all 0.18s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(45, 212, 191, 0.08);
        transform: translateX(3px);
    }

    /* =========================================
       ALERTS / EXPANDERS / TABLES
    ========================================= */

    div[data-testid="stAlert"] {
        border-radius: 15px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        backdrop-filter: blur(12px);
        animation: softPop 0.35s ease;
    }

    @keyframes softPop {
        from {
            opacity: 0;
            transform: scale(0.985);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    details {
        background: rgba(12, 22, 38, 0.70) !important;
        border: 1px solid rgba(94, 234, 212, 0.10) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 17px;
        overflow: hidden;
        border: 1px solid rgba(94, 234, 212, 0.10);
    }

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(45, 212, 191, 0.58),
            rgba(59, 130, 246, 0.36),
            transparent
        ) !important;
        margin: 1.5rem 0 !important;
    }

    /* =========================================
       RISK PANELS
    ========================================= */

    .risk-low,
    .risk-moderate,
    .risk-high,
    .risk-critical {
        padding: 21px;
        border-radius: 17px;
        font-weight: 900;
        text-align: center;
        font-size: 21px;
        letter-spacing: 0.035em;
        transition: all 0.25s ease;
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(22, 101, 52, 0.36), rgba(20, 83, 45, 0.24));
        color: #bbf7d0;
        border: 1px solid rgba(74, 222, 128, 0.28);
        box-shadow: 0 0 28px rgba(34, 197, 94, 0.06);
    }

    .risk-moderate {
        background: linear-gradient(135deg, rgba(161, 98, 7, 0.34), rgba(133, 77, 14, 0.24));
        color: #fde68a;
        border: 1px solid rgba(250, 204, 21, 0.28);
        box-shadow: 0 0 28px rgba(250, 204, 21, 0.06);
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(194, 65, 12, 0.36), rgba(154, 52, 18, 0.26));
        color: #fed7aa;
        border: 1px solid rgba(251, 146, 60, 0.30);
        box-shadow: 0 0 34px rgba(249, 115, 22, 0.09);
        animation: highGlow 2.2s ease-in-out infinite alternate;
    }

    @keyframes highGlow {
        from { box-shadow: 0 0 20px rgba(249, 115, 22, 0.06); }
        to { box-shadow: 0 0 42px rgba(249, 115, 22, 0.16); }
    }

    .risk-critical {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.44), rgba(127, 29, 29, 0.30));
        color: #fecaca;
        border: 1px solid rgba(248, 113, 113, 0.34);
        animation: criticalPulse 1.5s ease-in-out infinite;
    }

    @keyframes criticalPulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 20px rgba(248, 113, 113, 0.08);
        }
        50% {
            transform: scale(1.01);
            box-shadow: 0 0 50px rgba(248, 113, 113, 0.20);
        }
    }

    /* =========================================
       MOBILE
    ========================================= */

    @media (max-width: 760px) {
        .hero-shell {
            padding: 22px 18px;
            border-radius: 19px;
        }

        .main-title {
            font-size: 31px;
        }

        .subtitle {
            font-size: 14px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ==========================================
# EXTRA FIRST-IMPRESSION LANDING UI
# ==========================================

st.markdown(
    """
    <style>

    .mission-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 10px 0 24px 0;
    }

    .mission-card {
        position: relative;
        overflow: hidden;
        min-height: 108px;
        padding: 16px 18px;
        border-radius: 18px;
        border: 1px solid rgba(94, 234, 212, 0.13);
        background:
            linear-gradient(180deg, rgba(15, 28, 47, 0.96), rgba(8, 18, 31, 0.96));
        box-shadow: 0 14px 34px rgba(0,0,0,0.22);
        transition: all 0.28s ease;
        animation: missionEnter 0.7s ease both;
    }

    .mission-card:hover {
        transform: translateY(-5px);
        border-color: rgba(94, 234, 212, 0.42);
        box-shadow: 0 20px 45px rgba(0,0,0,0.32);
    }

    .mission-card::after {
        content: "";
        position: absolute;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        right: -45px;
        bottom: -55px;
        background: rgba(45, 212, 191, 0.06);
    }

    @keyframes missionEnter {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .mission-icon {
        font-size: 23px;
        margin-bottom: 9px;
    }

    .mission-label {
        color: #8fa8c1;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .mission-value {
        color: #ffffff;
        font-size: 25px;
        line-height: 1.1;
        font-weight: 900;
        margin-top: 4px;
    }

    .section-eyebrow {
        margin-top: 6px;
        margin-bottom: 5px;
        color: #5eead4;
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 900;
    }

    .section-title-pro {
        margin-bottom: 6px;
        color: #ffffff;
        font-size: 27px;
        font-weight: 900;
        letter-spacing: -0.025em;
    }

    .section-copy {
        margin-bottom: 18px;
        color: #9eb2c9;
        font-size: 14px;
        line-height: 1.65;
    }

    .capability-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 12px;
        margin-bottom: 24px;
    }

    .capability-card {
        position: relative;
        overflow: hidden;
        min-height: 150px;
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.13);
        background:
            linear-gradient(145deg, rgba(13, 26, 44, 0.96), rgba(8, 17, 30, 0.96));
        transition: all 0.28s ease;
    }

    .capability-card:hover {
        transform: translateY(-4px);
        border-color: rgba(96, 165, 250, 0.34);
        box-shadow: 0 18px 42px rgba(0,0,0,0.28);
    }

    .capability-card .cap-icon {
        font-size: 27px;
        margin-bottom: 10px;
    }

    .capability-card .cap-title {
        color: #f8fbff;
        font-size: 16px;
        font-weight: 900;
        margin-bottom: 7px;
    }

    .capability-card .cap-copy {
        color: #9fb2c8;
        font-size: 12.5px;
        line-height: 1.55;
    }

    .pipeline-shell {
        position: relative;
        overflow: hidden;
        margin: 12px 0 24px 0;
        padding: 22px;
        border-radius: 22px;
        border: 1px solid rgba(94, 234, 212, 0.12);
        background:
            linear-gradient(135deg, rgba(5, 46, 52, 0.22), rgba(23, 37, 84, 0.16)),
            rgba(9, 18, 31, 0.88);
    }

    .pipeline-shell::before {
        content: "";
        position: absolute;
        top: 0;
        bottom: 0;
        width: 120px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(94,234,212,0.06),
            transparent
        );
        animation: pipelineScan 6s linear infinite;
    }

    @keyframes pipelineScan {
        from { left: -140px; }
        to { left: calc(100% + 140px); }
    }

    .pipeline-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        flex-wrap: wrap;
    }

    .pipe-node {
        flex: 1;
        min-width: 130px;
        padding: 13px 12px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(148,163,184,0.14);
        background: rgba(15, 23, 42, 0.72);
        color: #e8f3ff;
        font-size: 12px;
        font-weight: 800;
        z-index: 1;
    }

    .pipe-arrow {
        color: #5eead4;
        font-size: 18px;
        font-weight: 900;
        opacity: 0.85;
        z-index: 1;
    }

    .state-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 10px;
    }

    .state-chip {
        padding: 8px 12px;
        border-radius: 999px;
        color: #dcecff;
        font-size: 12px;
        font-weight: 750;
        border: 1px solid rgba(96,165,250,0.18);
        background: rgba(30,64,175,0.12);
        transition: all 0.2s ease;
    }

    .state-chip:hover {
        transform: translateY(-2px);
        color: #ffffff;
        border-color: rgba(94,234,212,0.42);
        background: rgba(13,148,136,0.16);
    }

    .live-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 20px;
        padding: 13px 16px;
        border-radius: 15px;
        border: 1px solid rgba(34,197,94,0.18);
        background: rgba(20,83,45,0.15);
    }

    .live-banner-left {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #d7ffe6;
        font-size: 13px;
        font-weight: 800;
    }

    .live-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 15px #22c55e;
        animation: livePulse 1.5s ease-in-out infinite;
    }

    .live-banner-right {
        color: #9fc1ad;
        font-size: 12px;
    }

    @media (max-width: 1000px) {
        .mission-strip {
            grid-template-columns: repeat(2, 1fr);
        }

        .capability-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 620px) {
        .mission-strip {
            grid-template-columns: 1fr;
        }

        .pipe-arrow {
            display: none;
        }

        .pipe-node {
            min-width: 100%;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SESSION STATE
# ==========================================

if "risk_result" not in st.session_state:

    st.session_state.risk_result = None


if "risk_inputs" not in st.session_state:

    st.session_state.risk_inputs = None


if "risk_state" not in st.session_state:

    st.session_state.risk_state = None


if "last_auto_alert_key" not in st.session_state:

    st.session_state.last_auto_alert_key = None


if "last_auto_alert_status" not in st.session_state:

    st.session_state.last_auto_alert_status = None


if "genai_messages" not in st.session_state:

    st.session_state.genai_messages = []


if "live_weather_result" not in st.session_state:

    st.session_state.live_weather_result = None


if "live_weather_updated_at" not in st.session_state:

    st.session_state.live_weather_updated_at = None


# ==========================================
# FIXED EMERGENCY CONTACTS
# ==========================================

EMERGENCY_EMAIL = "uiwizards2026@gmail.com"
EMERGENCY_PHONE = "+919638636364"


# ==========================================
# HERO HEADER
# ==========================================

st.caption("🟢 LIVE DISASTER INTELLIGENCE PLATFORM")

st.title(
    "🏔️ NER Landslide Early Warning Command Center"
)

st.write(
    "AI-powered risk prediction, GIS monitoring, road connectivity, "
    "emergency prioritisation, citizen reporting and automated alerts "
    "for the North Eastern Region of India."
)

hero_col1, hero_col2, hero_col3, hero_col4 = st.columns(4)

with hero_col1:
    st.success("🟢 System Operational")

with hero_col2:
    st.info("🗺️ 8 NER States")

with hero_col3:
    st.info("🧠 AI + GIS + Weather")

with hero_col4:
    st.info("🚨 Email + SMS Alerts")


# ==========================================
# LOAD DATA
# ==========================================

DATA_FILE = (
    "Data/Processed/"
    "ner_landslide_training.csv"
)


@st.cache_data
def load_data():

    if os.path.exists(DATA_FILE):

        return pd.read_csv(
            DATA_FILE
        )

    return None


df = load_data()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">
            🛰️ NER Command Center
        </div>
        <div class="sidebar-brand-sub">
            Landslide Intelligence & Response
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Mission Control",
    [
        "🏠 Overview",
        "🤖 Live Risk Prediction",
        "🗺️ GIS Risk Map",
        "🌦️ Live Weather",
        "📸 Citizen Reporting",
        "🧠 GenAI Assistant",
        "📊 Historical Analytics"
    ]
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "SYSTEM SCOPE"
)

st.sidebar.write(
    "🌏 **Region:** North Eastern India"
)

st.sidebar.write(
    "🗺️ **Coverage:** 8 States"
)

st.sidebar.write(
    "🚨 **Alert Mode:** Email + SMS"
)

st.sidebar.caption(
    "AI-Based Landslide Early Warning & Risk Monitoring"
)


# ==========================================

# ==========================================
# LOW-NETWORK / OFFLINE STATUS
# ==========================================

offline_status = get_offline_status()

with st.sidebar.expander(
    "📶 Network & Offline Queue",
    expanded=False
):
    if offline_status["online"]:
        st.success("Online")
    else:
        st.warning("Offline / low-network mode")

    st.write(
        f"Pending reports: "
        f"{offline_status['pending_reports']}"
    )

    st.write(
        f"Pending alerts: "
        f"{offline_status['pending_alerts']}"
    )



# ==========================================
# SYSTEM HEALTH
# ==========================================

with st.sidebar.expander("🩺 System Health", expanded=False):
    st.caption("Quick status of key project services")
    if os.getenv("OPENWEATHER_API_KEY"):
        st.success("🌦️ Weather API: Configured")
    else:
        st.warning("🌦️ Weather API: Check key")
    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        st.success("📱 Twilio SMS: Configured")
    else:
        st.warning("📱 Twilio SMS: Check credentials")
    if os.getenv("GROQ_API_KEY"):
        st.success("🧠 Groq GenAI: Configured")
    else:
        st.warning("🧠 Groq GenAI: Check key")
    st.success("🧠 ML Engine: Ready")
    st.success("🗺️ GIS Module: Ready")

# ==========================================
# AUTOMATIC EMERGENCY ALERT SETTINGS
# ==========================================

with st.sidebar.expander(
    "🚨 Automatic Alerts",
    expanded=False
):

    # Automatic emergency alerts are permanently enabled.
    auto_alerts_enabled = True

    st.success(
        "🟢 Automatic HIGH / CRITICAL Alerts: ALWAYS ON"
    )

    auto_receiver_email = EMERGENCY_EMAIL
    auto_receiver_phone = EMERGENCY_PHONE

    st.write(
        f"📧 **Fixed Email:** {EMERGENCY_EMAIL}"
    )

    st.write(
        f"📱 **Fixed Mobile:** {EMERGENCY_PHONE}"
    )

    st.caption(
        "Every new HIGH or CRITICAL "
        "prediction automatically sends Email + Twilio SMS once "
        "to these fixed emergency contacts. "
        "Duplicate alerts for the same prediction "
        "are blocked."
    )


# PAGE 1 - OVERVIEW
# ==========================================

if page == "🏠 Overview":

    # ======================================
    # LIVE STATUS BANNER
    # ======================================

    network_label = (
        "ONLINE"
        if offline_status["online"]
        else
        "LOW-NETWORK / OFFLINE"
    )

    network_note = (
        "Live APIs and alert delivery available"
        if offline_status["online"]
        else
        "Store-and-forward queue is active"
    )

    st.markdown(
        f"""
        <div class="live-banner">
            <div class="live-banner-left">
                <span class="live-dot"></span>
                NER DISASTER INTELLIGENCE SYSTEM • {network_label}
            </div>
            <div class="live-banner-right">
                {network_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ======================================
    # MISSION KPIs
    # ======================================

    if df is not None:
        total_records = len(df)
        total_landslides = int(df["landslide_occurred"].sum())
        control_samples = total_records - total_landslides
        states = df["ner_state"].nunique()
    else:
        total_records = 0
        total_landslides = 0
        control_samples = 0
        states = 8

    st.subheader("📊 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🛰️ Regional Coverage", f"{states} NER States")

    with col2:
        st.metric("🏔️ Historical Events", total_landslides)

    with col3:
        st.metric("🧠 AI Training Samples", total_records)

    with col4:
        st.metric("🚨 Emergency Alerts", "Always ON")


    st.markdown("---")

    st.markdown("## 🩺 System Health")

    render_system_health()

    st.info(
        "💡 Demo Tip: For the strongest hackathon demo, "
        "generate a HIGH / CRITICAL prediction and then show "
        "road status, emergency priority, automatic Email/SMS "
        "and GenAI explanation."
    )


    # ======================================
    # WHAT THIS PLATFORM DOES
    # ======================================

    st.markdown("## 🛰️ System Capabilities")

    st.markdown(
        """
### 🤖 AI Risk Prediction
XGBoost-based landslide susceptibility scoring using rainfall, soil moisture,
slope, elevation, pressure and land-cover features.

### 🗺️ GIS Intelligence
Interactive regional mapping for vulnerable locations, state-wise monitoring
and road connectivity awareness.

### 🚨 Automatic Early Warning
HIGH and CRITICAL risk predictions automatically trigger emergency Email and
SMS alerts with duplicate protection.

### 📸 Citizen & Field Reporting
Geo-tagged field reports with photo/video evidence help authorities understand
ground conditions faster.

### 👁️ Computer Vision Support
Uploaded field images are analysed for visual hazard cues, exposed people or
vehicles and field-review severity.

### 📶 Low-Network Resilience
Alerts can be stored locally in an offline queue when connectivity is
unavailable and processed after recovery.
        """
    )


    # ======================================
    # DATA-TO-ACTION PIPELINE
    # ======================================

    st.markdown("## 🔄 Intelligence Pipeline")

    st.info(
        "🌧️ Weather & Rainfall  →  "
        "⛰️ DEM & Slope  →  "
        "🌳 Land Cover  →  "
        "🧠 XGBoost Risk Engine  →  "
        "🗺️ GIS & Road Status  →  "
        "🚨 Email & SMS Alert"
    )


    # ======================================
    # HISTORICAL ANALYTICS
    # ======================================

    if df is not None:

        positive_df = df[
            df[
                "landslide_occurred"
            ]
            ==
            1
        ]

        state_counts = (
            positive_df[
                "ner_state"
            ]
            .value_counts()
            .reindex(
                NER_STATES,
                fill_value=0
            )
        )

        st.markdown(
            """
            <div class="section-eyebrow">
                Regional Intelligence
            </div>

            <div class="section-title-pro">
                Historical landslide distribution
            </div>

            <div class="section-copy">
                Historical event distribution across the eight North Eastern states.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.bar_chart(
            state_counts
        )

    else:

        st.warning(
            "Training dataset not found."
        )


    # ======================================
    # NER COVERAGE
    # ======================================

    state_chips = "".join(
        [
            f'<span class="state-chip">{state}</span>'
            for state in NER_STATES
        ]
    )

    st.markdown(
        f"""
        <div class="section-eyebrow">
            Coverage
        </div>

        <div class="section-title-pro">
            North Eastern Region
        </div>

        <div class="state-cloud">
            {state_chips}
        </div>
        """,
        unsafe_allow_html=True
    )


# PAGE 2 - LIVE RISK PREDICTION
# ==========================================

elif page == "🤖 Live Risk Prediction":

    st.header(
        "🤖 NER Landslide Risk Prediction"
    )


    st.info(
        "Select a North Eastern state and location, "
        "then enter recent rainfall and soil conditions."
    )

    # ==========================================
    # STATE
    # ==========================================

    selected_state = st.selectbox(
        "🌏 Select NER State",
        NER_STATES
    )


    default_city = (
        STATE_DEFAULT_CITIES[
            selected_state
        ]
    )


    city = st.text_input(
        "📍 City / Location",
        value=default_city
    )


    # ==========================================
    # RAINFALL
    # ==========================================

    st.subheader(
        "🌧️ Recent Rainfall"
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        rainfall_24h = st.number_input(
            "Last 24 Hours (mm)",
            min_value=0.0,
            value=80.0,
            step=5.0
        )


    with col2:

        rainfall_3d = st.number_input(
            "Last 3 Days (mm)",
            min_value=0.0,
            value=180.0,
            step=5.0
        )


    with col3:

        rainfall_7d = st.number_input(
            "Last 7 Days (mm)",
            min_value=0.0,
            value=350.0,
            step=10.0
        )


    # ==========================================
    # SOIL
    # ==========================================

    st.subheader(
        "💧 Soil Conditions"
    )


    col4, col5 = (
        st.columns(2)
    )


    with col4:

        soil_water_1 = st.slider(
            "Soil Water Layer 1",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01
        )


    with col5:

        soil_water_2 = st.slider(
            "Soil Water Layer 2",
            min_value=0.0,
            max_value=1.0,
            value=0.38,
            step=0.01
        )


    # ==========================================
    # CALCULATE
    # ==========================================

    if st.button(
        "🚨 Calculate Landslide Risk",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner(
            "Analysing weather, terrain "
            "and land-cover conditions..."
        ):

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


        if not result[
            "success"
        ]:

            st.error(
                result["message"]
            )

            st.session_state.risk_result = None


        else:

            st.session_state.risk_result = (
                result
            )


            st.session_state.risk_state = (
                selected_state
            )


            st.session_state.risk_inputs = {

                "rainfall_24h":
                    rainfall_24h,

                "rainfall_3d":
                    rainfall_3d,

                "rainfall_7d":
                    rainfall_7d,

                "soil_water_1":
                    soil_water_1,

                "soil_water_2":
                    soil_water_2
            }


    # ==========================================
    # RESULT
    # ==========================================

    if st.session_state.risk_result is not None:

        result = (
            st.session_state.risk_result
        )


        risk_inputs = (
            st.session_state.risk_inputs
        )


        risk_state = (
            st.session_state.risk_state
        )


        risk_score = (
            result[
                "risk_score"
            ]
        )


        risk_level = (
            result[
                "risk_level"
            ]
        )


        st.markdown("---")


        st.success(
            "✅ Risk analysis completed"
        )


        st.subheader(
            "🚨 Risk Assessment"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.2f}%"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        with col3:

            st.metric(
                "State",
                risk_state
            )


        css_class = {

            "LOW":
                "risk-low",

            "MODERATE":
                "risk-moderate",

            "HIGH":
                "risk-high",

            "CRITICAL":
                "risk-critical"

        }.get(
            risk_level,
            "risk-moderate"
        )


        st.markdown(
            f"""
            <div class="{css_class}">
                {risk_level} LANDSLIDE RISK
                <br>
                Risk Score: {risk_score:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Prediction generated: {get_current_timestamp()}"
        )

        if risk_level in ["HIGH", "CRITICAL"]:

            st.error(
                "🚨 EMERGENCY RESPONSE ACTIVATED — "
                f"{risk_level} landslide risk detected for "
                f"{result['city']}, {risk_state}"
            )

        st.subheader("🔎 Top Risk Drivers")

        risk_drivers = get_risk_driver_summary(result, risk_inputs)

        for driver_title, driver_detail in risk_drivers:
            st.write(f"**{driver_title}** — {driver_detail}")

        st.caption(
            "These are quick explainability cues from the current input values. "
            "The final risk score is generated by the ML model using multiple features."
        )


        # ======================================
        # LOCATION
        # ======================================

        st.subheader(
            "📍 Location Information"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Latitude",
                round(
                    result[
                        "latitude"
                    ],
                    4
                )
            )


        with col2:

            st.metric(
                "Longitude",
                round(
                    result[
                        "longitude"
                    ],
                    4
                )
            )


        with col3:

            st.metric(
                "Elevation",
                f"{result['elevation_m']:.0f} m"
            )


        # ======================================
        # TERRAIN
        # ======================================

        st.subheader(
            "⛰️ Terrain Conditions"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Slope",
                f"{result['slope_degree']:.2f}°"
            )


        with col2:

            st.metric(
                "Aspect",
                f"{result['aspect_degree']:.2f}°"
            )


        with col3:

            st.metric(
                "Land Cover",
                result[
                    "landcover_class"
                ]
            )


        # ======================================
        # WEATHER
        # ======================================

        st.subheader(
            "🌦️ Current Weather"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Temperature",
                f"{result['temperature_c']:.1f} °C"
            )


        with col2:

            st.metric(
                "Humidity",
                f"{result['humidity']}%"
            )


        with col3:

            st.metric(
                "Pressure",
                f"{result['pressure_hpa']} hPa"
            )


        st.write(
            "**Weather Condition:**",
            result["weather"]
        )


        # ======================================
        # RAINFALL
        # ======================================

        st.subheader(
            "🌧️ Rainfall Used by Model"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "24 Hour Rainfall",
                f"{risk_inputs['rainfall_24h']:.1f} mm"
            )


        with col2:

            st.metric(
                "3 Day Rainfall",
                f"{risk_inputs['rainfall_3d']:.1f} mm"
            )


        with col3:

            st.metric(
                "7 Day Rainfall",
                f"{risk_inputs['rainfall_7d']:.1f} mm"
            )


        # ======================================
        # RECOMMENDATION
        # ======================================

        st.subheader(
            "⚠️ Recommended Action"
        )


        if risk_level == "CRITICAL":

            st.error(
                result[
                    "recommendation"
                ]
            )


        elif risk_level == "HIGH":

            st.warning(
                result[
                    "recommendation"
                ]
            )


        else:

            st.info(
                result[
                    "recommendation"
                ]
            )


        # ======================================
        # ROAD CONNECTIVITY
        # ======================================

        st.markdown("---")

        st.subheader(
            "🛣️ Road Connectivity Status"
        )

        road_result = assess_road_connectivity(

            location=(
                f"{result['city']}, "
                f"{risk_state}"
            ),

            risk_score=
                result[
                    "risk_score"
                ],

            risk_level=
                result[
                    "risk_level"
                ]
        )


        col1, col2 = st.columns(
            2
        )


        with col1:

            st.metric(
                "Road Status",
                road_result[
                    "road_status"
                ]
            )


        with col2:

            st.metric(
                "Response Priority",
                road_result[
                    "response_priority"
                ]
            )


        if road_result[
            "road_status"
        ] == "BLOCKED":

            st.error(
                road_result[
                    "message"
                ]
            )


        elif road_result[
            "road_status"
        ] == "PARTIALLY BLOCKED":

            st.warning(
                road_result[
                    "message"
                ]
            )


        elif road_result[
            "road_status"
        ] == "AT RISK":

            st.warning(
                road_result[
                    "message"
                ]
            )


        else:

            st.success(
                road_result[
                    "message"
                ]
            )


        # ======================================
        # EMERGENCY PRIORITISATION
        # ======================================

        st.markdown("---")

        st.subheader(
            "🚑 Emergency Response Prioritisation"
        )

        emergency_result = calculate_emergency_priority(

            risk_score=
                result[
                    "risk_score"
                ],

            risk_level=
                result[
                    "risk_level"
                ],

            road_status=
                road_result[
                    "road_status"
                ],

            rainfall_24h_mm=
                risk_inputs[
                    "rainfall_24h"
                ],

            rainfall_7d_mm=
                risk_inputs[
                    "rainfall_7d"
                ],

            slope_degree=
                result[
                    "slope_degree"
                ],

            visual_severity=
                None
        )


        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.metric(
                "Priority Score",
                f"{emergency_result['priority_score']}/100"
            )


        with col2:

            st.metric(
                "Emergency Priority",
                emergency_result[
                    "priority_level"
                ]
            )


        with col3:

            st.metric(
                "Target Response Time",
                emergency_result[
                    "response_time"
                ]
            )


        if emergency_result[
            "priority_level"
        ].startswith(
            "P1"
        ):

            st.error(
                "🚨 "
                +
                emergency_result[
                    "recommended_action"
                ]
            )


        elif emergency_result[
            "priority_level"
        ].startswith(
            "P2"
        ):

            st.warning(
                "⚠️ "
                +
                emergency_result[
                    "recommended_action"
                ]
            )


        elif emergency_result[
            "priority_level"
        ].startswith(
            "P3"
        ):

            st.info(
                emergency_result[
                    "recommended_action"
                ]
            )


        else:

            st.success(
                emergency_result[
                    "recommended_action"
                ]
            )



        # ======================================
        # AUTOMATIC EMAIL + SMS ALERT
        # ======================================

        if (
            auto_alerts_enabled
            and
            risk_level in [
                "HIGH",
                "CRITICAL"
            ]
        ):

            alert_location = (
                f"{result['city']}, "
                f"{risk_state}"
            )

            auto_alert_key = (
                f"{risk_state}|"
                f"{result['city']}|"
                f"{risk_level}|"
                f"{result['risk_score']:.2f}|"
                f"{risk_inputs['rainfall_24h']:.2f}|"
                f"{risk_inputs['rainfall_7d']:.2f}|"
                f"{road_result['road_status']}|"
                f"{emergency_result['priority_level']}"
            )

            if (
                st.session_state.last_auto_alert_key
                !=
                auto_alert_key
            ):

                email_status = {
                    "success": False,
                    "message": "Email not attempted."
                }

                sms_status = {
                    "success": False,
                    "message": "SMS not attempted."
                }

                if not offline_status["online"]:

                    queue_alert(
                        {
                            "channel": "EMAIL",
                            "receiver": auto_receiver_email,
                            "location": alert_location,
                            "risk_score": result["risk_score"],
                            "risk_level": result["risk_level"],
                            "road_status": road_result["road_status"],
                            "priority": emergency_result["priority_level"],
                            "status": "PENDING_NETWORK"
                        }
                    )

                    queue_alert(
                        {
                            "channel": "SMS",
                            "receiver": auto_receiver_phone,
                            "location": alert_location,
                            "risk_score": result["risk_score"],
                            "risk_level": result["risk_level"],
                            "road_status": road_result["road_status"],
                            "priority": emergency_result["priority_level"],
                            "status": "PENDING_NETWORK"
                        }
                    )

                    st.session_state.last_auto_alert_key = (
                        auto_alert_key
                    )

                    st.warning(
                        "📴 HIGH / CRITICAL risk detected, "
                        "but network is unavailable. "
                        "Email and SMS alerts were saved "
                        "to the offline queue."
                    )

                else:

                    if auto_receiver_email.strip():

                        email_status = send_email_alert(
                            receiver_email=auto_receiver_email.strip(),
                            location=alert_location,
                            risk_score=result["risk_score"],
                            risk_level=result["risk_level"],
                            rainfall_24h=risk_inputs["rainfall_24h"],
                            rainfall_3d=risk_inputs["rainfall_3d"],
                            rainfall_7d=risk_inputs["rainfall_7d"],
                            temperature=result["temperature_c"],
                            slope=result["slope_degree"],
                            elevation=result["elevation_m"],
                            recommendation=result["recommendation"]
                        )

                    else:

                        email_status = {
                            "success": False,
                            "message": "Authority email is empty."
                        }

                    if auto_receiver_phone.strip():

                        if not auto_receiver_phone.startswith("+"):

                            sms_status = {
                                "success": False,
                                "message": (
                                    "Mobile number must use international "
                                    "format, for example +919876543210."
                                )
                            }

                        else:

                            sms_status = send_sms_alert(
                                receiver_number=auto_receiver_phone.strip()
                            )

                    else:

                        sms_status = {
                            "success": False,
                            "message": "Authority mobile number is empty."
                        }

                    st.session_state.last_auto_alert_key = (
                        auto_alert_key
                    )

                    st.session_state.last_auto_alert_status = {
                        "email": email_status,
                        "sms": sms_status
                    }

                    if (
                        email_status["success"]
                        and
                        sms_status["success"]
                    ):

                        st.success(
                            "🚨 Automatic HIGH / CRITICAL alert sent: "
                            "Email ✅  SMS ✅"
                        )

                    elif (
                        email_status["success"]
                        or
                        sms_status["success"]
                    ):

                        st.warning(
                            "⚠️ Automatic alert was only "
                            "partially successful."
                        )

                        st.write(
                            "**Email:**",
                            (
                                "✅ Sent"
                                if email_status["success"]
                                else f"❌ {email_status['message']}"
                            )
                        )

                        st.write(
                            "**SMS:**",
                            (
                                "✅ Sent"
                                if sms_status["success"]
                                else f"❌ {sms_status['message']}"
                            )
                        )

                    else:

                        st.error(
                            "❌ Automatic Email and SMS alerts failed."
                        )

                        st.write(
                            "**Email:**",
                            email_status["message"]
                        )

                        st.write(
                            "**SMS:**",
                            sms_status["message"]
                        )

            else:

                st.info(
                    "🔁 Automatic alert already sent for "
                    "this exact risk prediction. "
                    "Duplicate alert blocked."
                )

        elif (
            auto_alerts_enabled
            and
            risk_level not in [
                "HIGH",
                "CRITICAL"
            ]
        ):

            st.info(
                "Automatic alerts are enabled, "
                "but current risk is below HIGH."
            )


        # ======================================
        # EMERGENCY COMMAND MODE
        # ======================================

        if risk_level in [
            "HIGH",
            "CRITICAL"
        ]:

            st.markdown("---")

            st.header(
                "🚨 Emergency Command Mode"
            )

            st.caption(
                "Single-screen operational summary for "
                "rapid authority decision-making."
            )


            # ==================================
            # INCIDENT SUMMARY
            # ==================================

            command_col1, command_col2 = (
                st.columns(
                    [2, 1]
                )
            )


            with command_col1:

                st.error(
                    f"🚨 {risk_level} INCIDENT — "
                    f"{result['city']}, {risk_state}"
                )

                st.write(
                    "**Recommended Action:**",
                    emergency_result[
                        "recommended_action"
                    ]
                )


            with command_col2:

                st.metric(
                    "Risk Score",
                    f"{risk_score:.2f}%"
                )

                st.metric(
                    "Priority",
                    emergency_result[
                        "priority_level"
                    ]
                )


            # ==================================
            # KEY OPERATIONAL STATUS
            # ==================================

            st.subheader(
                "📍 Operational Status"
            )

            status_col1, status_col2, (
                status_col3
            ), status_col4 = st.columns(4)


            with status_col1:

                st.metric(
                    "Location",
                    result[
                        "city"
                    ]
                )


            with status_col2:

                st.metric(
                    "Road Status",
                    road_result[
                        "road_status"
                    ]
                )


            with status_col3:

                st.metric(
                    "Response Target",
                    emergency_result[
                        "response_time"
                    ]
                )


            with status_col4:

                st.metric(
                    "Slope",
                    f"{result['slope_degree']:.1f}°"
                )


            # ==================================
            # TOP RISK DRIVERS
            # ==================================

            st.subheader(
                "🔎 Priority Risk Drivers"
            )

            command_drivers = (
                get_risk_driver_summary(
                    result,
                    risk_inputs
                )
            )

            for (
                driver_title,
                driver_detail
            ) in command_drivers:

                st.write(
                    f"• **{driver_title}** — "
                    f"{driver_detail}"
                )


            # ==================================
            # AUTOMATED RESPONSE STATUS
            # ==================================

            st.subheader(
                "⚡ Automated Response"
            )


            if not offline_status[
                "online"
            ]:

                email_command_status = (
                    "🟡 Queued — waiting for network"
                )

                sms_command_status = (
                    "🟡 Queued — waiting for network"
                )

            else:

                latest_alert_status = (
                    st.session_state
                    .last_auto_alert_status
                )

                if (
                    latest_alert_status
                    and
                    st.session_state
                    .last_auto_alert_key
                    ==
                    auto_alert_key
                ):

                    email_command_status = (
                        "✅ Sent"
                        if
                        latest_alert_status[
                            "email"
                        ][
                            "success"
                        ]
                        else
                        "❌ Failed"
                    )

                    sms_command_status = (
                        "✅ Sent"
                        if
                        latest_alert_status[
                            "sms"
                        ][
                            "success"
                        ]
                        else
                        "❌ Failed"
                    )

                else:

                    email_command_status = (
                        "🟢 Alert workflow processed"
                    )

                    sms_command_status = (
                        "🟢 Alert workflow processed"
                    )


            response_col1, response_col2 = (
                st.columns(2)
            )


            with response_col1:

                st.write(
                    "📧 **Emergency Email:** "
                    f"{email_command_status}"
                )

                st.write(
                    "🗺️ **GIS Location:** "
                    "✅ Coordinates identified"
                )


            with response_col2:

                st.write(
                    "📱 **Emergency SMS:** "
                    f"{sms_command_status}"
                )

                st.write(
                    "🔁 **Duplicate Protection:** "
                    "✅ Active"
                )


            # ==================================
            # FIELD RESPONSE GUIDANCE
            # ==================================

            st.subheader(
                "🚑 Immediate Response Guidance"
            )

            if emergency_result[
                "priority_level"
            ].startswith(
                "P1"
            ):

                st.error(
                    emergency_result[
                        "recommended_action"
                    ]
                )

            elif emergency_result[
                "priority_level"
            ].startswith(
                "P2"
            ):

                st.warning(
                    emergency_result[
                        "recommended_action"
                    ]
                )

            else:

                st.info(
                    emergency_result[
                        "recommended_action"
                    ]
                )


            st.caption(
                "This command view summarizes the existing "
                "ML prediction, road assessment, emergency "
                "priority and alert workflow. It does not "
                "create a separate prediction."
            )


        # ======================================
        # MULTILINGUAL ALERT PREVIEW
        # ======================================

        st.markdown("---")

        st.subheader(
            "🌐 Multilingual Early Warning"
        )

        alert_language = st.selectbox(
            "Select Alert Language",
            LANGUAGES,
            key="alert_language"
        )


        multilingual_alert = generate_multilingual_alert(

            language=
                alert_language,

            location=(
                f"{result['city']}, "
                f"{risk_state}"
            ),

            risk_score=
                result[
                    "risk_score"
                ],

            risk_level=
                result[
                    "risk_level"
                ],

            rainfall_24h=
                risk_inputs[
                    "rainfall_24h"
                ],

            rainfall_7d=
                risk_inputs[
                    "rainfall_7d"
                ],

            road_status=
                road_result[
                    "road_status"
                ],

            priority_level=
                emergency_result[
                    "priority_level"
                ],

            response_time=
                emergency_result[
                    "response_time"
                ]
        )


        st.write(
            "**Alert Subject:**"
        )

        st.info(
            multilingual_alert[
                "subject"
            ]
        )


        st.write(
            "**Alert Message Preview:**"
        )

        st.text_area(
            "Generated Multilingual Alert",
            value=
                multilingual_alert[
                    "message"
                ],
            height=280,
            key="multilingual_alert_preview"
        )


        # ======================================
        # OFFLINE ALERT QUEUE
        # ======================================

        if not offline_status["online"]:

            st.warning(
                "📴 Network is unavailable. "
                "You can save this warning locally "
                "and send it when connectivity returns."
            )

            if st.button(
                "💾 Save Alert to Offline Queue",
                use_container_width=True,
                key="save_alert_offline"
            ):

                queued_alert = queue_alert(
                    {
                        "subject":
                            multilingual_alert[
                                "subject"
                            ],

                        "message":
                            multilingual_alert[
                                "message"
                            ],

                        "language":
                            multilingual_alert[
                                "language"
                            ],

                        "location":
                            (
                                f"{result['city']}, "
                                f"{risk_state}"
                            ),

                        "risk_score":
                            result[
                                "risk_score"
                            ],

                        "risk_level":
                            result[
                                "risk_level"
                            ],

                        "road_status":
                            road_result[
                                "road_status"
                            ],

                        "priority":
                            emergency_result[
                                "priority_level"
                            ]
                    }
                )

                st.success(
                    "✅ Alert saved locally. "
                    f"Queue ID: {queued_alert['queue_id']}"
                )


        # ======================================
        # EMAIL ALERT
        # ======================================

        st.markdown("---")


        st.subheader(
            "📧 Emergency Email Alert"
        )


        receiver_email = st.text_input(
            "Authority / Receiver Email",
            placeholder="authority@example.com",
            key="receiver_email"
        )


        if risk_level in [
            "HIGH",
            "CRITICAL"
        ]:

            st.warning(
                f"⚠️ {risk_level} risk detected "
                f"in {risk_state}."
            )


            if st.button(
                "📧 Send Emergency Email Alert",
                use_container_width=True,
                key="send_email_button"
            ):

                if not receiver_email.strip():

                    st.warning(
                        "Please enter receiver email."
                    )


                else:

                    with st.spinner(
                        "Sending emergency email..."
                    ):

                        email_result = (
                            send_email_alert(

                                receiver_email=
                                    receiver_email,

                                location=(
                                    f"{result['city']}, "
                                    f"{risk_state}"
                                ),

                                risk_score=
                                    result[
                                        "risk_score"
                                    ],

                                risk_level=
                                    result[
                                        "risk_level"
                                    ],

                                rainfall_24h=
                                    risk_inputs[
                                        "rainfall_24h"
                                    ],

                                rainfall_3d=
                                    risk_inputs[
                                        "rainfall_3d"
                                    ],

                                rainfall_7d=
                                    risk_inputs[
                                        "rainfall_7d"
                                    ],

                                temperature=
                                    result[
                                        "temperature_c"
                                    ],

                                slope=
                                    result[
                                        "slope_degree"
                                    ],

                                elevation=
                                    result[
                                        "elevation_m"
                                    ],

                                recommendation=
                                    result[
                                        "recommendation"
                                    ]
                            )
                        )


                    if email_result[
                        "success"
                    ]:

                        st.success(
                            "✅ Emergency email alert sent!"
                        )


                    else:

                        st.error(
                            "❌ Email alert failed"
                        )

                        st.code(
                            email_result[
                                "message"
                            ]
                        )


        else:

            st.info(
                "Email alerts are enabled "
                "for HIGH and CRITICAL risk."
            )



        # ======================================
        # SMS ALERT - TWILIO TRIAL
        # ======================================

        st.markdown("---")

        st.subheader(
            "📱 Emergency SMS Alert"
        )

        st.caption(
            "Twilio Trial mode uses the predefined "
            "'Account Alerts / Notifications' SMS template."
        )

        receiver_phone = st.text_input(
            "Verified Authority Mobile Number",
            placeholder="+919876543210",
            key="receiver_phone"
        )

        if risk_level in [
            "HIGH",
            "CRITICAL"
        ]:

            st.warning(
                f"📱 {risk_level} risk detected. "
                "A trial SMS notification can be sent "
                "to a Twilio-verified mobile number."
            )

            if st.button(
                "📱 Send Emergency SMS Alert",
                use_container_width=True,
                key="send_sms_button"
            ):

                if not receiver_phone.strip():

                    st.warning(
                        "Please enter a verified mobile number "
                        "with country code."
                    )

                elif not receiver_phone.startswith("+"):

                    st.warning(
                        "Use international format, "
                        "for example: +919876543210"
                    )

                elif not offline_status["online"]:

                    queued_sms = queue_alert(
                        {
                            "channel": "SMS",
                            "receiver": receiver_phone,
                            "location": (
                                f"{result['city']}, "
                                f"{risk_state}"
                            ),
                            "risk_score": result["risk_score"],
                            "risk_level": result["risk_level"],
                            "road_status": road_result["road_status"],
                            "priority": emergency_result["priority_level"],
                            "status": "PENDING_NETWORK"
                        }
                    )

                    st.warning(
                        "📴 Network unavailable. "
                        "SMS alert saved to the offline queue."
                    )

                    st.write(
                        "**Queue ID:**",
                        queued_sms["queue_id"]
                    )

                else:

                    with st.spinner(
                        "Sending Twilio trial SMS..."
                    ):

                        sms_result = send_sms_alert(
                            receiver_number=receiver_phone
                        )

                    if sms_result["success"]:

                        st.success(
                            "✅ Emergency SMS sent successfully!"
                        )

                        if sms_result.get("sid"):

                            st.write(
                                "**Twilio Message SID:**",
                                sms_result["sid"]
                            )

                    else:

                        st.error(
                            "❌ SMS alert failed"
                        )

                        st.code(
                            sms_result["message"]
                        )

        else:

            st.info(
                "SMS alerts are enabled only for "
                "HIGH and CRITICAL landslide risk."
            )


# ==========================================
# PAGE 3 - GIS MAP
# ==========================================

elif page == "🗺️ GIS Risk Map":

    st.header(
        "🗺️ NER Landslide Risk Map"
    )


    if df is None:

        st.error(
            "Training dataset not found."
        )


    else:

        selected_map_state = st.selectbox(
            "🌏 Filter by State",
            [
                "All NER States"
            ]
            +
            NER_STATES
        )


        if selected_map_state == (
            "All NER States"
        ):

            map_df = df.copy()


        else:

            map_df = df[
                df[
                    "ner_state"
                ]
                ==
                selected_map_state
            ].copy()


        if selected_map_state in (
            STATE_DEFAULT_COORDINATES
        ):

            map_center = (
                STATE_DEFAULT_COORDINATES[
                    selected_map_state
                ]
            )

            zoom_level = 8


        else:

            map_center = (
                26.0,
                93.0
            )

            zoom_level = 6


        risk_map = folium.Map(

            location=[
                map_center[0],
                map_center[1]
            ],

            zoom_start=
                zoom_level,

            tiles=
                "OpenStreetMap"
        )


        # ======================================
        # RISK FUNCTION
        # ======================================

        def calculate_map_risk(
            row
        ):

            score = 0


            if row[
                "rainfall_7d_mm"
            ] >= 300:

                score += 35

            elif row[
                "rainfall_7d_mm"
            ] >= 150:

                score += 25

            elif row[
                "rainfall_7d_mm"
            ] >= 75:

                score += 15


            if row[
                "slope_degree"
            ] >= 40:

                score += 30

            elif row[
                "slope_degree"
            ] >= 25:

                score += 20

            elif row[
                "slope_degree"
            ] >= 15:

                score += 10


            if row[
                "soil_water_layer_1"
            ] >= 0.40:

                score += 20

            elif row[
                "soil_water_layer_1"
            ] >= 0.25:

                score += 10


            if row[
                "elevation_m"
            ] >= 1500:

                score += 15

            elif row[
                "elevation_m"
            ] >= 700:

                score += 10


            if score < 30:

                return (
                    "LOW",
                    "green",
                    score
                )


            elif score < 60:

                return (
                    "MODERATE",
                    "orange",
                    score
                )


            elif score < 80:

                return (
                    "HIGH",
                    "darkorange",
                    score
                )


            else:

                return (
                    "CRITICAL",
                    "red",
                    score
                )


        # ======================================
        # MARKERS
        # ======================================

        for _, row in (
            map_df.iterrows()
        ):

            risk_level, color, score = (
                calculate_map_risk(
                    row
                )
            )


            popup = f"""
            <b>State:</b>
            {row['ner_state']}
            <br>

            <b>Risk:</b>
            {risk_level}
            <br>

            <b>Risk Score:</b>
            {score}/100
            <br><br>

            <b>Rainfall 24h:</b>
            {row['rainfall_24h_mm']:.1f} mm
            <br>

            <b>Rainfall 3d:</b>
            {row['rainfall_3d_mm']:.1f} mm
            <br>

            <b>Rainfall 7d:</b>
            {row['rainfall_7d_mm']:.1f} mm
            <br><br>

            <b>Slope:</b>
            {row['slope_degree']:.1f}°
            <br>

            <b>Elevation:</b>
            {row['elevation_m']:.0f} m
            """


            folium.CircleMarker(

                location=[
                    row[
                        "latitude"
                    ],
                    row[
                        "longitude"
                    ]
                ],

                radius=6,

                popup=folium.Popup(
                    popup,
                    max_width=350
                ),

                color=color,

                fill=True,

                fill_color=color,

                fill_opacity=0.75

            ).add_to(
                risk_map
            )


        st.write(
            f"Showing **{len(map_df)}** locations"
        )


        st_folium(
            risk_map,
            width=None,
            height=650
        )


        st.caption(
            "🟢 LOW   |   "
            "🟡 MODERATE   |   "
            "🟠 HIGH   |   "
            "🔴 CRITICAL"
        )


# ==========================================
# PAGE 4 - LIVE WEATHER
# ==========================================

elif page == "🌦️ Live Weather":

    st.header(
        "🌦️ Live Weather Intelligence"
    )

    st.write(
        "Search any city or location worldwide to view "
        "current weather conditions and rainfall information."
    )

    st.caption(
        "Live weather data supports situational awareness. "
        "NER landslide risk prediction remains a separate ML workflow."
    )


    # ======================================
    # LOCATION SEARCH
    # ======================================

    search_col1, search_col2, search_col3 = st.columns(
        [2, 3, 1]
    )

    with search_col1:

        weather_state = st.selectbox(
            "🌏 Select State",
            NER_STATES,
            key="weather_state"
        )

    with search_col2:

        default_weather_city = (
            STATE_DEFAULT_CITIES[
                weather_state
            ]
        )

        city_weather = st.text_input(
            "📍 City / Location",
            value=default_weather_city,
            key="weather_city_anywhere"
        )

    with search_col3:

        st.write("")

        st.write("")

        get_weather_clicked = st.button(
            "🔍 Check Weather",
            use_container_width=True,
            type="primary"
        )


    # ======================================
    # FETCH WEATHER
    # ======================================

    if get_weather_clicked:

        location_query = (
            city_weather.strip()
        )

        if not location_query:

            st.warning(
                "⚠️ Please enter a city or location."
            )

        else:

            with st.spinner(
                f"Fetching live weather for "
                f"{location_query}..."
            ):

                weather = get_current_weather(
                    location_query
                )


            if not weather.get(
                "success",
                False
            ):

                st.session_state.live_weather_result = None

                st.error(
                    "❌ Unable to fetch live weather."
                )

                st.write(
                    weather.get(
                        "message",
                        "Please check the location name, "
                        "internet connection or Weather API key."
                    )
                )

            else:

                st.session_state.live_weather_result = (
                    weather
                )

                st.session_state.live_weather_updated_at = (
                    get_current_timestamp()
                )


    # ======================================
    # DISPLAY SAVED WEATHER RESULT
    # ======================================

    if (
        st.session_state.live_weather_result
        is not None
    ):

        weather = (
            st.session_state.live_weather_result
        )

        st.markdown("---")

        st.success(
            "✅ Live weather data received successfully"
        )


        # ==================================
        # LOCATION HEADER
        # ==================================

        location_col1, location_col2 = (
            st.columns(
                [3, 1]
            )
        )

        with location_col1:

            st.subheader(
                f"📍 {weather.get('city', 'Selected Location')}, "
                f"{weather_state}"
            )

            st.caption(
                "Coordinates: "
                f"{weather.get('latitude', 0)}, "
                f"{weather.get('longitude', 0)}"
            )


        with location_col2:

            st.caption(
                "🕒 Last Updated"
            )

            st.write(
                st.session_state
                .live_weather_updated_at
                or
                get_current_timestamp()
            )


        # ==================================
        # CURRENT CONDITION
        # ==================================

        condition = str(
            weather.get(
                "weather",
                "Unknown"
            )
        )

        st.info(
            f"☁️ **Current Condition:** {condition}"
        )


        # ==================================
        # PRIMARY WEATHER METRICS
        # ==================================

        st.subheader(
            "🌤️ Current Conditions"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🌡️ Temperature",
                f"{float(weather.get('temperature_c', 0)):.1f} °C"
            )

        with col2:

            st.metric(
                "💧 Humidity",
                f"{weather.get('humidity', 0)}%"
            )

        with col3:

            st.metric(
                "🌬️ Pressure",
                f"{weather.get('surface_pressure_hpa', 0)} hPa"
            )


        col4, col5, col6 = st.columns(3)

        with col4:

            st.metric(
                "💨 Wind Speed",
                f"{float(weather.get('wind_speed', 0)):.1f} m/s"
            )

        with col5:

            st.metric(
                "🌧️ Rainfall - 1 Hour",
                f"{float(weather.get('rain_1h_mm', 0)):.1f} mm"
            )

        with col6:

            st.metric(
                "🌧️ Rainfall - 3 Hours",
                f"{float(weather.get('rain_3h_mm', 0)):.1f} mm"
            )


        # ==================================
        # RAINFALL SITUATION
        # ==================================

        rain_1h = float(
            weather.get(
                "rain_1h_mm",
                0
            )
        )

        rain_3h = float(
            weather.get(
                "rain_3h_mm",
                0
            )
        )

        st.subheader(
            "🌧️ Rainfall Situation"
        )

        if (
            rain_1h >= 25
            or
            rain_3h >= 50
        ):

            st.error(
                "🔴 Heavy rainfall detected. "
                "In landslide-prone terrain, continuous "
                "monitoring and field verification are recommended."
            )

        elif (
            rain_1h >= 10
            or
            rain_3h >= 20
        ):

            st.warning(
                "🟠 Moderate rainfall is currently being observed. "
                "Monitor rainfall accumulation and local slope conditions."
            )

        elif (
            rain_1h > 0
            or
            rain_3h > 0
        ):

            st.info(
                "🟡 Light rainfall is currently being observed."
            )

        else:

            st.success(
                "🟢 No measurable recent rainfall reported "
                "by the current weather response."
            )


        # ==================================
        # LOCATION MAP
        # ==================================

        latitude = float(
            weather.get(
                "latitude",
                0
            )
        )

        longitude = float(
            weather.get(
                "longitude",
                0
            )
        )

        if (
            latitude != 0
            or
            longitude != 0
        ):

            st.subheader(
                "🗺️ Weather Location Map"
            )

            weather_map = folium.Map(
                location=[
                    latitude,
                    longitude
                ],
                zoom_start=10,
                tiles="OpenStreetMap"
            )

            folium.Marker(
                [
                    latitude,
                    longitude
                ],
                popup=(
                    f"<b>{weather.get('city', 'Location')}</b><br>"
                    f"Temperature: "
                    f"{weather.get('temperature_c', 0)} °C<br>"
                    f"Humidity: "
                    f"{weather.get('humidity', 0)}%<br>"
                    f"Weather: "
                    f"{condition}"
                ),
                tooltip=(
                    f"Weather at "
                    f"{weather.get('city', 'location')}"
                ),
                icon=folium.Icon(
                    color="blue",
                    icon="cloud"
                )
            ).add_to(
                weather_map
            )

            st_folium(
                weather_map,
                width=None,
                height=420,
                key="live_weather_map"
            )


        # ==================================
        # WEATHER DATA SUMMARY
        # ==================================

        with st.expander(
            "📋 Weather Data Summary",
            expanded=False
        ):

            st.write(
                f"**Location:** "
                f"{weather.get('city', 'Unknown')}"
            )

            st.write(
                f"**Latitude:** "
                f"{weather.get('latitude', 'N/A')}"
            )

            st.write(
                f"**Longitude:** "
                f"{weather.get('longitude', 'N/A')}"
            )

            st.write(
                f"**Condition:** "
                f"{condition}"
            )

            st.write(
                f"**Temperature:** "
                f"{weather.get('temperature_c', 0)} °C"
            )

            st.write(
                f"**Humidity:** "
                f"{weather.get('humidity', 0)}%"
            )

            st.write(
                f"**Pressure:** "
                f"{weather.get('surface_pressure_hpa', 0)} hPa"
            )

            st.write(
                f"**Wind Speed:** "
                f"{weather.get('wind_speed', 0)} m/s"
            )

            st.write(
                f"**Rainfall (1h):** "
                f"{weather.get('rain_1h_mm', 0)} mm"
            )

            st.write(
                f"**Rainfall (3h):** "
                f"{weather.get('rain_3h_mm', 0)} mm"
            )


# ==========================================
# PAGE 5 - CITIZEN REPORTING
# ==========================================

elif page == "📸 Citizen Reporting":

    st.header(
        "📸 NER Citizen / Field Officer Reporting"
    )


    st.write(
        """
        Citizens and field officials can report
        slope cracks, landslides, road blockages,
        rockfalls, debris movement or infrastructure
        damage from any North Eastern state.
        """
    )


    # ======================================
    # STATE
    # ======================================

    report_state = st.selectbox(
        "🌏 State",
        NER_STATES,
        key="report_state"
    )


    default_latitude = (
        STATE_DEFAULT_COORDINATES[
            report_state
        ][0]
    )


    default_longitude = (
        STATE_DEFAULT_COORDINATES[
            report_state
        ][1]
    )


    # ======================================
    # REPORTER
    # ======================================

    reporter_type = st.selectbox(
        "👤 Reporter Type",
        [
            "Citizen",
            "Field Officer",
            "Emergency Worker",
            "Local Authority"
        ]
    )


    # ======================================
    # ISSUE
    # ======================================

    issue_type = st.selectbox(
        "🚧 Issue Type",
        [
            "Slope Crack",
            "Road Blockage",
            "Rockfall",
            "Mud / Debris",
            "Slope Movement",
            "Landslide",
            "Infrastructure Damage",
            "Other"
        ]
    )


    # ======================================
    # LOCATION
    # ======================================

    st.subheader(
        "📍 Location"
    )


    col1, col2 = (
        st.columns(2)
    )


    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=20.0,
            max_value=31.0,
            value=float(
                default_latitude
            ),
            format="%.6f"
        )


    with col2:

        longitude = st.number_input(
            "Longitude",
            min_value=87.0,
            max_value=99.0,
            value=float(
                default_longitude
            ),
            format="%.6f"
        )


    # ======================================
    # DESCRIPTION
    # ======================================

    description = st.text_area(
        "📝 Description",
        placeholder=(
            "Describe cracks, debris, "
            "slope movement, blocked roads..."
        )
    )


    # ======================================
    # UPLOAD
    # ======================================

    uploaded_file = st.file_uploader(
        "📸 Upload Photo / Video",
        type=[
            "jpg",
            "jpeg",
            "png",
            "mp4",
            "mov"
        ],
        key="citizen_upload"
    )


    # ======================================
    # AI IMAGE ANALYSIS
    # ======================================

    if "cv_result" not in st.session_state:
        st.session_state.cv_result = None

    if "cv_temp_path" not in st.session_state:
        st.session_state.cv_temp_path = None


    if uploaded_file is not None:

        file_type = uploaded_file.type


        # ==================================
        # IMAGE PREVIEW
        # ==================================

        if file_type.startswith("image"):

            st.image(
                uploaded_file,
                caption="Uploaded Evidence",
                use_container_width=True
            )


            # ==================================
            # SAVE TEMP IMAGE
            # ==================================

            temp_folder = (
                "data/citizen_reports/temp"
            )

            os.makedirs(
                temp_folder,
                exist_ok=True
            )


            safe_name = os.path.basename(
                uploaded_file.name
            )

            temp_path = os.path.join(
                temp_folder,
                safe_name
            )


            with open(
                temp_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            st.session_state.cv_temp_path = (
                temp_path
            )


            # ==================================
            # ANALYSE BUTTON
            # ==================================

            if st.button(
                "🤖 Analyse Image with AI",
                use_container_width=True,
                key="analyse_image_button"
            ):

                with st.spinner(
                    "Analysing uploaded image..."
                ):

                    from src.computer_vision import analyse_image

                    cv_result = analyse_image(
                        temp_path
                    )

                st.session_state.cv_result = (
                    cv_result
                )


            # ==================================
            # SHOW SAVED AI RESULT
            # ==================================

            cv_result = (
                st.session_state.cv_result
            )


            if cv_result is not None:

                if not cv_result["success"]:

                    st.error(
                        cv_result["message"]
                    )


                else:

                    st.success(
                        "✅ AI image analysis completed"
                    )


                    col1, col2 = st.columns(2)


                    with col1:

                        st.metric(
                            "Visual Severity Score",
                            cv_result[
                                "visual_severity_score"
                            ]
                        )


                    with col2:

                        st.metric(
                            "Visual Severity Level",
                            cv_result[
                                "visual_severity_level"
                            ]
                        )


                    st.subheader(
                        "🔍 Detected Objects"
                    )


                    if cv_result["detections"]:

                        for detection in (
                            cv_result["detections"]
                        ):

                            object_name = (
                                detection["object"]
                            )

                            confidence = (
                                detection["confidence"]
                                *
                                100
                            )

                            st.write(
                                f"• {object_name} "
                                f"— {confidence:.2f}%"
                            )


                    else:

                        st.info(
                            "No supported objects detected."
                        )


                    st.subheader(
                        "⚠️ AI Observations"
                    )


                    if cv_result["observations"]:

                        for observation in (
                            cv_result["observations"]
                        ):

                            st.write(
                                f"• {observation}"
                            )


                    else:

                        st.info(
                            "No specific person or vehicle "
                            "exposure detected."
                        )


                    annotated_path = (
                        cv_result[
                            "annotated_image"
                        ]
                    )


                    if os.path.exists(
                        annotated_path
                    ):

                        st.image(
                            annotated_path,
                            caption="AI Annotated Image",
                            use_container_width=True
                        )


        # ==================================
        # VIDEO PREVIEW
        # ==================================

        elif file_type.startswith("video"):

            st.video(
                uploaded_file
            )

    # ======================================
    # SUBMIT
    # ======================================

    if st.button(
        "🚨 Submit Field Report",
        use_container_width=True,
        type="primary"
    ):

        if not description.strip():

            st.warning(
                "Please enter a description."
            )


        else:

            full_description = (
                f"State: {report_state}. "
                f"{description}"
            )


            report = save_citizen_report(

                reporter_type=
                    reporter_type,

                issue_type=
                    issue_type,

                latitude=
                    latitude,

                longitude=
                    longitude,

                description=
                    full_description,

                uploaded_file=
                    uploaded_file
            )


            st.success(
                "✅ Field report submitted!"
            )


            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.write(
                    "**Report ID:**",
                    report[
                        "report_id"
                    ]
                )


            with col2:

                st.write(
                    "**Status:**",
                    report[
                        "status"
                    ]
                )


    # ======================================
    # HISTORY
    # ======================================

    st.markdown("---")


    st.subheader(
        "📋 Recent Field Reports"
    )


    reports = load_reports()


    if reports.empty:

        st.info(
            "No citizen reports submitted yet."
        )


    else:

        display_columns = [
            "report_id",
            "timestamp",
            "reporter_type",
            "issue_type",
            "latitude",
            "longitude",
            "status"
        ]


        st.dataframe(

            reports[
                display_columns
            ].sort_values(
                by="timestamp",
                ascending=False
            ),

            use_container_width=True
        )



# ==========================================
# PAGE 6 - GENAI ASSISTANT
# ==========================================

elif page == "🧠 GenAI Assistant":

    st.header("🧠 GenAI Landslide Assistant")

    st.info(
        "Ask about the current landslide risk, important risk drivers, "
        "preparedness actions, road safety, emergency response, "
        "or how the NER monitoring system works."
    )

    st.caption(
        "GenAI explains the ML result and preparedness actions. "
        "It does not replace the XGBoost risk prediction engine."
    )

    groq_api_key = os.getenv("GROQ_API_KEY")

    if ChatGroq is None:

        st.error(
            "langchain-groq is not installed. "
            "Run: pip install langchain-groq"
        )

    elif not groq_api_key:

        st.error(
            "GROQ_API_KEY is missing from your .env file."
        )

        st.code(
            "GROQ_API_KEY=your_groq_api_key_here"
        )

    else:

        # ======================================
        # CURRENT LIVE RISK CONTEXT
        # ======================================

        current_risk_context = (
            "No live risk prediction has been generated yet."
        )

        if st.session_state.risk_result is not None:

            risk = st.session_state.risk_result
            inputs = st.session_state.risk_inputs or {}
            state = st.session_state.risk_state

            current_risk_context = f"""
Current live prediction:
State: {state}
Location: {risk.get('city', 'Unknown')}
Risk Level: {risk.get('risk_level', 'Unknown')}
Risk Score: {risk.get('risk_score', 0):.2f}%
Rainfall 24h: {inputs.get('rainfall_24h', 0):.1f} mm
Rainfall 3d: {inputs.get('rainfall_3d', 0):.1f} mm
Rainfall 7d: {inputs.get('rainfall_7d', 0):.1f} mm
Soil Water Layer 1: {inputs.get('soil_water_1', 0):.2f}
Soil Water Layer 2: {inputs.get('soil_water_2', 0):.2f}
Temperature: {risk.get('temperature_c', 0):.1f} C
Humidity: {risk.get('humidity', 0)}%
Pressure: {risk.get('pressure_hpa', 0)} hPa
Elevation: {risk.get('elevation_m', 0):.0f} m
Slope: {risk.get('slope_degree', 0):.2f} degrees
Land Cover: {risk.get('landcover_class', 'Unknown')}
Recommended Action: {risk.get('recommendation', 'Not available')}
"""

            st.success(
                "✅ Current live risk prediction is connected to the GenAI Assistant."
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Risk Level",
                    risk.get("risk_level", "Unknown")
                )

            with c2:
                st.metric(
                    "Risk Score",
                    f"{risk.get('risk_score', 0):.2f}%"
                )

            with c3:
                st.metric(
                    "Location",
                    f"{risk.get('city', 'Unknown')}, {state}"
                )

        else:

            st.warning(
                "Generate a Live Risk Prediction first if you want the "
                "assistant to explain a specific prediction."
            )

        # ======================================
        # QUICK QUESTIONS
        # ======================================

        st.subheader("⚡ Quick Questions")

        q1, q2, q3 = st.columns(3)
        quick_prompt = None

        with q1:
            if st.button(
                "🤔 Why is the risk high?",
                use_container_width=True
            ):
                quick_prompt = (
                    "Explain the main reasons behind the current landslide "
                    "risk using the available prediction data."
                )

        with q2:
            if st.button(
                "🚨 What action should we take?",
                use_container_width=True
            ):
                quick_prompt = (
                    "Give concise immediate preparedness and emergency-response "
                    "actions for authorities and field officers based on the "
                    "current risk."
                )

        with q3:
            if st.button(
                "🔄 Explain project workflow",
                use_container_width=True
            ):
                quick_prompt = (
                    "Explain the NER landslide early warning system workflow "
                    "from environmental data to AI prediction, GIS monitoring "
                    "and automatic alerts."
                )

        st.markdown("---")

        # ======================================
        # CHAT HISTORY
        # ======================================

        for message in st.session_state.genai_messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input(
            "Ask the NER Landslide AI Assistant..."
        )

        prompt_to_send = quick_prompt or user_prompt

        # ======================================
        # GENERATE RESPONSE
        # ======================================

        if prompt_to_send:

            st.session_state.genai_messages.append(
                {
                    "role": "user",
                    "content": prompt_to_send
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt_to_send)

            system_prompt = f"""
You are the GenAI Assistant inside an AI-Based Landslide
Early Warning and Risk Monitoring System for India's
North Eastern Region.

Your responsibilities:
- Explain landslide risk clearly and practically.
- Explain drivers such as rainfall, soil water, slope,
  elevation, weather, pressure and land cover.
- Suggest preparedness and emergency-response actions.
- Explain possible road-connectivity implications.
- Help authorities and field officers understand model output.
- Never claim that a landslide will definitely occur.
- State that the ML output is a risk estimate and should be
  combined with official warnings and field verification.
- Keep emergency advice concise, actionable and safety-focused.

System capabilities:
- XGBoost landslide risk prediction
- Live weather integration
- GIS risk mapping
- Road connectivity assessment
- Emergency prioritisation
- Citizen / field reporting
- Computer vision support
- Multilingual alerts
- Automatic Email and SMS alerts
- Low-network / offline alert queue

{current_risk_context}
"""

            try:

                model = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0.3,
                    groq_api_key=groq_api_key
                )

                messages = [
                    ("system", system_prompt)
                ]

                for message in st.session_state.genai_messages[-8:]:

                    messages.append(
                        (
                            message["role"],
                            message["content"]
                        )
                    )

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Analysing landslide context..."
                    ):

                        response = model.invoke(messages)
                        answer = response.content

                        st.markdown(answer)

                st.session_state.genai_messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error("GenAI Assistant error")
                st.code(str(error))

        if st.session_state.genai_messages:

            if st.button(
                "🗑️ Clear GenAI Chat",
                use_container_width=True
            ):

                st.session_state.genai_messages = []
                st.rerun()


# ==========================================
# PAGE 6 - HISTORICAL ANALYTICS
# ==========================================

elif page == "📊 Historical Analytics":

    st.header(
        "📊 NER Historical Landslide Analytics"
    )


    if df is None:

        st.error(
            "Training dataset not available."
        )


    else:

        analytics_state = st.selectbox(
            "🌏 Select State",
            [
                "All NER States"
            ]
            +
            NER_STATES
        )


        positive_df = df[
            df[
                "landslide_occurred"
            ]
            ==
            1
        ].copy()


        if analytics_state != (
            "All NER States"
        ):

            positive_df = positive_df[
                positive_df[
                    "ner_state"
                ]
                ==
                analytics_state
            ].copy()


        st.metric(
            "🏔️ Historical Landslide Records",
            len(
                positive_df
            )
        )


        # ======================================
        # STATE DISTRIBUTION
        # ======================================

        if analytics_state == (
            "All NER States"
        ):

            st.subheader(
                "🌏 Landslides by State"
            )


            state_counts = (
                positive_df[
                    "ner_state"
                ]
                .value_counts()
                .reindex(
                    NER_STATES,
                    fill_value=0
                )
            )


            st.bar_chart(
                state_counts
            )


        # ======================================
        # RAINFALL
        # ======================================

        st.subheader(
            "🌧️ Rainfall Statistics"
        )


        rainfall_columns = [
            "rainfall_24h_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm"
        ]


        if not positive_df.empty:

            st.dataframe(

                positive_df[
                    rainfall_columns
                ].describe(),

                use_container_width=True
            )


        # ======================================
        # TERRAIN
        # ======================================

        st.subheader(
            "⛰️ Terrain Statistics"
        )


        terrain_columns = [
            "elevation_m",
            "slope_degree",
            "aspect_degree"
        ]


        if not positive_df.empty:

            st.dataframe(

                positive_df[
                    terrain_columns
                ].describe(),

                use_container_width=True
            )


        # ======================================
        # LAND COVER
        # ======================================

        st.subheader(
            "🌳 Land Cover Distribution"
        )


        if not positive_df.empty:

            landcover_counts = (
                positive_df[
                    "landcover_code"
                ]
                .value_counts()
                .sort_index()
            )


            st.bar_chart(
                landcover_counts
            )


        # ======================================
        # PREVIEW
        # ======================================

        st.subheader(
            "📋 Dataset Preview"
        )


        preview_columns = [
            "event_date",
            "ner_state",
            "latitude",
            "longitude",
            "rainfall_24h_mm",
            "rainfall_7d_mm",
            "elevation_m",
            "slope_degree",
            "landcover_code"
        ]


        available_columns = [
            column
            for column
            in preview_columns
            if column
            in positive_df.columns
        ]


        st.dataframe(

            positive_df[
                available_columns
            ].head(50),

            use_container_width=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "🏔️ North Eastern Region AI-Based "
    "Landslide Early Warning and Risk Monitoring System"
)