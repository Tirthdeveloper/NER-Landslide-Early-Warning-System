"""
app.py
------

FastAPI backend for the NER AI-Based Landslide Early Warning System.

This file contains NO Streamlit UI code.

Run with:
    uvicorn app:app --reload

Frontend:
    static/index.html
    static/styles.css
    static/app.js
"""

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

load_dotenv()

# Existing working project modules
from src.weather_api import get_current_weather
from src.live_risk_prediction import get_live_risk
from src.citizen_reporting import save_citizen_report, load_reports
from src.alerts import send_email_alert
from src.sms_alert import send_sms_alert
from src.road_connectivity import assess_road_connectivity
from src.emergency_prioritisation import calculate_emergency_priority
from src.multilingual_alerts import LANGUAGES, generate_multilingual_alert
from src.offline_support import get_offline_status, queue_alert


# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="NER Landslide Early Warning System",
    version="2.0.0",
    description="FastAPI backend for HTML/CSS/JavaScript frontend."
)

# ==========================================
# CORS MIDDLEWARE
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def fix_vercel_path(request: Request, call_next):
    path_param = request.query_params.get("path")
    if path_param:
        clean_path = path_param.split("?")[0].lstrip("/")
        request.scope["path"] = f"/api/{clean_path}"
    return await call_next(request)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_FILE = BASE_DIR / "Data" / "Processed" / "ner_landslide_training.csv"
TEMP_DIR = Path("/tmp/citizen_reports/temp") if os.environ.get("VERCEL") else BASE_DIR / "Data" / "citizen_reports" / "temp"

try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

try:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )

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
    "Assam": "Haflong",
    "Arunachal Pradesh": "Itanagar",
    "Manipur": "Imphal",
    "Meghalaya": "Shillong",
    "Mizoram": "Aizawl",
    "Nagaland": "Kohima",
    "Sikkim": "Gangtok",
    "Tripura": "Agartala"
}

STATE_DEFAULT_COORDINATES = {
    "Assam": (25.1648, 93.0176),
    "Arunachal Pradesh": (27.0844, 93.6053),
    "Manipur": (24.8170, 93.9368),
    "Meghalaya": (25.5788, 91.8933),
    "Mizoram": (23.7271, 92.7176),
    "Nagaland": (25.6751, 94.1086),
    "Sikkim": (27.3314, 88.6138),
    "Tripura": (23.8315, 91.2868)
}

EMERGENCY_EMAIL = (
    os.getenv("ALERT_RECEIVER_EMAIL")
    or os.getenv("EMERGENCY_EMAIL")
    or "uiwizards2026@gmail.com"
)

EMERGENCY_PHONE = (
    os.getenv("ALERT_RECEIVER_PHONE")
    or os.getenv("EMERGENCY_PHONE")
    or os.getenv("TWILIO_RECEIVER_NUMBER")
    or ""
)

# In-memory duplicate protection for automatic email.
# Resets when the backend restarts, which is fine for local prototype use.
sent_automatic_alerts = set()


# ==========================================
# REQUEST MODELS
# ==========================================

class RiskRequest(BaseModel):
    state: str
    city: str

    rainfall_24h_mm: float
    rainfall_3d_mm: float
    rainfall_7d_mm: float

    soil_water_layer_1: float
    soil_water_layer_2: float


class EmailAlertRequest(BaseModel):
    receiver_email: EmailStr
    location: str
    risk_score: float
    risk_level: str

    rainfall_24h: float
    rainfall_3d: float
    rainfall_7d: float

    temperature: float
    slope: float
    elevation: float

    recommendation: str


class GenAIRequest(BaseModel):
    message: str
    risk_context: Optional[dict[str, Any]] = None
    history: Optional[list[dict[str, str]]] = None


class MultilingualAlertRequest(BaseModel):
    language: str
    location: str
    risk_level: str
    risk_score: float
    recommendation: Optional[str] = ""
    rainfall_24h: Optional[float] = 0.0
    rainfall_7d: Optional[float] = 0.0
    road_status: Optional[str] = "MONITOR"
    priority_level: Optional[str] = "STANDARD"
    response_time: Optional[str] = "IMMEDIATE"


# ==========================================
# HELPERS
# ==========================================

def get_current_timestamp():
    return datetime.now().strftime("%d %b %Y, %I:%M:%S %p")


def load_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    return None


def safe_value(value):
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def dataframe_records(dataframe: pd.DataFrame):
    records = dataframe.to_dict(orient="records")

    clean_records = []

    for record in records:
        clean_records.append(
            {
                key: safe_value(value)
                for key, value in record.items()
            }
        )

    return clean_records


def get_risk_driver_summary(result, risk_inputs):
    drivers = []

    rainfall_24h = float(
        risk_inputs.get("rainfall_24h", 0)
    )

    rainfall_7d = float(
        risk_inputs.get("rainfall_7d", 0)
    )

    soil_1 = float(
        risk_inputs.get("soil_water_1", 0)
    )

    soil_2 = float(
        risk_inputs.get("soil_water_2", 0)
    )

    slope = float(
        result.get("slope_degree", 0)
    )

    if rainfall_7d >= 300:
        drivers.append(
            {
                "title": "Heavy 7-day rainfall",
                "detail": f"{rainfall_7d:.1f} mm accumulated rainfall"
            }
        )

    elif rainfall_24h >= 100:
        drivers.append(
            {
                "title": "Intense recent rainfall",
                "detail": f"{rainfall_24h:.1f} mm in the last 24 hours"
            }
        )

    avg_soil = (
        soil_1 + soil_2
    ) / 2

    if avg_soil >= 0.55:
        drivers.append(
            {
                "title": "High soil saturation",
                "detail": f"Average soil water level: {avg_soil:.2f}"
            }
        )

    if slope >= 30:
        drivers.append(
            {
                "title": "Steep terrain",
                "detail": f"Slope angle: {slope:.2f}°"
            }
        )

    if not drivers:
        drivers.append(
            {
                "title": "Combined environmental conditions",
                "detail": (
                    "Risk is based on multiple weather, terrain "
                    "and land-cover features."
                )
            }
        )

    return drivers[:3]


def calculate_map_risk(row):
    """
    This is the SAME GIS display-risk heuristic used
    by the working Streamlit project.
    """

    score = 0

    rainfall_7d = float(
        row.get("rainfall_7d_mm", 0)
    )

    slope = float(
        row.get("slope_degree", 0)
    )

    soil_water = float(
        row.get("soil_water_layer_1", 0)
    )

    elevation = float(
        row.get("elevation_m", 0)
    )


    if rainfall_7d >= 300:
        score += 35

    elif rainfall_7d >= 150:
        score += 25

    elif rainfall_7d >= 75:
        score += 15


    if slope >= 40:
        score += 30

    elif slope >= 25:
        score += 20

    elif slope >= 15:
        score += 10


    if soil_water >= 0.40:
        score += 20

    elif soil_water >= 0.25:
        score += 10


    if elevation >= 1500:
        score += 15

    elif elevation >= 700:
        score += 10


    if score < 30:
        return "LOW", "#22c55e", score

    if score < 60:
        return "MODERATE", "#f59e0b", score

    if score < 80:
        return "HIGH", "#f97316", score

    return "CRITICAL", "#dc2626", score


class UploadedFileCompat:
    """
    Small adapter so existing Streamlit-era citizen-report
    code can keep using .name, .type and .getbuffer().
    """

    def __init__(
        self,
        name: str,
        content_type: str,
        content: bytes
    ):
        self.name = name
        self.type = content_type
        self._content = content

    def getbuffer(self):
        return self._content

    def read(self):
        return self._content


def build_auto_alert_key(
    state,
    result,
    risk_inputs,
    road_result,
    emergency_result
):
    return (
        f"{state}|"
        f"{result.get('city')}|"
        f"{result.get('risk_level')}|"
        f"{float(result.get('risk_score', 0)):.2f}|"
        f"{float(risk_inputs.get('rainfall_24h', 0)):.2f}|"
        f"{float(risk_inputs.get('rainfall_7d', 0)):.2f}|"
        f"{road_result.get('road_status')}|"
        f"{emergency_result.get('priority_level')}"
    )


def try_automatic_email(
    state,
    result,
    risk_inputs,
    road_result,
    emergency_result
):
    """Automatic Email + Twilio SMS for HIGH / CRITICAL risk."""

    risk_level = str(result.get("risk_level", "")).upper()

    if risk_level not in ["HIGH", "CRITICAL"]:
        return {
            "attempted": False,
            "success": False,
            "skipped": True,
            "message": "Automatic alerts are only sent for HIGH or CRITICAL risk.",
            "sms": {
                "attempted": False,
                "success": False,
                "skipped": True,
                "message": "Risk is below HIGH."
            }
        }

    alert_key = build_auto_alert_key(
        state=state,
        result=result,
        risk_inputs=risk_inputs,
        road_result=road_result,
        emergency_result=emergency_result
    )

    if alert_key in sent_automatic_alerts:
        return {
            "attempted": False,
            "success": True,
            "duplicate": True,
            "message": "Duplicate automatic Email + SMS alert blocked.",
            "sms": {
                "attempted": False,
                "success": True,
                "duplicate": True,
                "message": "Duplicate SMS blocked."
            }
        }

    offline_status = get_offline_status()
    location = f"{result.get('city', 'Unknown')}, {state}"

    if not offline_status.get("online", True):
        queue_alert({
            "channel": "EMAIL",
            "receiver": EMERGENCY_EMAIL,
            "location": location,
            "risk_score": result.get("risk_score"),
            "risk_level": result.get("risk_level"),
            "road_status": road_result.get("road_status"),
            "priority": emergency_result.get("priority_level"),
            "status": "PENDING_NETWORK"
        })

        if EMERGENCY_PHONE:
            queue_alert({
                "channel": "SMS",
                "receiver": EMERGENCY_PHONE,
                "location": location,
                "risk_score": result.get("risk_score"),
                "risk_level": result.get("risk_level"),
                "road_status": road_result.get("road_status"),
                "priority": emergency_result.get("priority_level"),
                "status": "PENDING_NETWORK"
            })

        sent_automatic_alerts.add(alert_key)

        return {
            "attempted": False,
            "success": True,
            "queued": True,
            "message": "Network unavailable. Email + SMS alerts queued.",
            "sms": {
                "attempted": False,
                "success": bool(EMERGENCY_PHONE),
                "queued": bool(EMERGENCY_PHONE),
                "message": (
                    "SMS alert queued."
                    if EMERGENCY_PHONE
                    else "Emergency phone number is not configured."
                )
            }
        }

    email_status = send_email_alert(
        receiver_email=EMERGENCY_EMAIL,
        location=location,
        risk_score=result.get("risk_score", 0),
        risk_level=result.get("risk_level", "UNKNOWN"),
        rainfall_24h=risk_inputs.get("rainfall_24h", 0),
        rainfall_3d=risk_inputs.get("rainfall_3d", 0),
        rainfall_7d=risk_inputs.get("rainfall_7d", 0),
        temperature=result.get("temperature_c", 0),
        slope=result.get("slope_degree", 0),
        elevation=result.get("elevation_m", 0),
        recommendation=result.get("recommendation", "")
    )

    if not EMERGENCY_PHONE:
        sms_status = {
            "attempted": False,
            "success": False,
            "message": "Set EMERGENCY_PHONE in .env."
        }
    elif not EMERGENCY_PHONE.startswith("+"):
        sms_status = {
            "attempted": False,
            "success": False,
            "message": "EMERGENCY_PHONE must use format +919876543210."
        }
    else:
        try:
            sms_result = send_sms_alert(
                receiver_number=EMERGENCY_PHONE,
                location=location,
                risk_score=result.get("risk_score", 0),
                risk_level=result.get("risk_level", "UNKNOWN"),
                recommendation=result.get("recommendation", "")
            )
            sms_status = {
                "attempted": True,
                **sms_result
            }
        except Exception as error:
            sms_status = {
                "attempted": True,
                "success": False,
                "message": str(error)
            }

    if (
        email_status.get("success", False)
        or sms_status.get("success", False)
    ):
        sent_automatic_alerts.add(alert_key)

    return {
        "attempted": True,
        **email_status,
        "sms": sms_status,
        "combined_success": (
            email_status.get("success", False)
            and sms_status.get("success", False)
        )
    }


# ==========================================
# FRONTEND
# ==========================================

@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return FileResponse(
            str(index_file)
        )

    return {
        "success": True,
        "message": (
            "Backend is running. "
            "Create static/index.html for the web UI."
        )
    }


# ==========================================
# HEALTH / OVERVIEW
# ==========================================

@app.get("/api/health")
def api_health():

    offline_status = get_offline_status()

    return {
        "success": True,
        "timestamp": get_current_timestamp(),
        "services": {
            "ml_engine": {
                "ready": True,
                "status": "Ready"
            },
            "gis": {
                "ready": True,
                "status": "Ready"
            },
            "weather": {
                "ready": bool(
                    os.getenv(
                        "OPENWEATHER_API_KEY"
                    )
                )
            },
            "email": {
                "ready": bool(
                    os.getenv("EMAIL_SENDER")
                    or os.getenv("EMAIL_ADDRESS")
                    or os.getenv("GMAIL_USER")
                )
            },
            "genai": {
                "ready": bool(
                    os.getenv(
                        "GROQ_API_KEY"
                    )
                )
            }
        },
        "network": offline_status
    }


@app.get("/api/overview")
def api_overview():

    df = load_data()

    if df is None:
        return {
            "success": True,
            "states": 8,
            "historical_events": 0,
            "training_samples": 0,
            "control_samples": 0
        }


    total_records = len(
        df
    )

    total_landslides = int(
        df[
            "landslide_occurred"
        ].sum()
    )

    states = int(
        df[
            "ner_state"
        ].nunique()
    )

    return {
        "success": True,
        "states": states,
        "historical_events": total_landslides,
        "training_samples": total_records,
        "control_samples": (
            total_records -
            total_landslides
        )
    }


# ==========================================
# LIVE RISK PREDICTION
# ==========================================

@app.post("/api/predict")
def api_predict(
    request: RiskRequest
):
    try:
        if request.state not in NER_STATES:
            raise HTTPException(
                status_code=400,
                detail="Invalid NER state."
            )

        result = get_live_risk(
            city=request.city,
            rainfall_24h_mm=request.rainfall_24h_mm,
            rainfall_3d_mm=request.rainfall_3d_mm,
            rainfall_7d_mm=request.rainfall_7d_mm,
            soil_water_layer_1=request.soil_water_layer_1,
            soil_water_layer_2=request.soil_water_layer_2
        )

        if not result.get(
            "success",
            False
        ):
            raise HTTPException(
                status_code=400,
                detail=result.get(
                    "message",
                    "Risk prediction failed."
                )
            )

        risk_inputs = {
            "rainfall_24h":
                request.rainfall_24h_mm,
            "rainfall_3d":
                request.rainfall_3d_mm,
            "rainfall_7d":
                request.rainfall_7d_mm,
            "soil_water_1":
                request.soil_water_layer_1,
            "soil_water_2":
                request.soil_water_layer_2
        }

        road_result = assess_road_connectivity(
            location=(
                f"{result.get('city', request.city)}, "
                f"{request.state}"
            ),
            risk_score=result.get(
                "risk_score",
                0
            ),
            risk_level=result.get(
                "risk_level",
                "UNKNOWN"
            )
        )

        emergency_result = calculate_emergency_priority(
            risk_score=result.get(
                "risk_score",
                0
            ),
            risk_level=result.get(
                "risk_level",
                "UNKNOWN"
            ),
            road_status=road_result.get(
                "road_status",
                "UNKNOWN"
            ),
            rainfall_24h_mm=
                request.rainfall_24h_mm,
            rainfall_7d_mm=
                request.rainfall_7d_mm,
            slope_degree=result.get(
                "slope_degree",
                0
            ),
            visual_severity=None
        )

        automatic_email = (
            try_automatic_email(
                state=request.state,
                result=result,
                risk_inputs=risk_inputs,
                road_result=road_result,
                emergency_result=emergency_result
            )
        )

        return {
            "success": True,
            "timestamp": get_current_timestamp(),
            "state": request.state,
            "risk_inputs": risk_inputs,
            "risk_drivers":
                get_risk_driver_summary(
                    result,
                    risk_inputs
                ),
            "road_connectivity":
                road_result,
            "emergency_priority":
                emergency_result,
            "automatic_email":
                automatic_email,
            "automatic_sms":
                automatic_email.get("sms", {}),
            **result
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {
            "success": False,
            "message": f"Risk assessment error: {str(exc)}"
        }


# ==========================================
# GIS - SAME DATASET DOT LOGIC AS STREAMLIT
# ==========================================

@app.get("/api/gis/points")
def api_gis_points(
    state: str = "All NER States"
):

    df = load_data()

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Training dataset not found."
        )


    if (
        state != "All NER States"
        and
        state not in NER_STATES
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid NER state."
        )


    if state == "All NER States":
        map_df = df.copy()

    else:
        map_df = df[
            df[
                "ner_state"
            ]
            ==
            state
        ].copy()


    points = []


    for _, row in map_df.iterrows():

        risk_level, color, score = (
            calculate_map_risk(
                row
            )
        )

        points.append(
            {
                "state":
                    safe_value(
                        row.get(
                            "ner_state"
                        )
                    ),

                "latitude":
                    safe_value(
                        row.get(
                            "latitude"
                        )
                    ),

                "longitude":
                    safe_value(
                        row.get(
                            "longitude"
                        )
                    ),

                "risk_level":
                    risk_level,

                "risk_score":
                    score,

                "marker_color":
                    color,

                "rainfall_24h_mm":
                    safe_value(
                        row.get(
                            "rainfall_24h_mm"
                        )
                    ),

                "rainfall_3d_mm":
                    safe_value(
                        row.get(
                            "rainfall_3d_mm"
                        )
                    ),

                "rainfall_7d_mm":
                    safe_value(
                        row.get(
                            "rainfall_7d_mm"
                        )
                    ),

                "slope_degree":
                    safe_value(
                        row.get(
                            "slope_degree"
                        )
                    ),

                "elevation_m":
                    safe_value(
                        row.get(
                            "elevation_m"
                        )
                    )
            }
        )


    center = (
        STATE_DEFAULT_COORDINATES.get(
            state,
            (26.0, 93.0)
        )
    )


    return {
        "success": True,
        "state": state,
        "count": len(
            points
        ),
        "center": {
            "latitude": center[0],
            "longitude": center[1]
        },
        "zoom": (
            8
            if state in STATE_DEFAULT_COORDINATES
            else
            6
        ),
        "points": points
    }


# ==========================================
# LIVE WEATHER
# ==========================================

@app.get("/api/weather")
def api_weather(
    city: str
):

    city = city.strip()

    if not city:
        raise HTTPException(
            status_code=400,
            detail="City / location is required."
        )


    weather = get_current_weather(
        city
    )


    if not weather.get(
        "success",
        False
    ):
        raise HTTPException(
            status_code=400,
            detail=weather.get(
                "message",
                "Unable to fetch live weather."
            )
        )


    return {
        "success": True,
        "updated_at":
            get_current_timestamp(),
        **weather
    }


# ==========================================
# CITIZEN / FIELD REPORTING
# ==========================================

@app.post("/api/reports")
async def api_submit_report(
    reporter_type: str = Form(...),
    issue_type: str = Form(...),
    state: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(...),
    photo: Optional[UploadFile] = File(None)
):

    if state not in NER_STATES:
        raise HTTPException(
            status_code=400,
            detail="Invalid NER state."
        )


    if not description.strip():
        raise HTTPException(
            status_code=400,
            detail="Description is required."
        )


    full_description = (
        f"State: {state}. "
        f"{description.strip()}"
    )


    uploaded_file_compat = None


    if photo is not None:

        content = await photo.read()

        uploaded_file_compat = UploadedFileCompat(
            name=os.path.basename(
                photo.filename or
                "evidence_upload"
            ),
            content_type=(
                photo.content_type
                or
                "application/octet-stream"
            ),
            content=content
        )


    report = save_citizen_report(
        reporter_type=reporter_type,
        issue_type=issue_type,
        latitude=latitude,
        longitude=longitude,
        description=full_description,
        uploaded_file=uploaded_file_compat
    )


    return {
        "success": True,
        "message":
            "Field report submitted successfully.",
        "report": report
    }


@app.get("/api/reports")
def api_reports():

    reports = load_reports()

    if reports is None:
        return {
            "success": True,
            "reports": []
        }


    if hasattr(
        reports,
        "empty"
    ) and reports.empty:

        return {
            "success": True,
            "reports": []
        }


    if isinstance(
        reports,
        pd.DataFrame
    ):

        reports = reports.sort_values(
            by="timestamp",
            ascending=False
        )

        return {
            "success": True,
            "reports":
                dataframe_records(
                    reports
                )
        }


    return {
        "success": True,
        "reports": reports
    }


# ==========================================
# COMPUTER VISION
# ==========================================

@app.post("/api/cv/analyse")
async def api_cv_analyse(
    image: UploadFile = File(...)
):

    content_type = (
        image.content_type
        or
        ""
    )


    if not content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )


    safe_name = os.path.basename(
        image.filename
        or
        "field_image.jpg"
    )

    temp_path = (
        TEMP_DIR /
        safe_name
    )


    content = await image.read()

    temp_path.write_bytes(
        content
    )


    try:
        from src.computer_vision import analyse_image

        result = analyse_image(
            str(
                temp_path
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Computer vision analysis failed: "
                f"{error}"
            )
        )


    if not result.get(
        "success",
        False
    ):
        raise HTTPException(
            status_code=400,
            detail=result.get(
                "message",
                "Image analysis failed."
            )
        )


    return {
        "success": True,
        **result
    }


# ==========================================
# EMAIL ALERTS
# ==========================================

@app.post("/api/alerts/email")
def api_email_alert(
    request: EmailAlertRequest
):

    risk_level = (
        request.risk_level
        .upper()
        .strip()
    )


    if risk_level not in [
        "HIGH",
        "CRITICAL"
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Manual emergency email is allowed "
                "only for HIGH or CRITICAL risk."
            )
        )


    result = send_email_alert(
        receiver_email=
            str(
                request.receiver_email
            ),

        location=
            request.location,

        risk_score=
            request.risk_score,

        risk_level=
            risk_level,

        rainfall_24h=
            request.rainfall_24h,

        rainfall_3d=
            request.rainfall_3d,

        rainfall_7d=
            request.rainfall_7d,

        temperature=
            request.temperature,

        slope=
            request.slope,

        elevation=
            request.elevation,

        recommendation=
            request.recommendation
    )


    return result


# ==========================================
# MULTILINGUAL ALERT
# ==========================================

@app.get("/api/alerts/languages")
def api_alert_languages():

    return {
        "success": True,
        "languages": LANGUAGES
    }


@app.post("/api/alerts/multilingual")
def api_multilingual_alert(
    request: MultilingualAlertRequest
):
    try:
        result = generate_multilingual_alert(
            language=request.language,
            location=request.location,
            risk_score=request.risk_score,
            risk_level=request.risk_level,
            rainfall_24h=request.rainfall_24h or 0.0,
            rainfall_7d=request.rainfall_7d or 0.0,
            road_status=request.road_status or "MONITOR",
            priority_level=request.priority_level or "STANDARD",
            response_time=request.response_time or "IMMEDIATE"
        )
    except Exception as e:
        result = {
            "error": str(e),
            "subject": f"NER Landslide {request.risk_level} Alert - {request.location}",
            "message": f"Landslide alert for {request.location}. Risk score: {request.risk_score:.1f}%."
        }

    return {
        "success": True,
        "alert": result
    }


# ==========================================
# GENAI ASSISTANT
# ==========================================

@app.post("/api/genai")
def api_genai(
    request: GenAIRequest
):

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )


    if ChatGroq is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "langchain-groq is not installed."
            )
        )


    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY is missing from .env."
            )
        )


    current_risk_context = (
        "No live risk prediction has been supplied."
    )


    if request.risk_context:

        risk = request.risk_context

        inputs = (
            risk.get(
                "risk_inputs",
                {}
            )
            or
            {}
        )

        current_risk_context = f"""
Current live prediction:
State: {risk.get('state', 'Unknown')}
Location: {risk.get('city', 'Unknown')}
Risk Level: {risk.get('risk_level', 'Unknown')}
Risk Score: {float(risk.get('risk_score', 0)):.2f}%
Rainfall 24h: {float(inputs.get('rainfall_24h', 0)):.1f} mm
Rainfall 3d: {float(inputs.get('rainfall_3d', 0)):.1f} mm
Rainfall 7d: {float(inputs.get('rainfall_7d', 0)):.1f} mm
Soil Water Layer 1: {float(inputs.get('soil_water_1', 0)):.2f}
Soil Water Layer 2: {float(inputs.get('soil_water_2', 0)):.2f}
Temperature: {float(risk.get('temperature_c', 0)):.1f} C
Humidity: {risk.get('humidity', 0)}%
Pressure: {risk.get('pressure_hpa', 0)} hPa
Elevation: {float(risk.get('elevation_m', 0)):.0f} m
Slope: {float(risk.get('slope_degree', 0)):.2f} degrees
Land Cover: {risk.get('landcover_class', 'Unknown')}
Recommended Action: {risk.get('recommendation', 'Not available')}
"""


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
- Automatic Email alerts
- Low-network / offline alert queue

{current_risk_context}
"""


    messages = [
        (
            "system",
            system_prompt
        )
    ]


    history = (
        request.history
        or
        []
    )


    for message in history[-8:]:

        role = (
            message.get(
                "role",
                "user"
            )
        )

        content = (
            message.get(
                "content",
                ""
            )
        )

        if content:
            messages.append(
                (
                    role,
                    content
                )
            )


    messages.append(
        (
            "user",
            request.message
        )
    )


    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        groq_api_key=groq_api_key
    )


    try:

        response = model.invoke(
            messages
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "GenAI Assistant error: "
                f"{error}"
            )
        )


    return {
        "success": True,
        "answer": response.content
    }


# ==========================================
# HISTORICAL ANALYTICS
# ==========================================

@app.get("/api/analytics")
def api_analytics(
    state: str = "All NER States"
):

    df = load_data()


    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Training dataset not found."
        )


    if (
        state != "All NER States"
        and
        state not in NER_STATES
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid NER state."
        )


    positive_df = df[
        df[
            "landslide_occurred"
        ]
        ==
        1
    ].copy()


    if state != "All NER States":

        positive_df = positive_df[
            positive_df[
                "ner_state"
            ]
            ==
            state
        ].copy()


    state_counts = (
        positive_df[
            "ner_state"
        ]
        .value_counts()
        .reindex(
            NER_STATES,
            fill_value=0
        )
        .to_dict()
    )


    rainfall_columns = [
        column
        for column
        in [
            "rainfall_24h_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm"
        ]
        if column in positive_df.columns
    ]


    terrain_columns = [
        column
        for column
        in [
            "elevation_m",
            "slope_degree",
            "aspect_degree"
        ]
        if column in positive_df.columns
    ]


    rainfall_stats = {}

    if (
        not positive_df.empty
        and
        rainfall_columns
    ):

        rainfall_stats = (
            positive_df[
                rainfall_columns
            ]
            .describe()
            .round(3)
            .to_dict()
        )


    terrain_stats = {}

    if (
        not positive_df.empty
        and
        terrain_columns
    ):

        terrain_stats = (
            positive_df[
                terrain_columns
            ]
            .describe()
            .round(3)
            .to_dict()
        )


    landcover_distribution = {}

    if (
        not positive_df.empty
        and
        "landcover_code"
        in positive_df.columns
    ):

        landcover_distribution = (
            positive_df[
                "landcover_code"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
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
        if column in positive_df.columns
    ]


    preview = (
        positive_df[
            available_columns
        ]
        .head(50)
    )


    return {
        "success": True,
        "state": state,
        "historical_records": len(
            positive_df
        ),
        "state_distribution":
            state_counts,
        "rainfall_statistics":
            rainfall_stats,
        "terrain_statistics":
            terrain_stats,
        "landcover_distribution":
            landcover_distribution,
        "preview":
            dataframe_records(
                preview
            )
    }


# ==========================================
# STATE / CONFIG DATA
# ==========================================

@app.get("/api/config")
def api_config():

    return {
        "success": True,
        "states": NER_STATES,
        "default_cities":
            STATE_DEFAULT_CITIES,
        "default_coordinates":
            {
                state: {
                    "latitude":
                        coordinates[0],
                    "longitude":
                        coordinates[1]
                }
                for state, coordinates
                in STATE_DEFAULT_COORDINATES.items()
            },
        "emergency_email":
            EMERGENCY_EMAIL
    }