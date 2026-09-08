"""
alerts.py
---------

Professional HTML email alert system for
NER Landslide Early Warning System.

Uses Gmail SMTP.

Required .env variables:
    EMAIL_ADDRESS=...
    EMAIL_APP_PASSWORD=...
"""

import os
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")


# ==========================================
# CHECK EMAIL CONFIGURATION
# ==========================================

def check_email_config():

    if not EMAIL_ADDRESS:
        return False

    if not EMAIL_APP_PASSWORD:
        return False

    return True


# ==========================================
# EMAIL STYLE HELPERS
# ==========================================

def get_risk_style(risk_level):

    level = str(risk_level).upper()

    styles = {
        "LOW": {
            "color": "#16a34a",
            "bg": "#dcfce7",
            "border": "#86efac",
            "icon": "✅"
        },
        "MODERATE": {
            "color": "#d97706",
            "bg": "#fef3c7",
            "border": "#fcd34d",
            "icon": "⚠️"
        },
        "HIGH": {
            "color": "#ea580c",
            "bg": "#ffedd5",
            "border": "#fdba74",
            "icon": "🚨"
        },
        "CRITICAL": {
            "color": "#dc2626",
            "bg": "#fee2e2",
            "border": "#fca5a5",
            "icon": "🆘"
        }
    }

    return styles.get(
        level,
        {
            "color": "#475569",
            "bg": "#f1f5f9",
            "border": "#cbd5e1",
            "icon": "⚠️"
        }
    )


def build_html_email(
    location,
    risk_score,
    risk_level,
    rainfall_24h,
    rainfall_3d,
    rainfall_7d,
    temperature,
    slope,
    elevation,
    recommendation
):

    style = get_risk_style(
        risk_level
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Landslide Risk Alert</title>
</head>

<body style="
    margin:0;
    padding:0;
    background:#eef2f7;
    font-family:Arial, Helvetica, sans-serif;
    color:#0f172a;
">

<table width="100%" cellspacing="0" cellpadding="0"
       style="background:#eef2f7; padding:30px 12px;">
<tr>
<td align="center">

<table width="100%" cellspacing="0" cellpadding="0"
       style="
           max-width:680px;
           background:#ffffff;
           border-radius:20px;
           overflow:hidden;
           box-shadow:0 10px 35px rgba(15,23,42,0.12);
       ">

    <tr>
        <td style="
            padding:30px 34px;
            background:#0b2540;
            color:#ffffff;
        ">
            <div style="
                font-size:13px;
                letter-spacing:2px;
                text-transform:uppercase;
                opacity:0.8;
                margin-bottom:8px;
            ">
                NER AI LANDSLIDE MONITORING SYSTEM
            </div>

            <div style="
                font-size:28px;
                font-weight:800;
                line-height:1.25;
            ">
                {style["icon"]} Landslide Risk Alert
            </div>

            <div style="
                margin-top:10px;
                font-size:15px;
                color:#dbeafe;
            ">
                AI-assisted situational awareness & decision-support alert
            </div>
        </td>
    </tr>

    <tr>
        <td style="padding:28px 34px 10px 34px;">

            <div style="
                border:1px solid {style["border"]};
                background:{style["bg"]};
                border-radius:16px;
                padding:22px;
            ">
                <div style="
                    font-size:12px;
                    font-weight:700;
                    text-transform:uppercase;
                    letter-spacing:1.5px;
                    color:{style["color"]};
                ">
                    Current Risk Level
                </div>

                <div style="
                    margin-top:5px;
                    font-size:32px;
                    font-weight:900;
                    color:{style["color"]};
                ">
                    {str(risk_level).upper()}
                </div>

                <div style="
                    margin-top:8px;
                    font-size:15px;
                    color:#334155;
                ">
                    AI Risk Score:
                    <strong>{float(risk_score):.2f}%</strong>
                </div>
            </div>

        </td>
    </tr>

    <tr>
        <td style="padding:14px 34px 5px 34px;">
            <div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:14px;
                padding:18px 20px;
            ">
                <div style="
                    font-size:12px;
                    color:#64748b;
                    text-transform:uppercase;
                    letter-spacing:1px;
                    font-weight:700;
                ">
                    📍 Location
                </div>

                <div style="
                    margin-top:7px;
                    font-size:19px;
                    font-weight:700;
                    color:#0f172a;
                ">
                    {location}
                </div>
            </div>
        </td>
    </tr>

    <tr>
        <td style="padding:20px 34px 0 34px;">

            <div style="
                font-size:18px;
                font-weight:800;
                margin-bottom:12px;
            ">
                🌧 Weather Conditions
            </div>

            <table width="100%" cellspacing="0" cellpadding="0">
                <tr>
                    <td width="33%" style="padding:5px;">
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:12px;
                            padding:16px;
                            text-align:center;
                        ">
                            <div style="font-size:11px;color:#64748b;">
                                RAINFALL 24H
                            </div>
                            <div style="margin-top:6px;font-size:20px;font-weight:800;">
                                {float(rainfall_24h):.2f}
                            </div>
                            <div style="font-size:12px;color:#64748b;">mm</div>
                        </div>
                    </td>

                    <td width="33%" style="padding:5px;">
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:12px;
                            padding:16px;
                            text-align:center;
                        ">
                            <div style="font-size:11px;color:#64748b;">
                                RAINFALL 3D
                            </div>
                            <div style="margin-top:6px;font-size:20px;font-weight:800;">
                                {float(rainfall_3d):.2f}
                            </div>
                            <div style="font-size:12px;color:#64748b;">mm</div>
                        </div>
                    </td>

                    <td width="33%" style="padding:5px;">
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:12px;
                            padding:16px;
                            text-align:center;
                        ">
                            <div style="font-size:11px;color:#64748b;">
                                RAINFALL 7D
                            </div>
                            <div style="margin-top:6px;font-size:20px;font-weight:800;">
                                {float(rainfall_7d):.2f}
                            </div>
                            <div style="font-size:12px;color:#64748b;">mm</div>
                        </div>
                    </td>
                </tr>
            </table>

            <table width="100%" cellspacing="0" cellpadding="0">
                <tr>
                    <td width="33%" style="padding:5px;">
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:12px;
                            padding:16px;
                            text-align:center;
                        ">
                            <div style="font-size:11px;color:#64748b;">
                                TEMPERATURE
                            </div>
                            <div style="margin-top:6px;font-size:20px;font-weight:800;">
                                {float(temperature):.1f}°C
                            </div>
                        </div>
                    </td>

                    <td width="33%" style="padding:5px;">
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:12px;
                            padding:16px;
                            text-align:center;
                        ">
                            <div style="font-size:11px;color:#64748b;">
                                SLOPE
                            </div>
                            <div style="margin-top:6px;font-size:20px;font-weight:800;">
                                {float(slope):.1f}°
                            </div>
                        </div>
                    </td>

                    <td width="33%" style="padding:5px;">
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:12px;
                            padding:16px;
                            text-align:center;
                        ">
                            <div style="font-size:11px;color:#64748b;">
                                ELEVATION
                            </div>
                            <div style="margin-top:6px;font-size:20px;font-weight:800;">
                                {float(elevation):.0f} m
                            </div>
                        </div>
                    </td>
                </tr>
            </table>

        </td>
    </tr>

    <tr>
        <td style="padding:22px 34px 0 34px;">

            <div style="
                border-left:5px solid {style["color"]};
                background:#f8fafc;
                border-radius:12px;
                padding:18px 20px;
            ">
                <div style="
                    font-size:14px;
                    font-weight:800;
                    color:#0f172a;
                    margin-bottom:8px;
                ">
                    🛡 Recommended Action
                </div>

                <div style="
                    font-size:15px;
                    line-height:1.65;
                    color:#334155;
                ">
                    {recommendation}
                </div>
            </div>

        </td>
    </tr>

    <tr>
        <td style="padding:22px 34px;">
            <div style="
                background:#fff7ed;
                border:1px solid #fed7aa;
                border-radius:12px;
                padding:14px 16px;
                color:#9a3412;
                font-size:13px;
                line-height:1.55;
            ">
                ⚠️ This is an AI-generated decision-support alert.
                Verify conditions using field observations and official
                disaster-management procedures before taking critical action.
            </div>
        </td>
    </tr>

    <tr>
        <td style="
            padding:22px 34px;
            background:#0b2540;
            color:#cbd5e1;
            text-align:center;
            font-size:12px;
            line-height:1.6;
        ">
            <strong style="color:#ffffff;">
                NER AI-Based Landslide Risk Monitoring System
            </strong>
            <br>
            Automated Email Alert • Email + Twilio SMS Enabled
        </td>
    </tr>

</table>

</td>
</tr>
</table>

</body>
</html>
"""


def build_plain_text_email(
    location,
    risk_score,
    risk_level,
    rainfall_24h,
    rainfall_3d,
    rainfall_7d,
    temperature,
    slope,
    elevation,
    recommendation
):

    return f"""
NER AI-BASED LANDSLIDE RISK MONITORING SYSTEM

LANDSLIDE RISK ALERT

Location: {location}
Risk Level: {risk_level}
AI Risk Score: {float(risk_score):.2f}%

Weather Conditions
------------------
Rainfall 24h: {float(rainfall_24h):.2f} mm
Rainfall 3d: {float(rainfall_3d):.2f} mm
Rainfall 7d: {float(rainfall_7d):.2f} mm
Temperature: {float(temperature):.2f} °C

Terrain
-------
Slope: {float(slope):.2f} degrees
Elevation: {float(elevation):.2f} metres

Recommended Action
------------------
{recommendation}

This is an AI-generated decision-support alert.
Verify with field observations and official procedures.
"""


# ==========================================
# SEND EMAIL ALERT
# ==========================================

def send_email_alert(
    receiver_email,
    location,
    risk_score,
    risk_level,
    rainfall_24h,
    rainfall_3d,
    rainfall_7d,
    temperature,
    slope,
    elevation,
    recommendation
):

    if not check_email_config():
        return {
            "success": False,
            "message": "Email configuration missing."
        }

    subject = (
        f"🚨 {str(risk_level).upper()} Landslide Risk Alert"
        f" | {location}"
    )

    html_body = build_html_email(
        location=location,
        risk_score=risk_score,
        risk_level=risk_level,
        rainfall_24h=rainfall_24h,
        rainfall_3d=rainfall_3d,
        rainfall_7d=rainfall_7d,
        temperature=temperature,
        slope=slope,
        elevation=elevation,
        recommendation=recommendation
    )

    plain_body = build_plain_text_email(
        location=location,
        risk_score=risk_score,
        risk_level=risk_level,
        rainfall_24h=rainfall_24h,
        rainfall_3d=rainfall_3d,
        rainfall_7d=rainfall_7d,
        temperature=temperature,
        slope=slope,
        elevation=elevation,
        recommendation=recommendation
    )

    message = MIMEMultipart("alternative")

    message["From"] = EMAIL_ADDRESS
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(
        MIMEText(
            plain_body,
            "plain",
            "utf-8"
        )
    )

    message.attach(
        MIMEText(
            html_body,
            "html",
            "utf-8"
        )
    )

    try:

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30
        )

        server.ehlo()
        server.starttls()
        server.ehlo()

        server.login(
            EMAIL_ADDRESS,
            EMAIL_APP_PASSWORD
        )

        server.sendmail(
            EMAIL_ADDRESS,
            receiver_email,
            message.as_string()
        )

        server.quit()

        return {
            "success": True,
            "message": "Styled HTML email alert sent successfully."
        }

    except Exception as error:

        return {
            "success": False,
            "message": str(error)
        }


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print("HTML EMAIL ALERT TEST")
    print("======================================")

    receiver = input(
        "\nEnter receiver email: "
    ).strip()

    result = send_email_alert(
        receiver_email=receiver,
        location="Haflong, Assam",
        risk_score=82.50,
        risk_level="CRITICAL",
        rainfall_24h=95,
        rainfall_3d=210,
        rainfall_7d=410,
        temperature=23.5,
        slope=38.5,
        elevation=680,
        recommendation=(
            "Immediate field inspection is recommended. "
            "Monitor vulnerable slopes and restrict traffic "
            "if slope movement is observed."
        )
    )

    print("\nResult:")
    print(result)
