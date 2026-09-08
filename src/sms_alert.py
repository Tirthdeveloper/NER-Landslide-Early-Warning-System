"""
sms_alert.py
------------

Twilio SMS alerts for the NER Landslide Risk Monitoring System.

This version is compatible with the current Twilio Trial restriction:
trial SMS requests must use one of Twilio's predefined body templates.

Required .env variables:
    TWILIO_ACCOUNT_SID=...
    TWILIO_AUTH_TOKEN=...

Optional / production variables:
    TWILIO_PHONE_NUMBER=+1XXXXXXXXXX
    EMERGENCY_PHONE=+91XXXXXXXXXX

Do not put credentials directly in this file.
"""

import os

from dotenv import load_dotenv
from twilio.rest import Client


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

TWILIO_PHONE_NUMBER = (
    os.getenv("TWILIO_PHONE_NUMBER")
    or os.getenv("TWILIO_FROM_NUMBER")
)

DEFAULT_RECEIVER_PHONE = (
    os.getenv("EMERGENCY_PHONE")
    or os.getenv("ALERT_RECEIVER_PHONE")
    or os.getenv("ALERT_PHONE_NUMBER")
    or os.getenv("TWILIO_RECEIVER_NUMBER")
)

# Set TWILIO_TRIAL_MODE=false after upgrading the Twilio account.
TWILIO_TRIAL_MODE = os.getenv(
    "TWILIO_TRIAL_MODE",
    "true"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on"
}

# Twilio currently permits this predefined SMS body name during trial.
TWILIO_TRIAL_TEMPLATE = os.getenv(
    "TWILIO_TRIAL_TEMPLATE",
    "sms_internal_alerts"
)


# ==========================================
# HELPERS
# ==========================================

def normalize_phone_number(phone_number):
    """Convert common Indian number formats to E.164."""

    if phone_number is None:
        return None

    phone = (
        str(phone_number)
        .strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )

    if len(phone) == 10 and phone.isdigit():
        return "+91" + phone

    if (
        len(phone) == 12
        and phone.startswith("91")
        and phone.isdigit()
    ):
        return "+" + phone

    return phone


def check_sms_config():
    """Check the credentials needed for the selected mode."""

    missing = []

    if not TWILIO_ACCOUNT_SID:
        missing.append("TWILIO_ACCOUNT_SID")

    if not TWILIO_AUTH_TOKEN:
        missing.append("TWILIO_AUTH_TOKEN")

    # In production/custom-body mode a configured sender is required.
    # Trial SMS may use the trial sender assigned by Twilio; if the
    # configured number works for the account we still use it.
    if not TWILIO_TRIAL_MODE and not TWILIO_PHONE_NUMBER:
        missing.append("TWILIO_PHONE_NUMBER")

    return {
        "success": len(missing) == 0,
        "missing": missing
    }


def build_custom_alert(
    location,
    risk_score,
    risk_level,
    recommendation
):
    """Build the custom body used after trial restrictions are removed."""

    lines = [
        "NER Landslide Risk Alert"
    ]

    if risk_level:
        lines.append(
            f"Risk Level: {risk_level}"
        )

    if risk_score is not None:
        try:
            lines.append(
                f"AI Risk Score: {float(risk_score):.2f}%"
            )
        except (TypeError, ValueError):
            lines.append(
                f"AI Risk Score: {risk_score}"
            )

    if location:
        lines.append(
            f"Location: {location}"
        )

    if recommendation:
        lines.append(
            f"Action: {recommendation}"
        )

    return "\n".join(lines)


# ==========================================
# SEND SMS ALERT
# ==========================================

def send_sms_alert(
    receiver_number=None,
    location="NER",
    risk_score=None,
    risk_level=None,
    recommendation=None,
    **kwargs
):
    """
    Send an SMS through Twilio.

    The converted FastAPI backend can call:
        send_sms_alert(receiver_number="+91...")

    It also accepts risk information for future production/custom messages.
    """

    receiver_number = (
        receiver_number
        or kwargs.get("receiver_phone")
        or kwargs.get("phone_number")
        or kwargs.get("mobile_number")
        or kwargs.get("receiver_mobile")
        or DEFAULT_RECEIVER_PHONE
    )

    receiver_number = normalize_phone_number(
        receiver_number
    )

    config = check_sms_config()

    if not config["success"]:
        return {
            "success": False,
            "message": (
                "Twilio configuration missing: "
                + ", ".join(config["missing"])
            ),
            "provider": "Twilio"
        }

    if not receiver_number:
        return {
            "success": False,
            "message": (
                "SMS receiver number is missing. "
                "Set EMERGENCY_PHONE in .env or pass receiver_number."
            ),
            "provider": "Twilio"
        }

    if not receiver_number.startswith("+"):
        return {
            "success": False,
            "message": (
                "Receiver number must use E.164 format, "
                "for example +919876543210."
            ),
            "provider": "Twilio"
        }

    # Trial accounts cannot send arbitrary custom SMS bodies.
    if TWILIO_TRIAL_MODE:
        sms_body = TWILIO_TRIAL_TEMPLATE
    else:
        sms_body = build_custom_alert(
            location=location,
            risk_score=risk_score,
            risk_level=risk_level,
            recommendation=recommendation
        )

    try:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN
        )

        message_args = {
            "body": sms_body,
            "to": receiver_number
        }

        # Use a configured sender when one is available.
        if TWILIO_PHONE_NUMBER:
            if not TWILIO_PHONE_NUMBER.startswith("+"):
                return {
                    "success": False,
                    "message": (
                        "TWILIO_PHONE_NUMBER must use E.164 "
                        "format, for example +1XXXXXXXXXX."
                    ),
                    "provider": "Twilio"
                }

            message_args["from_"] = TWILIO_PHONE_NUMBER

        message = client.messages.create(
            **message_args
        )

        return {
            "success": True,
            "message": (
                "Twilio trial SMS request accepted."
                if TWILIO_TRIAL_MODE
                else "Twilio SMS alert sent successfully."
            ),
            "provider": "Twilio",
            "sid": message.sid,
            "status": getattr(
                message,
                "status",
                None
            ),
            "trial_mode": TWILIO_TRIAL_MODE,
            "template": (
                TWILIO_TRIAL_TEMPLATE
                if TWILIO_TRIAL_MODE
                else None
            )
        }

    except Exception as error:
        return {
            "success": False,
            "message": str(error),
            "provider": "Twilio",
            "trial_mode": TWILIO_TRIAL_MODE
        }


# ==========================================
# DIRECT TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )
    print(
        "TWILIO SMS ALERT TEST"
    )
    print(
        "======================================"
    )

    print(
        "Mode:",
        "TRIAL TEMPLATE"
        if TWILIO_TRIAL_MODE
        else "CUSTOM / PRODUCTION"
    )

    receiver = input(
        "\nEnter verified receiver mobile number "
        "(example +919876543210): "
    ).strip()

    result = send_sms_alert(
        receiver_number=receiver,
        location="Haflong, Assam",
        risk_score=82.50,
        risk_level="CRITICAL",
        recommendation=(
            "Check vulnerable slopes and follow "
            "official disaster-management instructions."
        )
    )

    print(
        "\nResult:"
    )
    print(
        result
    )
