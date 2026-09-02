"""
sms_alert.py
------------

Twilio Trial SMS Alert
for NER Landslide Early Warning System.

Important:
Twilio Trial only allows predefined SMS templates.
This code uses:
    sms_account_alerts

Run:
    python src/sms_alert.py
"""

import os

from dotenv import load_dotenv
from twilio.rest import Client


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


TWILIO_ACCOUNT_SID = os.getenv(
    "TWILIO_ACCOUNT_SID"
)

TWILIO_AUTH_TOKEN = os.getenv(
    "TWILIO_AUTH_TOKEN"
)

TWILIO_PHONE_NUMBER = os.getenv(
    "TWILIO_PHONE_NUMBER"
)


# ==========================================
# CHECK CONFIGURATION
# ==========================================

def check_twilio_config():

    if not TWILIO_ACCOUNT_SID:
        return False

    if not TWILIO_AUTH_TOKEN:
        return False

    if not TWILIO_PHONE_NUMBER:
        return False

    return True


# ==========================================
# SEND TRIAL SMS
# ==========================================

def send_sms_alert(
    receiver_number
):

    if not check_twilio_config():

        return {
            "success": False,
            "message": (
                "Twilio configuration missing "
                "in .env file."
            )
        }


    try:

        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN
        )


        # Twilio Trial predefined template
        message = client.messages.create(

            body="sms_account_alerts",

            from_=TWILIO_PHONE_NUMBER,

            to=receiver_number
        )


        return {
            "success": True,
            "message": (
                "Twilio Trial SMS sent successfully."
            ),
            "sid": message.sid
        }


    except Exception as error:

        return {
            "success": False,
            "message": str(error)
        }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "TWILIO TRIAL SMS TEST"
    )

    print(
        "======================================"
    )


    receiver_number = input(
        "\nEnter verified mobile number "
        "with country code: "
    )


    result = send_sms_alert(
        receiver_number=
            receiver_number
    )


    print(
        "\nResult:"
    )

    print(
        result
    )