"""
weather_api.py
--------------

Live weather integration using OpenWeather API.

Run with:
    python src/weather_api.py
"""

import os
import requests
from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")


if not API_KEY:
    print("❌ OPENWEATHER_API_KEY missing in .env")
    raise SystemExit


# ==========================================
# CURRENT WEATHER FUNCTION
# ==========================================

def get_current_weather(city):

    url = (
        "https://api.openweathermap.org/"
        "data/2.5/weather"
    )

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    if response.status_code != 200:

        return {
            "success": False,
            "status_code": response.status_code,
            "message": response.text
        }


    data = response.json()


    # ======================================
    # RAINFALL
    # ======================================

    rain_1h = 0.0

    rain_3h = 0.0


    if "rain" in data:

        rain_1h = data["rain"].get(
            "1h",
            0.0
        )

        rain_3h = data["rain"].get(
            "3h",
            0.0
        )


    # ======================================
    # RESULT
    # ======================================

    result = {

        "success": True,

        "city":
            data["name"],

        "latitude":
            data["coord"]["lat"],

        "longitude":
            data["coord"]["lon"],

        "temperature_c":
            data["main"]["temp"],

        "humidity":
            data["main"]["humidity"],

        "surface_pressure_hpa":
            data["main"]["pressure"],

        "weather":
            data["weather"][0]["description"],

        "wind_speed":
            data["wind"]["speed"],

        "rain_1h_mm":
            rain_1h,

        "rain_3h_mm":
            rain_3h
    }


    return result


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    city = "Assam"

    weather = get_current_weather(
        city
    )


    if weather["success"]:

        print("\n✅ Weather API Working")

        print(
            "\nCity:",
            weather["city"]
        )

        print(
            "Temperature:",
            weather["temperature_c"],
            "°C"
        )

        print(
            "Humidity:",
            weather["humidity"],
            "%"
        )

        print(
            "Pressure:",
            weather["surface_pressure_hpa"],
            "hPa"
        )

        print(
            "Rain 1h:",
            weather["rain_1h_mm"],
            "mm"
        )

        print(
            "Rain 3h:",
            weather["rain_3h_mm"],
            "mm"
        )

        print(
            "Weather:",
            weather["weather"]
        )

    else:

        print("\n❌ Weather API Failed")

        print(
            weather["status_code"]
        )

        print(
            weather["message"]
        )