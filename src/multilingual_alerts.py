
"""
multilingual_alerts.py
----------------------

Multilingual Alert Generator for
NER Landslide Early Warning System.

Supported languages:
- English
- Hindi
- Assamese

This module generates alert text only.
Actual email/SMS sending remains handled
by alerts.py or the SMS provider module.
"""

LANGUAGES = [
    "English",
    "Hindi",
    "Assamese"
]


def generate_multilingual_alert(
    language,
    location,
    risk_score,
    risk_level,
    rainfall_24h,
    rainfall_7d,
    road_status,
    priority_level,
    response_time
):
    """
    Generate an emergency alert message
    in the selected language.
    """

    language = language.strip().title()

    if language not in LANGUAGES:
        language = "English"

    # ==========================================
    # ENGLISH
    # ==========================================

    if language == "English":

        subject = (
            f"NER Landslide {risk_level} Alert - {location}"
        )

        message = f"""
NER LANDSLIDE EARLY WARNING ALERT

Location: {location}
Risk Score: {risk_score:.2f}%
Risk Level: {risk_level}

Rainfall - Last 24 Hours: {rainfall_24h:.1f} mm
Rainfall - Last 7 Days: {rainfall_7d:.1f} mm

Road Status: {road_status}
Emergency Priority: {priority_level}
Target Response Time: {response_time}

Recommended Action:
Please verify field conditions, monitor vulnerable slopes
and roads, and follow district disaster management guidance.

This is a model-estimated risk alert and should be
verified with field observations.
""".strip()

    # ==========================================
    # HINDI
    # ==========================================

    elif language == "Hindi":

        subject = (
            f"NER भूस्खलन {risk_level} चेतावनी - {location}"
        )

        message = f"""
NER भूस्खलन प्रारंभिक चेतावनी

स्थान: {location}
जोखिम स्कोर: {risk_score:.2f}%
जोखिम स्तर: {risk_level}

पिछले 24 घंटे की वर्षा: {rainfall_24h:.1f} मिमी
पिछले 7 दिनों की वर्षा: {rainfall_7d:.1f} मिमी

सड़क स्थिति: {road_status}
आपात प्राथमिकता: {priority_level}
लक्षित प्रतिक्रिया समय: {response_time}

अनुशंसित कार्रवाई:
मैदानी स्थिति की तुरंत पुष्टि करें, संवेदनशील ढलानों और
सड़कों की निगरानी बढ़ाएं तथा जिला आपदा प्रबंधन
प्राधिकरण के निर्देशों का पालन करें।

यह मॉडल द्वारा अनुमानित जोखिम चेतावनी है।
मैदानी सत्यापन आवश्यक है।
""".strip()

    # ==========================================
    # ASSAMESE
    # ==========================================

    else:

        subject = (
            f"NER ভূমিস্খলন {risk_level} সতৰ্কবাণী - {location}"
        )

        message = f"""
NER ভূমিস্খলন আগতীয়া সতৰ্কবাণী

স্থান: {location}
বিপদ স্ক'ৰ: {risk_score:.2f}%
বিপদৰ স্তৰ: {risk_level}

যোৱা ২৪ ঘণ্টাৰ বৰষুণ: {rainfall_24h:.1f} মিমি
যোৱা ৭ দিনৰ বৰষুণ: {rainfall_7d:.1f} মিমি

ৰাস্তাৰ অৱস্থা: {road_status}
জৰুৰী অগ্ৰাধিকাৰ: {priority_level}
লক্ষ্য প্ৰতিক্ৰিয়া সময়: {response_time}

পৰামৰ্শ:
ক্ষেত্ৰৰ অৱস্থা সোনকালে পৰীক্ষা কৰক, বিপদজনক ঢাল আৰু
ৰাস্তাবোৰ ঘনিষ্ঠভাৱে নিৰীক্ষণ কৰক আৰু জিলা দুৰ্যোগ
ব্যৱস্থাপনা কৰ্তৃপক্ষৰ নিৰ্দেশনা অনুসৰণ কৰক।

এইটো মডেলৰ দ্বাৰা অনুমান কৰা বিপদৰ সতৰ্কবাণী।
ক্ষেত্ৰত সত্যাপন কৰাটো প্ৰয়োজনীয়।
""".strip()

    return {
        "language": language,
        "subject": subject,
        "message": message
    }


if __name__ == "__main__":

    result = generate_multilingual_alert(
        language="Assamese",
        location="Haflong, Assam",
        risk_score=78.06,
        risk_level="HIGH",
        rainfall_24h=80,
        rainfall_7d=350,
        road_status="AT RISK",
        priority_level="P2 - HIGH",
        response_time="15-60 min"
    )

    print("\nSubject:")
    print(result["subject"])

    print("\nMessage:")
    print(result["message"])