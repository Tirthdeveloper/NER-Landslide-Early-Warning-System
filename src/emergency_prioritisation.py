"""
emergency_prioritisation.py
---------------------------

Emergency Response Prioritisation Engine
for NER Landslide Early Warning System.

Combines:
- ML risk score / level
- Road connectivity status
- Rainfall intensity
- Slope severity
- Optional visual severity

Output:
- Priority score
- Priority level (P1-P4)
- Recommended response
"""

def calculate_emergency_priority(
    risk_score,
    risk_level,
    road_status,
    rainfall_24h_mm,
    rainfall_7d_mm,
    slope_degree,
    visual_severity=None
):
    score = 0

    # ==========================================
    # ML RISK CONTRIBUTION
    # ==========================================

    if risk_level == "CRITICAL":
        score += 40
    elif risk_level == "HIGH":
        score += 30
    elif risk_level == "MODERATE":
        score += 18
    else:
        score += 5

    # ==========================================
    # ROAD CONNECTIVITY CONTRIBUTION
    # ==========================================

    if road_status == "BLOCKED":
        score += 30
    elif road_status == "PARTIALLY BLOCKED":
        score += 22
    elif road_status == "AT RISK":
        score += 12
    else:
        score += 0

    # ==========================================
    # RAINFALL CONTRIBUTION
    # ==========================================

    if rainfall_24h_mm >= 100:
        score += 12
    elif rainfall_24h_mm >= 60:
        score += 8
    elif rainfall_24h_mm >= 30:
        score += 4

    if rainfall_7d_mm >= 400:
        score += 10
    elif rainfall_7d_mm >= 250:
        score += 7
    elif rainfall_7d_mm >= 120:
        score += 4

    # ==========================================
    # SLOPE CONTRIBUTION
    # ==========================================

    if slope_degree >= 45:
        score += 8
    elif slope_degree >= 30:
        score += 6
    elif slope_degree >= 20:
        score += 3

    # ==========================================
    # COMPUTER VISION CONTRIBUTION
    # ==========================================

    if visual_severity == "CRITICAL":
        score += 12
    elif visual_severity == "HIGH":
        score += 8
    elif visual_severity == "MODERATE":
        score += 4

    # Cap at 100
    score = min(int(round(score)), 100)

    # ==========================================
    # PRIORITY CLASSIFICATION
    # ==========================================

    if score >= 75:
        priority = "P1 - IMMEDIATE"
        response_time = "Immediate / 0-15 min"
        action = (
            "Deploy emergency response team immediately. "
            "Verify the site, restrict vulnerable road access, "
            "notify district disaster management authorities, "
            "and prepare evacuation support if field conditions worsen."
        )

    elif score >= 55:
        priority = "P2 - HIGH"
        response_time = "15-60 min"
        action = (
            "Send a field verification team, intensify monitoring, "
            "place road and local response teams on alert, and "
            "prepare traffic control or temporary closure if required."
        )

    elif score >= 30:
        priority = "P3 - MODERATE"
        response_time = "Within 2-4 hours"
        action = (
            "Continue close monitoring, inspect vulnerable slopes and roads, "
            "and keep local authorities informed about changing conditions."
        )

    else:
        priority = "P4 - ROUTINE"
        response_time = "Routine monitoring"
        action = (
            "Continue routine observation and weather monitoring. "
            "No immediate emergency deployment is indicated by the current model."
        )

    return {
        "priority_score": score,
        "priority_level": priority,
        "response_time": response_time,
        "recommended_action": action
    }


if __name__ == "__main__":

    result = calculate_emergency_priority(
        risk_score=78.0,
        risk_level="HIGH",
        road_status="AT RISK",
        rainfall_24h_mm=80,
        rainfall_7d_mm=350,
        slope_degree=35,
        visual_severity=None
    )

    print("\nEmergency Priority Test")
    print("=======================")
    print("Priority Score:", result["priority_score"])
    print("Priority Level:", result["priority_level"])
    print("Response Time:", result["response_time"])
    print("Recommended Action:", result["recommended_action"])