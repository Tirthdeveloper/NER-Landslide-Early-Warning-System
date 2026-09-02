"""
road_connectivity.py
--------------------

Road Connectivity Assessment Module
for NER Landslide Early Warning System.

Features:
- Road status estimation
- Risk-linked road condition
- Citizen-report impact
- Priority assessment

Run:
    python src/road_connectivity.py
"""

import os

import pandas as pd


# ==========================================
# FILE PATHS
# ==========================================

REPORT_FILE = (
    "data/citizen_reports/"
    "citizen_reports.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "road_connectivity_status.csv"
)


# ==========================================
# ROAD STATUS FUNCTION
# ==========================================

def calculate_road_status(
    risk_score,
    issue_type=None,
    visual_severity=None
):

    """
    Estimate road connectivity status.

    Status:
    - OPEN
    - AT RISK
    - PARTIALLY BLOCKED
    - BLOCKED
    """

    score = float(
        risk_score
    )


    # ======================================
    # CRITICAL FIELD REPORT OVERRIDE
    # ======================================

    if issue_type in [
        "Road Blockage",
        "Landslide"
    ]:

        if visual_severity in [
            "HIGH",
            "CRITICAL"
        ]:

            return (
                "BLOCKED",
                "P1"
            )

        return (
            "PARTIALLY BLOCKED",
            "P1"
        )


    # ======================================
    # DEBRIS / ROCKFALL
    # ======================================

    if issue_type in [
        "Rockfall",
        "Mud / Debris",
        "Slope Movement"
    ]:

        if score >= 80:

            return (
                "BLOCKED",
                "P1"
            )

        elif score >= 60:

            return (
                "PARTIALLY BLOCKED",
                "P2"
            )


    # ======================================
    # RISK BASED STATUS
    # ======================================

    if score >= 80:

        return (
            "AT RISK",
            "P1"
        )


    elif score >= 60:

        return (
            "AT RISK",
            "P2"
        )


    elif score >= 30:

        return (
            "OPEN",
            "P3"
        )


    else:

        return (
            "OPEN",
            "P4"
        )


# ==========================================
# ROAD MESSAGE
# ==========================================

def get_road_message(
    road_status
):

    if road_status == "BLOCKED":

        return (
            "Road may be unsafe or blocked. "
            "Immediate field verification "
            "and traffic restriction recommended."
        )


    elif road_status == "PARTIALLY BLOCKED":

        return (
            "Possible partial blockage detected. "
            "Traffic should be controlled and "
            "field inspection is recommended."
        )


    elif road_status == "AT RISK":

        return (
            "Road currently accessible but "
            "landslide conditions indicate elevated risk."
        )


    else:

        return (
            "Road connectivity currently appears normal."
        )


# ==========================================
# ASSESS ROAD CONNECTIVITY
# ==========================================

def assess_road_connectivity(
    location,
    risk_score,
    risk_level,
    issue_type=None,
    visual_severity=None
):

    road_status, priority = (
        calculate_road_status(

            risk_score=
                risk_score,

            issue_type=
                issue_type,

            visual_severity=
                visual_severity
        )
    )


    message = get_road_message(
        road_status
    )


    return {

        "location":
            location,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "issue_type":
            issue_type,

        "visual_severity":
            visual_severity,

        "road_status":
            road_status,

        "response_priority":
            priority,

        "message":
            message
    }


# ==========================================
# BUILD ROAD STATUS FROM CITIZEN REPORTS
# ==========================================

def build_road_connectivity_dataset():

    if not os.path.exists(
        REPORT_FILE
    ):

        print(
            "⚠️ Citizen reports file not found."
        )

        return None


    reports = pd.read_csv(
        REPORT_FILE
    )


    if reports.empty:

        print(
            "⚠️ No citizen reports available."
        )

        return None


    results = []


    for _, row in reports.iterrows():

        issue_type = row.get(
            "issue_type",
            None
        )


        # Prototype default:
        # no ML risk score stored in report yet.
        risk_score = 50


        # If future CV severity column exists
        visual_severity = row.get(
            "cv_severity",
            None
        )


        road_status, priority = (
            calculate_road_status(

                risk_score=
                    risk_score,

                issue_type=
                    issue_type,

                visual_severity=
                    visual_severity
            )
        )


        result = {

            "report_id":
                row.get(
                    "report_id"
                ),

            "timestamp":
                row.get(
                    "timestamp"
                ),

            "latitude":
                row.get(
                    "latitude"
                ),

            "longitude":
                row.get(
                    "longitude"
                ),

            "issue_type":
                issue_type,

            "road_status":
                road_status,

            "response_priority":
                priority,

            "message":
                get_road_message(
                    road_status
                )
        }


        results.append(
            result
        )


    road_df = pd.DataFrame(
        results
    )


    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    road_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    return road_df


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "ROAD CONNECTIVITY TEST"
    )

    print(
        "======================================"
    )


    result = assess_road_connectivity(

        location=
            "Haflong, Assam",

        risk_score=
            78.0,

        risk_level=
            "HIGH",

        issue_type=
            "Mud / Debris",

        visual_severity=
            "HIGH"
    )


    print(
        "\nLocation:"
    )

    print(
        result[
            "location"
        ]
    )


    print(
        "\nRisk Score:"
    )

    print(
        result[
            "risk_score"
        ]
    )


    print(
        "\nRoad Status:"
    )

    print(
        result[
            "road_status"
        ]
    )


    print(
        "\nResponse Priority:"
    )

    print(
        result[
            "response_priority"
        ]
    )


    print(
        "\nRecommendation:"
    )

    print(
        result[
            "message"
        ]
    )


    # ======================================
    # BUILD DATASET
    # ======================================

    road_df = (
        build_road_connectivity_dataset()
    )


    if road_df is not None:

        print(
            "\n✅ Road connectivity dataset created"
        )

        print(
            road_df.head()
        )

        print(
            "\nSaved at:"
        )

        print(
            OUTPUT_FILE
        )