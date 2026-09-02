"""
citizen_reporting.py
--------------------

Citizen / Field Officer reporting module
for NER Landslide Early Warning System.

Features:
- Photo/video upload
- Latitude/Longitude
- Issue type
- Description
- Reporter type
- Timestamp
- CSV report storage

Run with:
    python src/citizen_reporting.py
"""

import os
import uuid

import pandas as pd

from datetime import datetime


# ==========================================
# FILE PATHS
# ==========================================

REPORT_FOLDER = (
    "data/citizen_reports"
)

UPLOAD_FOLDER = (
    "data/citizen_reports/uploads"
)

REPORT_FILE = (
    "data/citizen_reports/"
    "citizen_reports.csv"
)


# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==========================================
# SAVE UPLOADED FILE
# ==========================================

def save_uploaded_file(
    uploaded_file
):

    if uploaded_file is None:
        return None


    # Create unique file name
    unique_id = str(
        uuid.uuid4()
    )[:8]


    original_name = (
        uploaded_file.name
    )


    file_name = (
        f"{unique_id}_"
        f"{original_name}"
    )


    file_path = os.path.join(
        UPLOAD_FOLDER,
        file_name
    )


    # Save uploaded file
    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    return file_path


# ==========================================
# SAVE CITIZEN REPORT
# ==========================================

def save_citizen_report(
    reporter_type,
    issue_type,
    latitude,
    longitude,
    description,
    uploaded_file=None
):

    # ======================================
    # REPORT ID
    # ======================================

    report_id = (
        "RPT_"
        +
        str(
            uuid.uuid4()
        )[:8]
        .upper()
    )


    # ======================================
    # TIMESTAMP
    # ======================================

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    # ======================================
    # SAVE PHOTO / VIDEO
    # ======================================

    file_path = save_uploaded_file(
        uploaded_file
    )


    # ======================================
    # REPORT DATA
    # ======================================

    report = {

        "report_id":
            report_id,

        "timestamp":
            timestamp,

        "reporter_type":
            reporter_type,

        "issue_type":
            issue_type,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "description":
            description,

        "file_path":
            file_path,

        "status":
            "Pending Review"
    }


    # ======================================
    # CREATE DATAFRAME
    # ======================================

    new_report = pd.DataFrame(
        [report]
    )


    # ======================================
    # SAVE CSV
    # ======================================

    if os.path.exists(
        REPORT_FILE
    ):

        old_reports = pd.read_csv(
            REPORT_FILE
        )


        reports = pd.concat(
            [
                old_reports,
                new_report
            ],
            ignore_index=True
        )

    else:

        reports = new_report


    reports.to_csv(
        REPORT_FILE,
        index=False
    )


    return report


# ==========================================
# LOAD REPORTS
# ==========================================

def load_reports():

    if not os.path.exists(
        REPORT_FILE
    ):

        return pd.DataFrame()


    return pd.read_csv(
        REPORT_FILE
    )


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\nCitizen Reporting Module Ready ✅"
    )

    print(
        f"Reports will be stored in:"
    )

    print(
        REPORT_FILE
    )