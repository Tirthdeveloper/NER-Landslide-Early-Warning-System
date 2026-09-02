"""
computer_vision.py
------------------

Enhanced Computer Vision module for
NER Landslide Early Warning System.

Features:
- YOLO object detection
- Person / vehicle exposure detection
- Edge / crack-like texture estimation
- Earth / debris color estimation
- Visual hazard score
- Severity level
- Annotated image output

Run:
    python src/computer_vision.py
"""

import os

import cv2
import numpy as np

from ultralytics import YOLO


# ==========================================
# FILE PATHS
# ==========================================

MODEL_FOLDER = "models/cv"

OUTPUT_FOLDER = "outputs/cv"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "yolov8n.pt"
)


# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("\nLoading Computer Vision model...")


if os.path.exists(MODEL_FILE):

    model = YOLO(
        MODEL_FILE
    )

else:

    model = YOLO(
        "yolov8n.pt"
    )


print(
    "✅ Computer Vision model loaded"
)


# ==========================================
# HAZARD-RELATED OBJECTS
# ==========================================

HAZARD_OBJECTS = {

    "person":
        20,

    "car":
        10,

    "motorcycle":
        10,

    "bus":
        15,

    "truck":
        15
}


# ==========================================
# OBJECT OBSERVATIONS
# ==========================================

OBJECT_MESSAGES = {

    "person":
        "People detected in or near the reported area.",

    "car":
        "Vehicle traffic detected near the reported location.",

    "motorcycle":
        "Two-wheeler movement detected near the reported location.",

    "bus":
        "Public transport exposure detected.",

    "truck":
        "Heavy vehicle exposure detected."
}


# ==========================================
# EARTH / DEBRIS ANALYSIS
# ==========================================

def calculate_earth_ratio(
    image
):

    """
    Estimate percentage of brown / earth-like
    pixels in the image.

    This is only a heuristic and does not
    confirm a landslide.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # Brown / soil-like color range
    lower_brown = np.array(
        [5, 40, 30]
    )

    upper_brown = np.array(
        [30, 255, 220]
    )


    mask = cv2.inRange(
        hsv,
        lower_brown,
        upper_brown
    )


    earth_pixels = np.count_nonzero(
        mask
    )


    total_pixels = (
        image.shape[0]
        *
        image.shape[1]
    )


    if total_pixels == 0:

        return 0.0


    ratio = (
        earth_pixels
        /
        total_pixels
    )


    return float(
        ratio
    )


# ==========================================
# EDGE / CRACK-LIKE TEXTURE
# ==========================================

def calculate_edge_density(
    image
):

    """
    Estimate edge density.

    High edge density can indicate rough
    surfaces, debris, cracks, vegetation,
    structures etc.

    This is not a direct crack detector.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    edges = cv2.Canny(
        gray,
        50,
        150
    )


    edge_pixels = np.count_nonzero(
        edges
    )


    total_pixels = (
        edges.shape[0]
        *
        edges.shape[1]
    )


    if total_pixels == 0:

        return 0.0


    density = (
        edge_pixels
        /
        total_pixels
    )


    return float(
        density
    )


# ==========================================
# SEVERITY LEVEL
# ==========================================

def get_visual_severity(
    score
):

    if score < 25:

        return "LOW"

    elif score < 50:

        return "MODERATE"

    elif score < 75:

        return "HIGH"

    else:

        return "CRITICAL"


# ==========================================
# VISUAL HAZARD SCORE
# ==========================================

def calculate_visual_hazard_score(
    detections,
    earth_ratio,
    edge_density
):

    """
    Combine multiple visual indicators.

    Score is capped at 100.
    """

    score = 0


    # ======================================
    # OBJECT EXPOSURE
    # ======================================

    for detection in detections:

        object_name = detection[
            "object"
        ]


        if object_name in HAZARD_OBJECTS:

            score += HAZARD_OBJECTS[
                object_name
            ]


    # ======================================
    # EARTH / DEBRIS CONTRIBUTION
    # ======================================

    if earth_ratio >= 0.45:

        score += 35

    elif earth_ratio >= 0.30:

        score += 25

    elif earth_ratio >= 0.15:

        score += 15

    elif earth_ratio >= 0.08:

        score += 8


    # ======================================
    # EDGE / ROUGHNESS CONTRIBUTION
    # ======================================

    if edge_density >= 0.20:

        score += 25

    elif edge_density >= 0.12:

        score += 18

    elif edge_density >= 0.07:

        score += 10

    elif edge_density >= 0.03:

        score += 5


    score = min(
        score,
        100
    )


    return int(
        score
    )


# ==========================================
# BUILD OBSERVATIONS
# ==========================================

def build_observations(
    detections,
    earth_ratio,
    edge_density,
    severity
):

    observations = []


    # ======================================
    # OBJECT OBSERVATIONS
    # ======================================

    detected_names = {
        item["object"]
        for item in detections
    }


    for object_name in detected_names:

        if object_name in OBJECT_MESSAGES:

            observations.append(
                OBJECT_MESSAGES[
                    object_name
                ]
            )


    # ======================================
    # EARTH / DEBRIS OBSERVATION
    # ======================================

    if earth_ratio >= 0.30:

        observations.append(
            "Large earth/debris-colored surface area detected."
        )

    elif earth_ratio >= 0.15:

        observations.append(
            "Moderate exposed soil or debris-like surface detected."
        )


    # ======================================
    # EDGE / TEXTURE OBSERVATION
    # ======================================

    if edge_density >= 0.12:

        observations.append(
            "High surface irregularity / edge density detected."
        )

    elif edge_density >= 0.07:

        observations.append(
            "Moderate rough-surface or crack-like texture detected."
        )


    # ======================================
    # FINAL SEVERITY OBSERVATION
    # ======================================

    if severity == "CRITICAL":

        observations.append(
            "Visual evidence indicates very high field-review priority."
        )

    elif severity == "HIGH":

        observations.append(
            "Visual evidence indicates high field-review priority."
        )


    return observations


# ==========================================
# ANALYSE IMAGE
# ==========================================

def analyse_image(
    image_path
):

    """
    Analyse one image and return
    structured computer vision results.
    """

    if not os.path.exists(
        image_path
    ):

        return {

            "success":
                False,

            "message":
                "Image file not found."
        }


    # ======================================
    # READ IMAGE
    # ======================================

    image = cv2.imread(
        image_path
    )


    if image is None:

        return {

            "success":
                False,

            "message":
                "Unable to read image."
        }


    # ======================================
    # YOLO DETECTION
    # ======================================

    try:

        results = model(
            image_path,
            verbose=False
        )

    except Exception as error:

        return {

            "success":
                False,

            "message":
                f"YOLO error: {error}"
        }


    detections = []


    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )


            confidence = float(
                box.conf[0]
            )


            class_name = (
                model.names[
                    class_id
                ]
            )


            detections.append(

                {

                    "object":
                        class_name,

                    "confidence":
                        confidence
                }

            )


    # ======================================
    # IMAGE HEURISTICS
    # ======================================

    earth_ratio = (
        calculate_earth_ratio(
            image
        )
    )


    edge_density = (
        calculate_edge_density(
            image
        )
    )


    # ======================================
    # VISUAL HAZARD SCORE
    # ======================================

    visual_score = (
        calculate_visual_hazard_score(

            detections=
                detections,

            earth_ratio=
                earth_ratio,

            edge_density=
                edge_density
        )
    )


    severity = (
        get_visual_severity(
            visual_score
        )
    )


    # ======================================
    # OBSERVATIONS
    # ======================================

    observations = (
        build_observations(

            detections=
                detections,

            earth_ratio=
                earth_ratio,

            edge_density=
                edge_density,

            severity=
                severity
        )
    )


    # ======================================
    # SAVE ANNOTATED IMAGE
    # ======================================

    annotated_image = (
        results[0].plot()
    )


    output_name = (
        "annotated_"
        +
        os.path.basename(
            image_path
        )
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )


    cv2.imwrite(
        output_path,
        annotated_image
    )


    # ======================================
    # UNIQUE OBJECTS
    # ======================================

    unique_objects = sorted(
        {
            item["object"]
            for item in detections
        }
    )


    # ======================================
    # RETURN RESULT
    # ======================================

    return {

        "success":
            True,

        "detections":
            detections,

        "detected_objects":
            unique_objects,

        "earth_ratio":
            earth_ratio,

        "edge_density":
            edge_density,

        "visual_severity_score":
            visual_score,

        "visual_severity_level":
            severity,

        "observations":
            observations,

        "annotated_image":
            output_path,

        "message":
            "Computer vision analysis completed."
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "ENHANCED COMPUTER VISION TEST"
    )

    print(
        "======================================"
    )


    image_path = input(
        "\nEnter image path: "
    )


    result = analyse_image(
        image_path
    )


    if not result[
        "success"
    ]:

        print(
            "\n❌",
            result[
                "message"
            ]
        )

        raise SystemExit


    print(
        "\n======================================"
    )

    print(
        "VISUAL ANALYSIS RESULT"
    )

    print(
        "======================================"
    )


    print(
        "\nDetected Objects:"
    )


    if result[
        "detections"
    ]:

        for detection in result[
            "detections"
        ]:

            print(

                "-",

                detection[
                    "object"
                ],

                f"({detection['confidence'] * 100:.2f}%)"
            )


    else:

        print(
            "No YOLO objects detected."
        )


    print(
        "\nEarth / Debris Ratio:"
    )

    print(
        f"{result['earth_ratio'] * 100:.2f}%"
    )


    print(
        "\nEdge Density:"
    )

    print(
        f"{result['edge_density'] * 100:.2f}%"
    )


    print(
        "\nVisual Hazard Score:"
    )

    print(
        f"{result['visual_severity_score']}/100"
    )


    print(
        "\nVisual Severity:"
    )

    print(
        result[
            "visual_severity_level"
        ]
    )


    print(
        "\nAI Observations:"
    )


    if result[
        "observations"
    ]:

        for observation in result[
            "observations"
        ]:

            print(
                "-",
                observation
            )


    else:

        print(
            "No major visual hazard cues detected."
        )


    print(
        "\nAnnotated Image:"
    )

    print(
        result[
            "annotated_image"
        ]
    )