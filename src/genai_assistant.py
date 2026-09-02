"""
genai_assistant.py
------------------

GenAI Assistant for NER Landslide Early Warning System.

Uses Groq LLM to explain:
- Landslide risk
- Weather conditions
- Terrain conditions
- Citizen reports
- Recommended actions

Run:
    python src/genai_assistant.py
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# CHECK API KEY
# ==========================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


if not GROQ_API_KEY:

    print(
        "❌ GROQ_API_KEY missing in .env"
    )

    raise SystemExit


# ==========================================
# MODEL
# ==========================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are an AI assistant for a landslide early warning
and risk monitoring system for North-East India.

Your role is to:

1. Explain landslide risk in simple language.
2. Explain the impact of rainfall, soil moisture,
   slope, elevation and land cover.
3. Recommend practical monitoring and preparedness actions.
4. Summarize citizen or field officer reports.
5. Help authorities understand model results.
6. Never claim that the model prediction is a guaranteed
   real-world landslide event.
7. Always describe the output as model-estimated risk
   or landslide susceptibility.
8. If the information is incomplete, clearly say so.

Keep answers clear, practical and professional.
"""


# ==========================================
# ASK ASSISTANT
# ==========================================

def ask_landslide_assistant(
    question,
    context=None
):

    if not question.strip():

        return (
            "Please enter a question."
        )


    # ======================================
    # BUILD PROMPT
    # ======================================

    if context:

        prompt = f"""
{SYSTEM_PROMPT}

CURRENT SYSTEM CONTEXT:

{context}

USER QUESTION:

{question}

Give a clear and useful answer based on the
available system context.
"""

    else:

        prompt = f"""
{SYSTEM_PROMPT}

USER QUESTION:

{question}

Give a clear and useful answer.
"""


    # ======================================
    # MODEL RESPONSE
    # ======================================

    try:

        response = model.invoke(
            prompt
        )


        return response.content


    except Exception as error:

        return (
            f"Assistant error: {error}"
        )


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "NER LANDSLIDE GENAI ASSISTANT"
    )

    print(
        "======================================"
    )


    question = input(
        "\nAsk a question: "
    )


    answer = ask_landslide_assistant(
        question
    )


    print(
        "\nAI Assistant:"
    )

    print(
        answer
    )