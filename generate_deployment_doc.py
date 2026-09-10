import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.2)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(0x24, 0x29, 0x2E)
    # Add a border / background box
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F6F8FA")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    cp = cell.paragraphs[0]
    cp.paragraph_format.space_before = Pt(2)
    cp.paragraph_format.space_after = Pt(2)
    crun = cp.add_run(text)
    crun.font.name = 'Consolas'
    crun.font.size = Pt(9.5)
    crun.font.color.rgb = RGBColor(0x1F, 0x23, 0x28)
    doc.element.body.remove(p._p)

def build_docx(output_path):
    doc = Document()

    # Document Title
    title = doc.add_heading(level=0)
    trun = title.add_run("NER Landslide Early Warning System\nDeployment & Hosting Guide")
    trun.font.name = 'Arial'
    trun.font.size = Pt(24)
    trun.font.bold = True
    trun.font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)

    subtitle = doc.add_paragraph()
    srun = subtitle.add_run("Step-by-Step Production & MVP Deployment: Vercel Frontend + Free Model Hosting")
    srun.font.name = 'Arial'
    srun.font.size = Pt(12)
    srun.font.italic = True
    srun.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph() # Spacer

    # Section 1: Executive Summary & System Architecture
    h1 = doc.add_heading("1. Executive Summary & Architecture", level=1)
    h1.runs[0].font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)

    p = doc.add_paragraph("The NER Landslide Early Warning System is an end-to-end intelligent disaster preparedness application tailored for North Eastern India. It incorporates a single-page interactive GIS dashboard, live weather intelligence, machine learning risk assessment (XGBoost), Computer Vision hazard evaluation (YOLOv8), and an emergency GenAI conversational assistant.")
    
    doc.add_paragraph("For production deployment, an optimal Decoupled Architecture is recommended:")

    # Table of Tiers
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Shading Accent 1'
    hdr = table.rows[0].cells
    hdr[0].text = "Component"
    hdr[1].text = "Recommended Platform"
    hdr[2].text = "Specifications & Cost"
    for c in hdr:
        set_cell_background(c, "0F4C81")
        for p_elem in c.paragraphs:
            for r in p_elem.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.bold = True

    data = [
        ("Frontend (HTML/CSS/JS + Leaflet)", "Vercel", "Free forever, Global CDN, SSL, Custom Domain"),
        ("Backend & ML Engine (FastAPI + XGBoost + YOLOv8)", "Hugging Face Spaces", "Free 16 GB RAM, 2 vCPUs, 50 GB Disk (Docker/Python)"),
        ("GenAI Assistant API", "Groq Cloud", "Free Tier (High-speed LLaMA/Mixtral inference)")
    ]

    for row_data in data:
        row = table.add_row()
        for idx, text in enumerate(row_data):
            cell = row.cells[idx]
            cell.text = text
            set_cell_background(cell, "F7F9FB" if len(table.rows) % 2 == 0 else "FFFFFF")

    doc.add_paragraph()

    # Section 2: Why Not Vercel for ML?
    h2 = doc.add_heading("2. Analysis: Why Deploy Model on Hugging Face instead of Vercel?", level=1)
    h2.runs[0].font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)

    p2 = doc.add_paragraph("While Vercel is best-in-class for static websites and simple serverless micro-APIs, it cannot effectively run this machine learning stack due to serverless constraints:")
    
    b1 = doc.add_paragraph(style='List Bullet')
    b1.add_run("Bundle Size Ceiling: ").bold = True
    b1.add_run("Vercel limits serverless lambdas to 250 MB uncompressed. PyTorch, Torchvision, and Ultralytics (YOLO) alone require over 1.2 GB.")
    
    b2 = doc.add_paragraph(style='List Bullet')
    b2.add_run("Native C Extensions: ").bold = True
    b2.add_run("Geospatial packages like rasterio require complex GDAL/GEOS system binaries that frequently break on AWS Lambda runtimes.")

    b3 = doc.add_paragraph(style='List Bullet')
    b3.add_run("Execution Timeouts: ").bold = True
    b3.add_run("Vercel's free Hobby plan enforces a 10-15s timeout, which can trigger gateway 504 errors on image analysis or heavy raster queries.")

    b4 = doc.add_paragraph(style='List Bullet')
    b4.add_run("Read-Only Filesystem: ").bold = True
    b4.add_run("Local writes to Data/citizen_reports/ are blocked on serverless architectures.")

    doc.add_paragraph("In contrast, Hugging Face Spaces offers 16 GB of dedicated RAM, 2 vCPUs, persistent storage, and full Docker capability at zero cost.")

    # Section 3: Pre-Deployment Code Fixes
    h3 = doc.add_heading("3. Mandatory Pre-Deployment Codebase Fixes", level=1)
    h3.runs[0].font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)

    doc.add_paragraph("Before pushing your project to GitHub or Hugging Face, apply these crucial fixes:")

    p_fix1 = doc.add_paragraph()
    p_fix1.add_run("Fix 1: Add landslide-aws.zip to .gitignore\n").bold = True
    p_fix1.add_run("The root folder contains a 3.44 GB archive (landslide-aws.zip). GitHub will block git push if files exceed 100 MB. Add this to .gitignore:")
    add_code_block(doc, "*.zip\nlandslide-aws.zip")

    p_fix2 = doc.add_paragraph()
    p_fix2.add_run("Fix 2: Correct Linux File Path Casing\n").bold = True
    p_fix2.add_run("Linux containers are case-sensitive. Update python imports and file constants:")
    add_code_block(doc, "# In src/risk_prediction.py:\nMODEL_FILE = 'Models/landslide_model_optimized.pkl'\nFEATURE_FILE = 'Models/model_features_optimized.pkl'\n\n# In src/live_risk_prediction.py:\nDEM_FILE = 'Data/raw/dem/ner_dem_90m.tiff'")

    p_fix3 = doc.add_paragraph()
    p_fix3.add_run("Fix 3: Enable CORS in converted_app.py\n").bold = True
    p_fix3.add_run("When the Vercel frontend contacts the Hugging Face API, browser security blocks it unless CORS is allowed. Insert this into converted_app.py:")
    add_code_block(doc, "from fastapi.middleware.cors import CORSMiddleware\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['*'],\n    allow_credentials=True,\n    allow_methods=['*'],\n    allow_headers=['*'],\n)")

    # Section 4: Step-by-Step Deployment Walkthrough
    h4 = doc.add_heading("4. Step-by-Step Deployment Instructions", level=1)
    h4.runs[0].font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)

    # Substep A
    h4_a = doc.add_heading("Phase 1: Deploy Backend & Model on Hugging Face Spaces (Free)", level=2)
    p_a1 = doc.add_paragraph("1. Sign up or log in at https://huggingface.co\n2. Click New Space (or visit https://huggingface.co/new-space).\n3. Set Space Name: ner-landslide-api\n4. License: MIT or Apache-2.0\n5. Space SDK: Select Docker (Blank)\n6. Space Hardware: Choose CPU basic • 2 vCPU • 16 GB RAM • Free\n7. Set Space Visibility to Public, then click Create Space.")

    doc.add_paragraph("8. Create a Dockerfile in your repository root with the following configuration:")
    add_code_block(doc, """FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libgdal-dev \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "converted_app:app", "--host", "0.0.0.0", "--port", "7860"]""")

    doc.add_paragraph("9. In Space Settings -> Variables and Secrets, add your secret keys:\n   • GROQ_API_KEY: Your Groq Cloud API Key\n   • TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN (Optional)\n10. Once built, copy your public endpoint URL: https://<username>-ner-landslide-api.hf.space")

    # Substep B
    h4_b = doc.add_heading("Phase 2: Configure Frontend (static/app.js)", level=2)
    doc.add_paragraph("1. Open static/app.js\n2. At Line 6, replace the empty API string with your new Hugging Face Space URL:")
    add_code_block(doc, 'const API = "https://<your-username>-ner-landslide-api.hf.space";')

    # Substep C
    h4_c = doc.add_heading("Phase 3: Deploy Frontend to Vercel (Free)", level=2)
    doc.add_paragraph("1. Commit and push your code to GitHub (https://github.com/Tirthdeveloper/NER-Landslide-Early-Warning-System.git).\n2. Navigate to https://vercel.com and log in with GitHub.\n3. Click 'Add New...' -> 'Project'.\n4. Select your NER repository.\n5. Under 'Root Directory', choose 'static' (or keep './' with a vercel.json rewrite file).\n6. Click 'Deploy'.\n7. Within 30 seconds, your site is published with a global URL (e.g. https://ner-landslide.vercel.app).")

    # Section 5: Verification & Testing
    h5 = doc.add_heading("5. Verification & Testing Checklist", level=1)
    h5.runs[0].font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)

    t_check = doc.add_table(rows=1, cols=3)
    t_check.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_check.style = 'Light Shading Accent 1'
    chdr = t_check.rows[0].cells
    chdr[0].text = "Feature / Route"
    chdr[1].text = "Verification Method"
    chdr[2].text = "Expected Result"
    for c in chdr:
        set_cell_background(c, "0F4C81")
        for p_elem in c.paragraphs:
            for r in p_elem.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.bold = True

    tests = [
        ("GET /api/health", "curl -X GET https://<api-url>/api/health", "Returns status: healthy"),
        ("POST /api/predict", "Submit risk evaluation with rainfall & soil water values", "Returns risk_score (0-100), risk_level, and emergency priority"),
        ("POST /api/cv/analyse", "Upload field photo via Computer Vision tab", "Returns YOLO object counts, visual hazard score, and annotated image"),
        ("POST /api/genai", "Ask question in Disaster Assistant chatbox", "Returns streamed/markdown response from Groq LLM"),
        ("GET /api/gis/points", "View GIS Risk Map tab in dashboard", "Populates interactive Leaflet map with geocoded event pins")
    ]

    for t_data in tests:
        row = t_check.add_row()
        for idx, text in enumerate(t_data):
            cell = row.cells[idx]
            cell.text = text
            set_cell_background(cell, "F7F9FB" if len(t_check.rows) % 2 == 0 else "FFFFFF")

    doc.add_paragraph()

    # Section 6: Alternative 1-Click HF Deployment
    h6 = doc.add_heading("6. Bonus: 1-Click All-in-One Deployment (Easiest)", level=1)
    h6.runs[0].font.color.rgb = RGBColor(0x0F, 0x4C, 0x81)
    doc.add_paragraph("Because converted_app.py already mounts the static/ directory and serves index.html at root (/), you can also run the entire project in a single Hugging Face Space without needing Vercel separately!")
    doc.add_paragraph("Simply deploy the repository to Hugging Face Spaces using Docker, and your UI and API will both run together at https://<username>-ner-landslide.hf.space with zero configuration.")

    # Save
    doc.save(output_path)
    print(f"Successfully generated DOCX at {output_path}")

if __name__ == "__main__":
    output_docx = r"c:\Gen_AI_series11\app1\NER_Landslide_System_Deployment_Guide.docx"
    build_docx(output_docx)
