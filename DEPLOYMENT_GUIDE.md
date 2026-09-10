# NER AI Landslide Early Warning System - Deployment Guide

This guide provides step-by-step instructions and architectural analysis for deploying the **NER Landslide Early Warning System MVP**.

---

## 1. System Architecture: Vercel vs Render

### Can we use Vercel for BOTH Frontend and Backend instead of Render?
**YES! In fact, the production deployment is currently running 100% on Vercel.**

Here is how **All-in-One Vercel** compares directly to **Render**:

| Feature / Metric | Vercel (All-in-One Serverless) | Render (Web Service Free Tier) |
| :--- | :--- | :--- |
| **Cost** | **100% Free** (Hobby Tier) | **Free** (with strict monthly hour caps) |
| **Cold Start Latency** | **~1 - 2 seconds** | **50 - 90 seconds** (spins down after 15m idle) |
| **Architecture / Domains** | **Single unified domain** (`ner-landslide-early-warning-system-nine.vercel.app`) | **Two separate domains** (Frontend on Vercel + Backend on Render) |
| **CORS Latency** | **Zero CORS issues** (frontend calls `/api/...` directly) | Requires cross-origin preflight requests (`OPTIONS`) on every API call |
| **Free Quota** | **100,000 requests / day**, 100 GB bandwidth | **750 hours / month** shared across all services |
| **Container Maintenance** | **Zero Docker needed** (Vercel builds Python natively) | Requires Docker container builds and port handling |
| **Global CDN** | Edge CDN with worldwide low latency | Single region (e.g. Oregon or Frankfurt) |

---

### Architecture Diagram: All-in-One Vercel

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   End User Browser                                       │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│             Vercel Cloud Platform (ner-landslide-early-warning-system-nine.vercel.app)   │
│                                                                                          │
│   ┌────────────────────────────────────────┐    ┌────────────────────────────────────┐   │
│   │   Global Edge CDN                      │    │   Python Serverless Runtime        │   │
│   │   - static/index.html                  │    │   - api/index.py                   │   │
│   │   - static/styles.css                  │───▶│   - converted_app.py (FastAPI)     │   │
│   │   - static/app.js                      │    │   - ML inference & risk assessment │   │
│   │   - Leaflet GIS Interactive Map        │    │   - OpenWeather & Groq GenAI       │   │
│   └────────────────────────────────────────┘    └────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                             │ (Model Artifacts Reference)
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │         Hugging Face Model Hub (Free)         │
                     │  https://huggingface.co/Tirthptl/             │
                     │  ner-landslide-models                         │
                     │  - landslide_model_optimized.pkl              │
                     │  - model_features_optimized.pkl               │
                     │  - yolov8n.pt                                 │
                     └───────────────────────────────────────────────┘
```

---

## 2. Pre-Deployment Code Fixes

Before pushing to any cloud git repository or container runtime, four critical adjustments must be made:

### A. Exclude Heavy 3.4 GB Archive from Git
In `.gitignore`, add:
```gitignore
*.zip
landslide-aws.zip
```
> **Warning**: GitHub strictly rejects any push containing files larger than 100 MB.

### B. Fix Linux Path Case Sensitivity
Windows is case-insensitive, but Linux servers (Hugging Face, Docker, Vercel) are case-sensitive:
* **Models Folder**: On disk it is `Models/`. Ensure `src/risk_prediction.py` and `src/computer_vision.py` use `Models/` instead of `models/`.
* **Data Folder**: On disk it is `Data/`. Ensure all scripts use `Data/` instead of `data/`.
* **YOLO Weights**: Move or symlink `yolov8n.pt` into `Models/cv/yolov8n.pt` or update `src/computer_vision.py` to check both locations.

### C. Enable CORS in FastAPI (`converted_app.py`)
To allow the Vercel frontend to query the Hugging Face backend, CORS middleware must be added:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your specific Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### D. Safe Fallback for Large GeoTIFFs (1.5 GB)
In `src/live_risk_prediction.py`, wrap `get_landcover_features()` and `get_terrain_features()` with safe defaults so that live risk predictions succeed even when multi-gigabyte satellite GeoTIFFs are not uploaded to cloud storage.

---

## 3. Step-by-Step Deployment

### Option A: Recommended Architecture (Vercel Frontend + Hugging Face Backend)

#### Step 1: Deploy Model & Backend on Hugging Face Spaces (Free)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click on your profile icon (top right) ➔ **New Space**.
3. Fill in the details:
   * **Space name**: `ner-landslide-api`
   * **License**: `mit` or `open-source`
   * **SDK**: Select **Docker** (Blank)
   * **Hardware**: Choose **CPU basic • 2 vCPU • 16 GB RAM • Free**
   * **Visibility**: **Public**
4. Click **Create Space**.
5. Add your backend files to the Space:
   * Create a `Dockerfile`:
     ```dockerfile
     FROM python:3.11-slim

     WORKDIR /app

     RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         libgl1-mesa-glx \
         libglib2.0-0 \
         libgdal-dev \
         && rm -rf /var/lib/apt/lists/*

     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt

     COPY . .

     ENV PORT=7860
     EXPOSE 7860

     CMD ["uvicorn", "converted_app:app", "--host", "0.0.0.0", "--port", "7860"]
     ```
   * Set Space Secret variables under **Settings ➔ Variables and Secrets**:
     * `GROQ_API_KEY`: Your Groq Cloud API key.
     * `EMAIL_ADDRESS` & `EMAIL_PASSWORD` (Optional for alerts).
6. Once built, Hugging Face provides your public API URL:
   `https://<your-username>-ner-landslide-api.hf.space`

---

#### Step 2: Configure Frontend (`static/app.js`)

1. Open `static/app.js`.
2. Locate line 6:
   ```javascript
   const API = "";
   ```
3. Update it to your Hugging Face Space URL:
   ```javascript
   const API = "https://<your-username>-ner-landslide-api.hf.space";
   ```

---

#### Step 3: Deploy Frontend to Vercel (Free)

1. Push your repository to GitHub (ensure `landslide-aws.zip` is ignored).
2. Log in to [vercel.com](https://vercel.com) using your GitHub account.
3. Click **Add New... ➔ Project**.
4. Import your `NER-Landslide-Early-Warning-System` repository.
5. In **Project Settings**:
   * **Framework Preset**: Select **Other**.
   * **Root Directory**: Select `static` (or leave as `./` with a `vercel.json`).
6. If keeping the root directory as `./`, add a `vercel.json` in the project root:
   ```json
   {
     "version": 2,
     "rewrites": [
       { "source": "/static/(.*)", "destination": "/static/$1" },
       { "source": "/(.*)", "destination": "/static/$1" },
       { "source": "/", "destination": "/static/index.html" }
     ]
   }
   ```
7. Click **Deploy**.
8. In less than a minute, your frontend will be live with a URL like `https://ner-landslide.vercel.app`.

---

### Option B: 1-Click All-in-One Deployment (Hugging Face Spaces)

If you prefer to host both the frontend and backend together under a single URL for zero configuration:
1. Create a Hugging Face Space (Docker runtime).
2. Push the full repository including `static/` and `converted_app.py`.
3. Because `converted_app.py` already serves `static/index.html` at `/`, your entire command center and API are live at a single link with 16 GB RAM and zero cost!

---

## 4. Verification and Health Check

After deployment, test the key endpoints:

1. **Health Check**:
   ```bash
   curl -X GET "https://<your-api-url>/api/health"
   ```
   *Expected Response*: `{"status": "healthy", "service": "NER Landslide Early Warning API"}`

2. **Model Risk Prediction**:
   ```bash
   curl -X POST "https://<your-api-url>/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "state": "Sikkim",
       "city": "Gangtok",
       "rainfall_24h_mm": 45.0,
       "rainfall_3d_mm": 110.0,
       "rainfall_7d_mm": 180.0,
       "soil_water_layer_1": 0.38,
       "soil_water_layer_2": 0.35
     }'
   ```

3. **Field Image Analysis (YOLOv8)**:
   Upload an image through the frontend "Computer Vision Hazard Analysis" panel to verify YOLO detection and crack hazard scoring.

---

## 5. Troubleshooting Checklist

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `CORS Error in Browser` | Backend missing CORS headers | Ensure `CORSMiddleware` with `allow_origins=["*"]` is enabled in `converted_app.py`. |
| `File Not Found: models/...` | Linux case sensitivity | Change path in Python code from `models/` to `Models/`. |
| `GitHub Push Rejected (100MB)` | `landslide-aws.zip` staged | Run `git rm --cached landslide-aws.zip` and add `*.zip` to `.gitignore`. |
| `Terrain / Landcover Missing` | Large GeoTIFFs not uploaded | Implement cloud fallback values in `src/live_risk_prediction.py`. |
| `500 on GenAI Assistant` | Missing Groq API Key | Add `GROQ_API_KEY` to Space Secrets in Hugging Face. |
