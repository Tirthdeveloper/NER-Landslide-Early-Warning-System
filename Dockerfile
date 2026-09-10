FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and Rasterio / GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements_web.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "converted_app:app", "--host", "0.0.0.0", "--port", "7860"]
