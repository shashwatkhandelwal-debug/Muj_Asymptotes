# ============================================================================
# AEGIS-SENTINEL: Multi-Modal Deepfake & Forensic Defense Core
# Local On-Premise Container Deployment (Zero Cloud Dependencies)
# ============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system-level dependencies for OpenCV, PyZBar, Tesseract OCR, FFmpeg, and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libzbar0 \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository source code
COPY . .

# Expose API and frontend static server port
EXPOSE 8000

# Container healthcheck ensuring orchestrator is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Launch the FastAPI orchestrator
CMD ["python", "-m", "uvicorn", "api.orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
