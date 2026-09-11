FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DOWNLOAD_PRETRAINED=0

WORKDIR /app

# Install essential Linux system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libsndfile1 \
    ffmpeg \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch & torchvision first to prevent CUDA package bloat
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Expose port
EXPOSE 8000

# Start Uvicorn bound to dynamic Render $PORT
CMD ["sh", "-c", "uvicorn api.orchestrator:app --host 0.0.0.0 --port ${PORT:-8000}"]
