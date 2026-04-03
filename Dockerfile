FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps matching cog.yaml
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install pget (Replicate's fast downloader — predict.py uses it)
RUN curl -o /usr/local/bin/pget -L \
    "https://github.com/replicate/pget/releases/latest/download/pget_Linux_x86_64" && \
    chmod +x /usr/local/bin/pget

WORKDIR /app

# Set HF cache to model_cache (predict.py reads this)
ENV HF_HOME=/app/model_cache
ENV TORCH_HOME=/app/model_cache
ENV HF_DATASETS_CACHE=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HUGGINGFACE_HUB_CACHE=/app/model_cache

# Install Python deps
# - requirements.txt from cog-dia repo
# - cog: predict.py imports BasePredictor, Input, Path from it
# - runpod: serverless handler SDK
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt cog runpod

# Copy everything
COPY dia/ dia/
COPY predict.py .
COPY handler.py .

# Download model weights at build time using pget (same as predict.py does).
# This bakes weights into the image for fast cold starts.
RUN mkdir -p /app/model_cache && \
    python3 -c "\
import os; \
os.makedirs('model_cache', exist_ok=True); \
from predict import download_weights; \
url = 'https://weights.replicate.delivery/default/dia/model_cache/models--nari-labs--Dia-1.6B-0626.tar'; \
dest = 'model_cache/models--nari-labs--Dia-1.6B-0626.tar'; \
download_weights(url, dest) \
"

CMD ["python3", "-u", "handler.py"]
