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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install Python deps (from cog-dia requirements.txt + runpod SDK)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt runpod soundfile numpy

# Copy model code and predictor
COPY dia/ dia/
COPY predict.py .
COPY handler.py .

# Download model weights at build time for fast cold starts.
# The predict.py setup() uses pget (Replicate's downloader) which won't be
# available here. We download from HuggingFace instead.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('nari-labs/Dia-1.6B', local_dir='/app/model_cache/nari-labs/Dia-1.6B')\
"

# Set model cache env so predict.py can find pre-downloaded weights
ENV DIA_MODEL_DIR=/app/model_cache/nari-labs/Dia-1.6B

CMD ["python3", "-u", "handler.py"]
