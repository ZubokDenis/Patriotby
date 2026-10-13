FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    openssh-client \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    opencv-python-headless \
    qrcode \
    gradio_client \
    "huggingface_hub<1.0" \
    "diffusers==0.31.0" \
    "transformers==4.46.3" \
    "accelerate==1.1.1" \
    safetensors \
    "peft==0.13.2" \
    numpy

EXPOSE 8000

CMD ["python", "-u", "watchdog.py"]
