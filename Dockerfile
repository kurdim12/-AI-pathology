# Naseej demo image. CPU-only by default so it runs anywhere; for training use a
# CUDA base image and the GPU torch wheels instead.
FROM python:3.11-slim

# System libs Pillow / OpenCV-style image IO may need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU torch wheels first (smaller, no CUDA), then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        numpy pillow scikit-learn matplotlib gradio tqdm

COPY . .

# Gradio default port.
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

# Launch the booth demo. Mount a trained checkpoint at
# /app/checkpoints/best_model.pt to serve real predictions.
CMD ["python", "-m", "app.app"]
