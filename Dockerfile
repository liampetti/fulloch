FROM python:3.12-slim

WORKDIR /app

# Point HuggingFace cache at the mounted data volume so from_pretrained()
# finds models downloaded by launch.sh into data/models/hub/
ENV HF_HOME=/app/data/models
ENV HF_HUB_OFFLINE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    libsox-fmt-all \
    ffmpeg \
    git \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Force transformers 4.57.3 for offline Qwen3 ASR to work)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-deps git+https://github.com/rekuenkdr/Qwen3-TTS-streaming.git@97da215

# Copy application code
COPY app.py .
COPY core/ core/
COPY tools/ tools/
COPY utils/ utils/
COPY audio/ audio/

# Run the app
CMD ["python", "app.py"]
