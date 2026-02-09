FROM python:3.12-slim

WORKDIR /app

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
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Force transformers 4.57.3 for offline Qwen3 ASR to work)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-deps git+https://github.com/rekuenkdr/Qwen3-TTS-streaming.git@97da215

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser core/ core/
COPY --chown=appuser:appuser tools/ tools/
COPY --chown=appuser:appuser utils/ utils/
COPY --chown=appuser:appuser audio/ audio/

# Run the app
CMD ["python", "app.py"]
