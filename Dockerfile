FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    libsox-fmt-all \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
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
