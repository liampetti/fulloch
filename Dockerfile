# Stage 1: Build compiled CUDA extensions
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-deps git+https://github.com/rekuenkdr/Qwen3-TTS-streaming.git@97da215 && \
    pip install flash-attn --no-build-isolation

# Stage 2: Runtime (no CUDA compilers/headers)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV HF_HOME=/app/data/models
ENV HF_HUB_OFFLINE=1

# Install runtime system dependencies
# - gcc: needed by Triton JIT for TTS kernels
# - libasound2-plugins + libpulse0: ALSA→PulseAudio bridge so PortAudio's ALSA
#   backend transparently routes through pulseaudio (handles resampling +
#   device selection regardless of what mic is plugged in)
RUN apt-get update && apt-get install -y \
    gcc \
    sox \
    libsox-dev \
    libsox-fmt-all \
    ffmpeg \
    libportaudio2 \
    libasound2-plugins \
    libpulse0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Route ALSA's default device through PulseAudio so the host's selected
# input/output devices and resampling are used automatically
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

# Copy Python environment with compiled packages from builder
COPY --from=builder /opt/conda /opt/conda

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser app.py fulloch.png ./
COPY --chown=appuser:appuser core/ core/
COPY --chown=appuser:appuser tools/ tools/
COPY --chown=appuser:appuser utils/ utils/
COPY --chown=appuser:appuser audio/ audio/
COPY --chown=appuser:appuser server/ server/
COPY --chown=appuser:appuser wav/ wav/

CMD ["python", "app.py"]
