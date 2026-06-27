# Stage 1: Build compiled CUDA extensions
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
# Cap flash-attn's parallel nvcc jobs so the build doesn't exhaust RAM on small
# CI runners (each job needs a few GB). Override with --build-arg MAX_JOBS=N.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-deps git+https://github.com/liampetti/Qwen3-TTS-streaming.git@97da215 && \
    pip install flash-attn --no-build-isolation && \
    python -m spacy download en_core_web_sm

# Stage 2: Runtime (no CUDA compilers/headers)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV HF_HOME=/app/data/models
# HF_HUB_OFFLINE is managed at runtime by app.py (online for the first-run
# wizard download, offline once models are cached) — not pinned here.
# Image variant — the wizard offers all tiers on the GPU image.
ENV FULLOCH_VARIANT=gpu

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
    espeak-ng \
    libportaudio2 \
    libasound2-plugins \
    libpulse0 \
    procps \
    curl \
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
COPY --chown=appuser:appuser app.py fulloch.png parloch.png ./
COPY --chown=appuser:appuser core/ core/
COPY --chown=appuser:appuser tools/ tools/
COPY --chown=appuser:appuser utils/ utils/
COPY --chown=appuser:appuser audio/ audio/
COPY --chown=appuser:appuser server/ server/
COPY --chown=appuser:appuser wav/ wav/

# First-run seeds (copied into the empty ./data volume by core/bootstrap.py):
# the config template the wizard fills in, and the app's own GBNF grammar (which
# the wizard's downloader can't fetch). Weights stay OUT of the image — the
# wizard pulls them on first run.
COPY --chown=appuser:appuser data/config.example.yml /app/seed/config.example.yml
COPY --chown=appuser:appuser data/models/grammars/agent.gbnf /app/seed/grammars/agent.gbnf

# Bootstrap + setup-or-run is handled in app.main() (core/bootstrap.py): an empty
# ./data boots into the setup wizard, a populated one straight to the assistant.
CMD ["python", "app.py"]
