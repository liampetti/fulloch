# Stage 0: Build the Obsidian plugin bundle + downloadable zip from source, so
# the dashboard always serves what's in obsidian-plugin/main.ts rather than a
# stale hand-built artifact.
FROM node:20-slim AS obsidian-plugin-builder
WORKDIR /plugin
RUN apt-get update && apt-get install -y zip && rm -rf /var/lib/apt/lists/*
COPY obsidian-plugin/package.json obsidian-plugin/package-lock.json ./
RUN npm ci
COPY obsidian-plugin/main.ts obsidian-plugin/manifest.json obsidian-plugin/esbuild.config.mjs obsidian-plugin/tsconfig.json ./
RUN npm run build && zip fulloch-obsidian.zip manifest.json main.js

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

# First-run seeds (copied into the empty ./data volume by core/bootstrap.py):
# the config template the wizard fills in, the app's own GBNF grammar (which
# the wizard's downloader can't fetch), and the timer alert tone. Weights stay
# OUT of the image — the wizard pulls them on first run.
COPY --chown=appuser:appuser data/config.example.yml /app/seed/config.example.yml
COPY --chown=appuser:appuser data/models/grammars/agent.gbnf /app/seed/grammars/agent.gbnf
COPY --chown=appuser:appuser data/wav/ /app/seed/wav/

# Obsidian plugin zip, built from source in the obsidian-plugin-builder stage
# above — served by GET /api/obsidian/plugin.zip (server/dashboard.py).
COPY --from=obsidian-plugin-builder --chown=appuser:appuser /plugin/fulloch-obsidian.zip /app/obsidian-plugin/fulloch-obsidian.zip

# Entrypoint chowns the data dir to appuser on every boot — needed because
# Docker-named volumes start root-owned, and a fresh volume would otherwise
# fail the app's first write to /app/data. Idempotent (no-op when the dir
# is already correctly owned) and safe on read-only mounts.
COPY --chown=root:root --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Liveness probe: the dashboard's GET /ready returns 200 as soon as the
# web server is up. A user who hits the URL right now sees the wizard,
# download progress, or an actionable error — never connection-refused.
# `docker compose ps` flips this to "healthy" the moment the UI is
# reachable. Task 7a of docs/ease-of-use-tasks.md.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=2).status == 200 else 1)"
CMD ["python", "app.py"]
