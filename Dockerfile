# Stage 1: Build compiled CUDA extensions
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
# Cap flash-attn's parallel nvcc jobs so the build doesn't exhaust RAM on small
# CI runners (each job needs a few GB). Override with --build-arg MAX_JOBS=N.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}
ARG HIGGS_TTS_REPO=https://github.com/liampetti/HiggsTTS.cpp.git
ARG HIGGS_TTS_REF=061134ad8c3dc544ebf1362f12882fd26a879f8f
# llama.cpp b10199: pinned local-LLM runtime revision.
ARG LLAMA_CPP_REF=b4ca032ae3729516943884786de4ae39fba0bbca

RUN apt-get update && apt-get install -y git cmake && rm -rf /var/lib/apt/lists/*

# HiggsTTS.cpp runs in its own process. Clone the Fulloch fork at an immutable
# commit so images are reproducible without vendoring its ggml runtime.
RUN git clone "${HIGGS_TTS_REPO}" /tmp/higgs-tts && \
    git -C /tmp/higgs-tts checkout --detach "${HIGGS_TTS_REF}" && \
    test "$(git -C /tmp/higgs-tts rev-parse HEAD)" = "${HIGGS_TTS_REF}" && \
    cmake -S /tmp/higgs-tts -B /tmp/higgs-tts/build -DGGML_CUDA=ON && \
    cmake --build /tmp/higgs-tts/build --target higgs_server -j "${MAX_JOBS}" && \
    mkdir -p /opt/higgs-tts && \
    cp /tmp/higgs-tts/build/bin/higgs_server /opt/higgs-tts/ && \
    cp /tmp/higgs-tts/build/ggml/src/libggml.so.0.* /opt/higgs-tts/libggml.so.0 && \
    cp /tmp/higgs-tts/build/ggml/src/libggml-base.so.0.* /opt/higgs-tts/libggml-base.so.0 && \
    cp /tmp/higgs-tts/build/ggml/src/libggml-cpu.so.0.* /opt/higgs-tts/libggml-cpu.so.0 && \
    cp /tmp/higgs-tts/build/ggml/src/ggml-cuda/libggml-cuda.so.0.* /opt/higgs-tts/libggml-cuda.so.0 && \
    rm -rf /tmp/higgs-tts

# All local GGUF models run through this matching upstream llama-server build.
# Release images must not inherit the CI builder's CPU instruction set: a native
# binary can SIGILL when users run it on a different x86-64 processor.
RUN git init /tmp/llama.cpp && \
    git -C /tmp/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git && \
    git -C /tmp/llama.cpp fetch --depth 1 origin "${LLAMA_CPP_REF}" && \
    git -C /tmp/llama.cpp checkout --detach FETCH_HEAD && \
    cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_NATIVE=OFF && \
    cmake --build /tmp/llama.cpp/build --target llama-server -j "${MAX_JOBS}" && \
    mkdir -p /opt/llama-cpp && \
    cp /tmp/llama.cpp/build/bin/llama-server /opt/llama-cpp/ && \
    rm -rf /tmp/llama.cpp

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-deps git+https://github.com/liampetti/Qwen3-TTS-streaming.git@97da215 && \
    pip install flash-attn --no-build-isolation && \
    python -m spacy download en_core_web_sm

# openWakeWord's wheel excludes its CC BY-NC-SA 4.0 feature extractors. They
# are vendored with their attribution notice, avoiding an external build-time
# download while excluding unrelated pre-trained wakeword classifiers.
COPY third_party/openwakeword-models/ /tmp/openwakeword-models/
RUN set -eux; \
    target="$(python -c 'from pathlib import Path; import openwakeword; target = Path(openwakeword.__file__).parent / "resources" / "models"; target.mkdir(parents=True, exist_ok=True); print(target)')"; \
    cp /tmp/openwakeword-models/*.tflite /tmp/openwakeword-models/*.onnx "$target/"

# Stage 2: Runtime (no CUDA compilers/headers)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ARG CRISPASR_RELEASE_TAG=v0.8.7

WORKDIR /app

ENV HF_HOME=/app/data/models
# HF_HUB_OFFLINE is managed at runtime by app.py (online for the first-run
# wizard download, offline once models are cached) — not pinned here.
# Image variant — the wizard offers all tiers on the GPU image.
ENV FULLOCH_VARIANT=gpu
# Native llama-server is separate from Python, so it does not receive PyTorch's
# internal CUDA-library discovery. Make the CUDA wheels' shared libraries visible
# to its dynamic linker as well.
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

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

# CrispASR publishes its Python glue and CUDA native runtime separately. Keep
# the assembled runtime outside /app/data so setup only downloads model weights.
RUN set -eux; \
    curl -L "https://github.com/CrispStrobe/CrispASR/releases/download/${CRISPASR_RELEASE_TAG}/crispasr-python-linux-x86_64.tar.gz" -o /tmp/crispasr-python.tar.gz; \
    curl -L "https://github.com/CrispStrobe/CrispASR/releases/download/${CRISPASR_RELEASE_TAG}/libcrispasr-linux-x86_64-cuda.tar.gz" -o /tmp/crispasr-cuda.tar.gz; \
    tar -xzf /tmp/crispasr-python.tar.gz -C /tmp; \
    mv /tmp/crispasr-python-linux-x86_64 /opt/crispasr-python-cuda; \
    tar -xzf /tmp/crispasr-cuda.tar.gz -C /tmp; \
    rm -f /opt/crispasr-python-cuda/crispasr/libcrispasr.so /opt/crispasr-python-cuda/crispasr/libggml*.so*; \
    cp /tmp/libcrispasr-linux-x86_64-cuda/src/libcrispasr.so.0.8.7 /opt/crispasr-python-cuda/crispasr/libcrispasr.so; \
    cp /tmp/libcrispasr-linux-x86_64-cuda/ggml/src/libggml.so.0.10.2 /opt/crispasr-python-cuda/crispasr/libggml.so.0; \
    cp /tmp/libcrispasr-linux-x86_64-cuda/ggml/src/libggml-cpu.so.0.10.2 /opt/crispasr-python-cuda/crispasr/libggml-cpu.so.0; \
    cp /tmp/libcrispasr-linux-x86_64-cuda/ggml/src/libggml-base.so.0.10.2 /opt/crispasr-python-cuda/crispasr/libggml-base.so.0; \
    cp /tmp/libcrispasr-linux-x86_64-cuda/ggml/src/libggml-cuda.so.0.10.2 /opt/crispasr-python-cuda/crispasr/libggml-cuda.so.0; \
    rm -rf /tmp/crispasr-python.tar.gz /tmp/crispasr-cuda.tar.gz /tmp/libcrispasr-linux-x86_64-cuda

# Route ALSA's default device through PulseAudio so the host's selected
# input/output devices and resampling are used automatically
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

# Copy Python environment with compiled packages from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/higgs-tts /opt/higgs-tts
COPY --from=builder /opt/llama-cpp /opt/llama-cpp

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
# Retain attribution for the vendored CC BY-NC-SA 4.0 feature extractors in
# the published runtime image.
COPY --chown=appuser:appuser third_party/openwakeword-models/NOTICE.md /app/THIRD_PARTY_NOTICES/openwakeword-models.md

# First-run seeds (copied into the empty ./data volume by core/bootstrap.py):
# the config template the wizard fills in, the app's own GBNF grammar (which
# the wizard's downloader can't fetch), the default wakeword classifier, and
# the timer alert tone. Downloadable model weights stay out of the image.
COPY --chown=appuser:appuser data/config.example.yml /app/seed/config.example.yml
COPY --chown=appuser:appuser data/models/grammars/agent.gbnf /app/seed/grammars/agent.gbnf
COPY --chown=appuser:appuser data/models/wakeword/hey_atticus_v0.5.onnx /app/seed/wakeword/hey_atticus_v0.5.onnx
COPY --chown=appuser:appuser data/wav/ /app/seed/wav/
COPY --chown=appuser:appuser data/voices/ /app/seed/voices/
COPY --chown=appuser:appuser data/fulloch-obsidian-plugin.zip /app/seed/fulloch-obsidian-plugin.zip

# Entrypoint chowns the data dir to appuser on every boot — needed because
# Docker-named volumes start root-owned, and a fresh volume would otherwise
# fail the app's first write to /app/data. Idempotent (no-op when the dir
# is already correctly owned) and safe on read-only mounts.
COPY --chown=root:root --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# Start the entrypoint as root so it can repair a fresh root-owned volume; it
# drops to appuser before launching Fulloch.
USER root
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Liveness probe: the dashboard's GET /ready returns 200 as soon as the
# web server is up. A user who hits the URL right now sees the wizard,
# download progress, or an actionable error — never connection-refused.
# Docker marks this "healthy" the moment the UI is reachable. Read the live
# config because users can change the dashboard port.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import ssl,sys,urllib.request,yaml; cfg=yaml.safe_load(open('/app/data/config.yml')) or {}; general=cfg.get('general') or {}; port=general.get('dashboard_port', 8765); scheme='https' if general.get('dashboard_ssl_certfile') and general.get('dashboard_ssl_keyfile') else 'http'; opts={'context': ssl._create_unverified_context()} if scheme == 'https' else {}; sys.exit(0 if urllib.request.urlopen(f'{scheme}://127.0.0.1:{port}/ready', timeout=2, **opts).status == 200 else 1)"
CMD ["python", "app.py"]
