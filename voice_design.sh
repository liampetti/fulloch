#!/bin/bash
set -e

# Voice Designer — generate a Fulloch voice-clone reference pair from a
# natural-language description using Qwen3-TTS-12Hz-1.7B-VoiceDesign.
#
# Bootstraps deps + model download here, then exec's into the Python
# helper which owns the actual interactive prompt → generate → play →
# save loop (so the GPU model loads once, not per attempt).

BASE_DIR="$(pwd)/data/models"
HUB_DIR="$BASE_DIR/hub"
VOICE_DESIGN_DIR="$HUB_DIR/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign"

echo "🎨 Fulloch Voice Designer"
echo ""

# 1. Pick a Python — prefer the repo's .venv to match runtime deps.
if [ -x "./.venv/bin/python" ]; then
    PYTHON="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "❌ No Python found. Activate the project's .venv or install python3 followed by pip install -r requirements.txt"
    exit 1
fi
echo "🐍 Using $PYTHON"

# 2. Make sure huggingface CLI is around (used to fetch the model).
if ! command -v hf >/dev/null 2>&1; then
    echo "⬇️ hf CLI not found. Installing via standalone installer..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "❌ curl is required to install the hf CLI. Install curl first."
        exit 1
    fi
    curl -LsSf https://hf.co/cli/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v hf >/dev/null 2>&1; then
        echo "❌ hf still not found after install. Re-run ./launch.sh once, then try again."
        exit 1
    fi
fi

# 3. Ensure model cache layout exists.
mkdir -p "$HUB_DIR"

# 4. Download VoiceDesign model if it's not already cached.
if [ ! -d "$VOICE_DESIGN_DIR" ]; then
    echo "⬇️  Qwen3-TTS-12Hz-1.7B-VoiceDesign not found in $HUB_DIR."
    read -p "Download it now? (~3.4GB) (Y/n): " response
    response=${response,,}
    if [[ "$response" == "n" || "$response" == "no" ]]; then
        echo "Aborted — model is required."
        exit 0
    fi
    hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --cache-dir "$HUB_DIR"
else
    echo "✅ VoiceDesign model present."
fi

# 5. Hand off to the Python helper which drives the interactive loop.
echo ""
exec "$PYTHON" scripts/voice_design.py
