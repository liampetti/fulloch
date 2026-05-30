#!/bin/bash
set -e

# Directory definitions
BASE_DIR="$(pwd)/data/models"
GRAMMAR_DIR="$BASE_DIR/grammars"
HUB_DIR="$BASE_DIR/hub"

# Model specific variables
QWEN_REPO="unsloth/Qwen3.5-9B-GGUF"
QWEN_FILE="Qwen3.5-9B-Q5_K_M.gguf"

# Define the cache folders
ASR_DIR="$HUB_DIR/models--Qwen--Qwen3-ASR-1.7B"
TTS_DIR="$HUB_DIR/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"

# 1. Ensure Dependencies are installed
echo "🔍 Checking dependencies..."

# Check for curl
if ! command -v curl &> /dev/null; then
    echo "❌ curl not found. Please install curl first."
    echo "   e.g. sudo apt install curl"
    exit 1
fi

# Check for wget
if ! command -v wget &> /dev/null; then
    echo "❌ wget not found. Please install wget first."
    echo "   e.g. sudo apt install wget"
    exit 1
fi

# Check for docker and docker compose
if ! command -v docker &> /dev/null; then
    echo "❌ docker not found. Please install Docker first."
    echo "   See https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ docker compose not found. Please install the Docker Compose plugin."
    echo "   See https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for hf (install via standalone installer if missing)
if ! command -v hf &> /dev/null; then
    echo "⬇️ hf not found. Installing via standalone installer..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    # Source updated PATH so hf is available in this session
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v hf &> /dev/null; then
        echo "❌ hf still not found after install."
        echo "   Try running: curl -LsSf https://hf.co/cli/install.sh | bash"
        echo "   Then restart your shell and run ./launch.sh again."
        exit 1
    fi
fi

echo "✅ All dependencies found."

# 2. Create Directory Structure
echo "📂 Checking directory structure..."
mkdir -p "$HUB_DIR" "$GRAMMAR_DIR"

# 2a. Check for config.yml and .env, create from templates if missing
CONFIG_FILE="$(pwd)/data/config.yml"
CONFIG_EXAMPLE="$(pwd)/data/config.example.yml"
ENV_FILE="$(pwd)/.env"
ENV_EXAMPLE="$(pwd)/.env.example"

CREATED_FILES=()

if [ ! -f "$CONFIG_FILE" ]; then
    echo "📝 config.yml not found. Creating from template..."
    cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
    CREATED_FILES+=("data/config.yml")
else
    echo "✅ config.yml exists."
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "📝 .env not found. Creating from template..."
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    CREATED_FILES+=(".env")
else
    echo "✅ .env exists."
fi

if [ ${#CREATED_FILES[@]} -gt 0 ]; then
    echo ""
    echo "📄 Created: ${CREATED_FILES[*]}"
    echo ""
    read -p "Continue with defaults or exit to edit these files first? (c)ontinue / (e)xit: " response
    response=${response,,}
    if [[ "$response" == "e" || "$response" == "exit" ]]; then
        echo ""
        echo "   Edit the files and run ./launch.sh again when ready."
        exit 0
    fi
    echo "✅ Continuing with defaults."
fi

# 3. Check and Download json.gbnf
if [ ! -f "$GRAMMAR_DIR/json.gbnf" ]; then
    echo "⬇️ Downloading json.gbnf (grammar file)..."
    wget -q --show-progress -O "$GRAMMAR_DIR/json.gbnf" \
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/json.gbnf"
else
    echo "✅ json.gbnf exists."
fi

# 4. Check and Download Qwen3 GGUF
if [ ! -f "$BASE_DIR/$QWEN_FILE" ]; then
    echo "⬇️ Downloading Qwen3.5 9B SLM (6.6GB)..."
    hf download "$QWEN_REPO" "$QWEN_FILE" \
        --local-dir "$BASE_DIR"
else
    echo "✅ $QWEN_FILE exists."
fi

# 5. Check and Download Qwen3 TTS model (Base variant — voice cloning,
# conditioned on the reference pair named in general.voice_clone)
if [ ! -d "$TTS_DIR" ]; then
    echo "⬇️ Downloading Qwen3 TTS Base (3.4GB)..."
    hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --cache-dir "$HUB_DIR"
else
    echo "✅ Qwen3 TTS Base model exists."
fi

# 6. Check and Download Qwen3 ASR model
if [ ! -d "$ASR_DIR" ]; then
    echo "⬇️ Downloading Qwen3 ASR (3.4GB)..."
    hf download Qwen/Qwen3-ASR-1.7B \
        --cache-dir "$HUB_DIR"
else
    echo "✅ Qwen3 ASR model exists."
fi

# 6b. Check and Download BGE embedding model
BGE_DIR="$HUB_DIR/models--BAAI--bge-small-en-v1.5"
if [ ! -d "$BGE_DIR" ]; then
    echo "⬇️ Downloading BGE-small-en-v1.5 for semantic note search (~130MB)..."
    hf download BAAI/bge-small-en-v1.5 \
        --cache-dir "$HUB_DIR"
else
    echo "✅ BGE-small-en-v1.5 model exists."
fi


# 7. Prepare runtime environment
# Ensure XDG_RUNTIME_DIR is set with the correct UID (compose files reference it)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Ensure PulseAudio cookie exists (PipeWire systems may not create one)
mkdir -p "${HOME}/.config/pulse"
[ -f "${HOME}/.config/pulse/cookie" ] || touch "${HOME}/.config/pulse/cookie"

# 7b. Load PulseAudio echo-cancellation module (required for usable barge-in
# on speakerphone setups). Idempotent. Opt out with FULLOCH_SKIP_AEC=1.
if [ "${FULLOCH_SKIP_AEC:-0}" = "1" ]; then
    echo "ℹ️  FULLOCH_SKIP_AEC=1 — skipping module-echo-cancel load."
elif command -v pactl &> /dev/null && pactl info &> /dev/null; then
    if pactl list short modules | grep -q "module-echo-cancel"; then
        echo "✅ module-echo-cancel already loaded."
    else
        echo "🎙️  Loading PulseAudio module-echo-cancel (webrtc backend)..."
        if pactl load-module module-echo-cancel \
            source_name=echocancel_source \
            sink_name=echocancel_sink \
            aec_method=webrtc \
            source_properties=device.description=Echo-Cancelled\ Source \
            sink_properties=device.description=Echo-Cancelled\ Sink \
            > /dev/null 2>&1; then
            echo "✅ module-echo-cancel loaded."
            echo "   compose.yml routes the container through it via PULSE_SOURCE / PULSE_SINK."
        else
            echo "⚠️  Failed to load module-echo-cancel. Continuing without echo cancellation."
            echo "   Barge-in may self-trigger from the assistant's own voice."
        fi
    fi
else
    echo "ℹ️  pactl not found or no PulseAudio server — skipping echo-cancel load."
fi

# 8. Optional rebuild
read -p "Rebuild image before starting? (y/N): " response
response=${response,,}
if [[ "$response" == "y" || "$response" == "yes" ]]; then
    BUILD_FLAG="--build"
    echo "🔨 Rebuilding image"
else
    BUILD_FLAG=""
fi

# 8. Optional detach
read -p "Detach the containers? (y/N): " response
response=${response,,}
if [[ "$response" == "y" || "$response" == "yes" ]]; then
    DETACH_FLAG="-d"
    echo "Detaching containers"
else
    DETACH_FLAG=""
fi

# 9. Launch Docker Compose
echo "🚀 All files checked. Starting services..."
docker compose up $DETACH_FLAG $BUILD_FLAG
