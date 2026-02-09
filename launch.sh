#!/bin/bash
set -e

# Helper function to ask user for confirmation
ask_download() {
    local model_name="$1"
    read -p "Download $model_name? (y/n): " response
    response=${response,,}
    [[ "$response" == "y" || "$response" == "yes" ]]
}

# Directory definitions
BASE_DIR="$(pwd)/data/models"
GRAMMAR_DIR="$BASE_DIR/grammars"
HUB_DIR="$BASE_DIR/hub"

# Model specific variables
QWEN_REPO="unsloth/Qwen3-4B-Instruct-2507-GGUF"
QWEN_FILE="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# Define the cache folders
ASR_TINY_DIR="$HUB_DIR/models--UsefulSensors--moonshine-tiny"
ASR_DIR="$HUB_DIR/models--Qwen--Qwen3-ASR-1.7B"
TTS_TINY_DIR="$HUB_DIR/models--hexgrad--Kokoro-82M"
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
    if ask_download "json.gbnf (grammar file)"; then
        echo "⬇️ Downloading json.gbnf..."
        wget -q --show-progress -O "$GRAMMAR_DIR/json.gbnf" \
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/json.gbnf"
    else
        echo "⏭️ Skipping json.gbnf"
    fi
else
    echo "✅ json.gbnf exists."
fi

# 4. Check and Download Qwen3 GGUF
if [ ! -f "$BASE_DIR/$QWEN_FILE" ]; then
    if ask_download "Qwen3 4B SLM (2.5GB)"; then
        echo "⬇️ Downloading $QWEN_FILE..."
        hf download "$QWEN_REPO" "$QWEN_FILE" \
            --local-dir "$BASE_DIR"
    else
        echo "⏭️ Skipping Qwen3 SLM"
    fi
else
    echo "✅ $QWEN_FILE exists."
fi

# 5. Check and Download Qwen3 TTS model
if [ ! -d "$TTS_DIR" ]; then
    if ask_download "Qwen3 TTS (3.4GB)"; then
        echo "⬇️ Downloading Qwen3 TTS..."
        hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
            --cache-dir "$HUB_DIR"
    else
        echo "⏭️ Skipping Qwen3 TTS"
    fi
else
    echo "✅ Qwen3 TTS model exists."
fi

# 5a. Check and Download Kokoro-82M (TTS)
if [ ! -d "$TTS_TINY_DIR" ]; then
    if ask_download "Kokoro-82M TTS Tiny (200MB)"; then
        echo "⬇️ Downloading Kokoro-82M..."
        hf download hexgrad/Kokoro-82M \
            --cache-dir "$HUB_DIR"
    else
        echo "⏭️ Skipping Kokoro-82M"
    fi
else
    echo "✅ Kokoro-82M exists."
fi

# 6. Check and Download Qwen3 ASR model
if [ ! -d "$ASR_DIR" ]; then
    if ask_download "Qwen3 ASR (3.4GB)"; then
        echo "⬇️ Downloading Qwen3 ASR..."
        hf download Qwen/Qwen3-ASR-1.7B \
            --cache-dir "$HUB_DIR"
    else
        echo "⏭️ Skipping Qwen3 ASR"
    fi
else
    echo "✅ Qwen3 ASR model exists."
fi

# 6a. Check and Download Moonshine Tiny ASR model
if [ ! -d "$ASR_TINY_DIR" ]; then
    if ask_download "Moonshine Tiny ASR (60MB)"; then
        echo "⬇️ Downloading Moonshine Tiny..."
        hf download UsefulSensors/moonshine-tiny \
            --cache-dir "$HUB_DIR"
    else
        echo "⏭️ Skipping Moonshine Tiny"
    fi
else
    echo "✅ Moonshine-tiny exists."
fi

# 7. Prompt the user
read -p "Are you using an NVIDIA GPU? (y/n): " response
response=${response,,}
if [[ "$response" == "y" || "$response" == "yes" ]]; then
    COMPOSE_FILE="compose_gpu.yml"
    echo "✅ Using GPU enabled containers"
else
    COMPOSE_FILE="compose.yml"
    echo "✅ Using default containers"
fi

# 8. Prepare runtime environment
# Ensure XDG_RUNTIME_DIR is set with the correct UID (compose files reference it)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Ensure PulseAudio cookie exists (PipeWire systems may not create one)
mkdir -p "${HOME}/.config/pulse"
[ -f "${HOME}/.config/pulse/cookie" ] || touch "${HOME}/.config/pulse/cookie"

# 9. Launch Docker Compose
echo "🚀 All files checked. Starting services..."
docker compose -f "$COMPOSE_FILE" up -d
