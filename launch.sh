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
if ! command -v huggingface-cli &> /dev/null; then
    echo "⬇️ huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# 2. Create Directory Structure
echo "📂 Checking directory structure..."
mkdir -p "$HUB_DIR"

# 2a. Check for config.yml
CONFIG_FILE="$(pwd)/data/config.yml"
CONFIG_EXAMPLE="$(pwd)/data/config.example.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "📝 config.yml not found. Creating from template..."
    cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
    echo ""
    echo "⚠️  Please edit data/config.yml with your settings before continuing."
    echo "   See data/config.example.yml for documentation on each option."
    echo ""
    echo "   Run ./launch.sh again when ready."
    exit 0
else
    echo "✅ config.yml exists."
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
        huggingface-cli download "$QWEN_REPO" "$QWEN_FILE" \
            --local-dir "$BASE_DIR" \
            --local-dir-use-symlinks False
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
        huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
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
        huggingface-cli download hexgrad/Kokoro-82M \
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
        huggingface-cli download Qwen/Qwen3-ASR-1.7B \
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
        huggingface-cli download UsefulSensors/moonshine-tiny \
            --cache-dir "$HUB_DIR"
    else
        echo "⏭️ Skipping Moonshine Tiny"
    fi
else
    echo "✅ Moonshine-tiny exists."
fi

# 7. Prompt the user
read -p "Are you using a GPU? (y/n): " response
response=${response,,}
if [[ "$response" == "y" || "$response" == "yes" ]]; then
    mv Dockerfile Dockerfile_cpu
    mv Dockerfile_gpu Dockerfile
    mv compose.yml compose_cpu.yml
    mv compose_gpu.yml compose.yml
    echo "✅ Using GPU enabled containers"
else
    echo "✅ Using default containers"
fi

# 8. Launch Docker Compose
echo "🚀 All files checked. Starting services..."
docker compose up -d
