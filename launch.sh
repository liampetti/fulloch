#!/bin/bash
set -e

# Directory definitions
BASE_DIR="$(pwd)/data/models"
GRAMMAR_DIR="$BASE_DIR/grammars"
HUB_DIR="$BASE_DIR/hub"

# Model specific variables
QWEN_REPO="unsloth/Qwen3-4B-Instruct-2507-GGUF"
QWEN_FILE="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# Define the cache folders
ASR_TINY_DIR="$HUB_DIR/models--UsefulSensors--moonshine-tiny"
ASR_DIR="$HUB_DIR/models--Qwen--Qwen3-ASR-0.6B"
TTS_DIR="$HUB_DIR/models--hexgrad--Kokoro-82M"

# 1. Ensure Dependencies are installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "⬇️ huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# 2. Create Directory Structure
echo "📂 Checking directory structure..."
mkdir -p "$HUB_DIR"

# 3. Check and Download json.gbnf
if [ ! -f "$GRAMMAR_DIR/json.gbnf" ]; then
echo "⬇️ Downloading json.gbnf..."
wget -q --show-progress -O "$GRAMMAR_DIR/json.gbnf" \
"https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/json.gbnf"
else
echo "✅ json.gbnf exists."
fi

# 4. Check and Download Qwen3 GGUF
if [ ! -f "$BASE_DIR/$QWEN_FILE" ]; then
    echo "⬇️ Downloading $QWEN_FILE..."
    huggingface-cli download "$QWEN_REPO" "$QWEN_FILE" \
        --local-dir "$BASE_DIR" \
        --local-dir-use-symlinks False
else
    echo "✅ $QWEN_FILE exists."
fi

# 5. Check and Download Kokoro-82M (TTS)
if [ ! -d "$TTS_DIR" ]; then
echo "⬇️ Downloading Kokoro-82M..."
huggingface-cli download hexgrad/Kokoro-82M \
--cache-dir "$HUB_DIR"
else
echo "✅ Kokoro-82M exists."
fi

# 6. Check and Download Qwen3 ASR model
if [ ! -d "$ASR_DIR" ]; then
    echo "⬇️ Downloading Qwen3 ASR..."
    huggingface-cli download Qwen/Qwen3-ASR-0.6B \
        --cache-dir "$HUB_DIR"
else
    echo "✅ ASR model exists."
fi

# 6a. Check and Download Moonshine Tiny ASR model
if [ ! -d "$ASR_TINY_DIR" ]; then
echo "⬇️ Downloading Moonshine Tiny..."
huggingface-cli download UsefulSensors/moonshine-tiny \
--cache-dir "$HUB_DIR"
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
