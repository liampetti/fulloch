# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fulloch (the **Full**y **Loc**al **H**ome voice assistant) is a fully local, privacy-focused AI voice home assistant. It runs speech recognition (Qwen3 ASR), text-to-speech (Qwen3 TTS with voice cloning), and a small language model (Qwen 3 4B) entirely on-device with no cloud dependencies.

## Build and Run Commands

### Development
```bash
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with dev dependencies
python app.py
```

### Docker Deployment
```bash
./launch.sh  # Downloads models, configures GPU/CPU, starts services
```

The launch script handles model downloads (Qwen GGUF, Qwen3 ASR, Qwen3 TTS, Kokoro, Moonshine) and Docker Compose setup.

### Testing
```bash
pytest tests/                      # Run all tests
pytest tests/test_intent_catch.py  # Test regex intent patterns
pytest tests/test_tool_registry.py # Test tool registration
```

### Testing Individual Components
```bash
python utils/intent_catch.py   # Test regex intent patterns
python utils/intents.py        # Test intent handler with tool registry
```

## Architecture

### Core Package (`core/`)
The main assistant logic is split into focused modules:

- `core/audio.py` - AudioCapture class, silence detection, recorder thread
- `core/asr.py` - Qwen3 ASR loading and pipeline (default)
- `core/asr_tiny.py` - Moonshine Tiny ASR for edge devices
- `core/tts.py` - Qwen3 TTS with voice cloning (default)
- `core/tts_tiny.py` - Kokoro TTS for edge devices
- `core/slm.py` - Qwen SLM loading and generate_slm()
- `core/assistant.py` - Main orchestration, transcriber thread, wakeword detection

### Audio Pipeline (Two Threads)
- **Recorder thread** (`core/audio.py`): Captures microphone input, detects silence/speech via RMS threshold, enqueues complete utterances
- **Transcriber thread** (`core/assistant.py`): Runs ASR (Qwen3 or Moonshine Tiny), detects wakeword, processes intents

### Intent Resolution (Three-Tier Fallback)
1. **Regex catch** (`utils/intent_catch.py`): Fast pattern matching for common commands (play, stop, pause, timer, time)
2. **AI intent detection**: Qwen SLM with JSON grammar constraint parses `{"intent": "name", "args": [...]}`
3. **Free-form chat**: Falls back to conversational AI if intent is ambiguous

### Tool Registry System
Tools are registered via decorator in `tools/tool_registry.py`:
```python
from tools.tool_registry import tool

@tool(name="function_name", description="...", aliases=["alias1"])
def my_function(param: str) -> str:
    ...
```

All tools auto-import via `tools/__init__.py`. Schemas auto-generate for OpenAI function calling format.

### Intent Formats (Two Supported)
- Function call: `{"function_call": {"name": "...", "arguments": "..."}}`
- Legacy: `{"intent": "...", "args": [...]}`

## Key Configuration

### Audio Parameters (`core/audio.py`)
```python
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200      # Callback slice
SILENCE_DURATION_MS = 1000   # End of utterance threshold
MIN_UTTERANCE_MS = 1500      # Minimum speech length
MAX_UTTERANCE_MS = 10000     # Maximum speech length
SILENCE_THRESHOLD = 0.001    # RMS threshold (lower = more sensitive)
```

### Config Files (not in git)
- `data/config.yml`: Service endpoints, wakeword, integration settings
- `.env`: Credentials (Spotify, Google, etc.)
- `data/models/`: Local model cache (~4-5GB)

### Example Config Files (in git)
- `data/config.example.yml`: Template with all settings documented
- `.env.example`: Template for credentials

## Adding New Tools

1. Create `tools/new_tool.py`
2. Use `@tool()` decorator to register functions
3. Import in `tools/__init__.py`
4. Tool automatically available in intent prompts and registry

## Project Structure

```
fulloch/
├── app.py              # Entry point
├── core/               # Core modules
│   ├── __init__.py
│   ├── audio.py        # Audio capture
│   ├── asr.py          # Qwen3 ASR (default)
│   ├── asr_tiny.py     # Moonshine Tiny ASR (edge)
│   ├── tts.py          # Qwen3 TTS (default)
│   ├── tts_tiny.py     # Kokoro TTS (edge)
│   ├── slm.py          # Language model
│   └── assistant.py    # Orchestration
├── tools/              # Smart home tools
│   ├── __init__.py
│   ├── tool_registry.py
│   └── ...
├── utils/              # Utilities
│   ├── __init__.py
│   ├── intent_catch.py
│   ├── intents.py
│   └── system_prompts.py
├── audio/              # Audio utilities
│   ├── __init__.py
│   └── beep_manager.py
├── tests/              # Test suite
│   ├── conftest.py
│   ├── test_intent_catch.py
│   └── test_tool_registry.py
└── data/               # Config and models
    └── config.example.yml
```

## Available Integrations

Spotify, Philips Hue, Google Calendar, LG ThinQ, Pioneer AVR, Airtouch HVAC, WebOS TV, SearXNG search, BOM Australia weather
