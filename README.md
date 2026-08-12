# Fulloch

[![GitHub Release][releases-shield]][releases] [![License][license-shield]](LICENSE) [![hacs][hacs-shield]][hacs]

<p align="center">
  <img src="fulloch.png" alt="Fulloch Logo" width="200">
</p>

_The **Ful**ly **Loc**al **H**ome Voice Assistant, a private voice layer for your notes, your home, and the web._

Fulloch is your privacy-focused local voice assistant running on your own PC or Mac. Ask questions, capture thoughts, and search your **[Obsidian](https://github.com/obsidianmd/obsidian-releases)** vault by voice. Control your home via **[Home Assistant](https://github.com/home-assistant)**. Pull live answers from the web with **[SearXNG](https://github.com/searxng/searxng)**. The local stack keeps speech, language processing, notes, facts, and conversation history on your machine; optional web search, Spotify, and remote language models contact their configured services.

[Click here to see the satellite companion project](https://github.com/liampetti/fulloch-satellite) for running multiple ESP32-S3 satellites around your home.

## Features

- **Obsidian notes** - read, write, append, and search your vault by voice; capture a conversation as a note without leaving what you're doing
- **Semantic search** - *"what did I write about the car service?"* finds the right note by meaning, not just keywords
- **Web search** - ask a question, get a spoken summary pulled live from a self-hosted search engine, optionally saved to your vault
- **Conversational** - holds context across a turn; follow-ups like *"and tomorrow?"* just work
- **Memory** - facts persist across restarts and build up over time
- **Smart-home control** - control any of your smart home devices using Home Assistant, integrate Fulloch into Home Assistant to trigger voice notifications and track your conversational history
- **Music search & play** - *"play the Beatles"*, *"play jazz in the kitchen"*, *"play music everywhere"* - smart search on Spotify directly and hand playback off to Home Assistant
- **Calendar reminders** - creates events on a dedicated HA calendar and speaks them at the right time
- **Barge-in** - interrupt mid-sentence with the wakeword
- **Voice options** - the GPU stack clones from reference audio; CPU defaults to Pocket TTS one-shot cloning, with minimal built-in Kokoro voices available as an alternative
- **Quiet delivery** - ask it to whisper or speak quietly; every TTS backend lowers output volume to 30% by default

> **Higgs TTS 3:** The optional Higgs GPU backend is available only under Boson AI's Research and Non-Commercial License, not Fulloch's MIT license. It requires explicit consent for every voice reference. See [Model Sources and Licenses](MODELS.md#higgs-tts-3-license).

> **Pocket TTS PyTorch:** The experimental GPU streaming option uses Kyutai's official gated model. Accept its Hugging Face terms before selecting it. If the download is denied, the wizard prompts for a Hugging Face read token, saves it in `data/credentials.json`, and retries. See [Model Sources and Licenses](MODELS.md#hugging-face-access).

## Quick installation

The default stack runs on **CPU (mac/linux/windows)**. Audio runs through the browser dashboard. The LLM is either regex-only (simple commands) or off-box via an OpenAI-compatible endpoint you configure in the wizard (e.g. Ollama / LM Studio / another machine on your LAN). The dashboard avatar swaps to Parloch, the **Par**tially-**loc**al **h**ome voice assistant, when the LLM is running off-device.

Install Docker Desktop (or Docker Engine) first. The first run downloads the selected speech models, so it needs an internet connection, several GB of free disk, and enough Docker memory for the wizard's displayed estimate. For the GPU image, install a current NVIDIA driver and NVIDIA Container Toolkit, then confirm `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` works before launching Fulloch.

### CPU (mac/linux/windows)

```bash
docker run -d \
  --name fulloch-ai \
  --restart unless-stopped \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  -p 8765:8765 \
  -e DASHBOARD_HOST=0.0.0.0 \
  -v fulloch-data:/app/data:rw \
  ghcr.io/liampetti/fulloch:cpu
```

The named volume is repaired automatically on first boot and receives the bundled starter voice references. If you prefer a bind mount, use an existing writable directory such as `-v "$PWD/data:/app/data:rw"`; see [data-directory permissions](TECHNICAL_DETAILS.md#docker-data-directory-permissions) if startup reports a permission error. Open the HTTPS URL shown in `docker logs fulloch-ai` in a browser to continue the setup wizard. A self-signed certificate warning is expected on first visit; accept it to enable browser microphone access. The dashboard is LAN-visible with this command, so set a dashboard password in the wizard before using it beyond a trusted network.

Add `-v /path/to/your/ObsidianVault:/vault:rw` before the image name to expose an Obsidian vault.

### GPU (Linux/Windows + NVIDIA)

Swap `:cpu` for `:latest` (the CUDA image with Qwen3-TTS voice cloning and the on-GPU 9B SLM) and add `--gpus all`:

```bash
docker run -d \
  --name fulloch-ai \
  --restart unless-stopped \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  --gpus all \
  -p 8765:8765 \
  -e DASHBOARD_HOST=0.0.0.0 \
  -v fulloch-data:/app/data:rw \
  ghcr.io/liampetti/fulloch:latest
```

### OpenAI Endpoint

<img src="parloch.png" alt="Fulloch being Parloch" width="180" align="right">

The moment you point the language model at an OpenAI-compatible endpoint, the avatar and favicon swap to a travelling version of the character (Let's call him *Parloch*: The **Par**tially **loc**al **h**ome voice assistant), and the tagline reads *"language model is off-device."* Pick a local model again (**None** or the GPU 9B) and Fulloch comes home. It triggers as soon as a remote endpoint is *configured*, even if it is on your home network.

### Recovering Or Reconfiguring

Use **Settings** for normal changes: voices, model backends, language-model mode, Home Assistant, Search, and dashboard preferences. Model changes need a restart; if their weights are missing, the restart returns to the wizard to download them. The **Re-run setup wizard** action makes a backup under `data/backups/`; use it to choose a different stack, not to edit a custom model path or advanced remote-LLM settings. To repair a failed download, re-run setup and select the same stack; incomplete model caches are detected and downloaded again.

Advanced options not shown in Settings are documented in `data/config.example.yml`: Spotify OAuth, native satellite tokens, and external-LLM timeouts. Keep those files under the persistent `data` volume.

### Obsidian Integration
Connect your Obsidian vault so Fulloch reads, writes, appends, and searches your notes by voice.

Just add your vault's directory as a volume when launching Docker container `-v /Users/you/Documents/MyVault:/vault:rw`

The Obsidian wizard's "Auto-detect" scans the container filesystem for a vault, so it'll find `/vault` (or wherever you mounted it) without further config.

The included Obsidian plugin adds live assistance for the active document. Install it from the dashboard's Obsidian tab; see the [Obsidian setup details](TECHNICAL_DETAILS.md#obsidian-integration).

## Home Assistant Integration

When Fulloch runs in Docker, enter Home Assistant's LAN address, for example `http://192.168.1.50:8123`. `localhost` refers to the Fulloch container, not Home Assistant.

A HACS-installable integration for status sensors, mic control, proactive speech, and automation triggers.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=liampetti&repository=fulloch&category=Integration)

| Entity | Description |
| -- | -- |
| `sensor.fulloch_status` | `idle` / `thinking` / `speaking` |
| `sensor.fulloch_last_utterance` | Last thing the user said |
| `sensor.fulloch_last_response` | Last thing Fulloch said (`full_text` attribute has the full string) |
| `switch.fulloch_mic` | Mute / unmute the microphone |
| `text.fulloch_speak` | Submit text → Fulloch speaks it |
| `text.fulloch_chat` | Submit a query → full agent loop |

| Action | Field | Description |
| -- | -- | -- |
| `fulloch.speak` | `text` | Speak a message |
| `fulloch.chat` | `text` | Run a full agent query and speak the result |
| `fulloch.mic` | `enabled` | Turn the mic on or off |

| Event | When |
| -- | -- |
| `fulloch_wakeword_detected` | Voice turn starts |
| `fulloch_turn_ended` | Fulloch finishes speaking |

## Instant Commands

Common commands take a regex fast-path that skips the language model entirely, for an instant response. Compound requests (*"… and …"*), vague references (*"turn it off"*), or conversation, falls through to the full agent.

| Say | Does |
| -- | -- |
| *"turn on/off the fan"* | on / off |
| *"toggle the porch light"* | toggle |
| *"set the kitchen lights to 60 percent"* | brightness |
| *"dim / brighten the lights"* | dim (30%) / brighten (100%) |
| *"make the lamp blue"* | colour |
| *"turn the volume up / down"* | volume |
| *"lock / unlock the front door"* | lock / unlock |
| *"open / close the blinds"* | covers |
| *"play the Beatles"* | music search & play |
| *"stop"* · *"skip"* · *"resume"* | media control |
| *"set a timer for 5 minutes"* · *"list my timers"* | timers |
| *"what time is it"* | time |
| *"think about …"* · *"summarise your thinking"* | thinking mode |

## Reporting a Problem

Submit a [Bug Report](https://github.com/liampetti/fulloch/issues/new?labels=bug) or a [Feature Request](https://github.com/liampetti/fulloch/issues/new?labels=enhancement).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tools and submit changes.

## Credits

Model download sources and licensing notes are listed in [MODELS.md](MODELS.md).

Voices in `data/voices/`:
- **`atticus` / `tulloch`** - generated with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (Apache-2.0) from text descriptions; synthetic, not clones of real people
- **`cori`** - sample from [Piper](https://github.com/rhasspy/piper) `en_GB/cori/high` by Bryce Beattie, trained on LibriVox recordings (MIT / public domain)
- **`All Kokoro voices`** - generated with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0)

Pocket TTS uses a selected `data/voices/<name>.wav` reference for one-shot cloning; use only voices you have permission to reproduce. The experimental official PyTorch backend streams PCM as it generates; the GGUF and ONNX options use independent conversions.

## License

MIT - see [LICENSE](LICENSE).

***

[releases-shield]: https://img.shields.io/github/release/liampetti/fulloch.svg?style=for-the-badge
[releases]: https://github.com/liampetti/fulloch/releases
[license-shield]: https://img.shields.io/github/license/liampetti/fulloch.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacs-shield]: https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge
