# Fulloch

[![GitHub Release][releases-shield]][releases] [![License][license-shield]](LICENSE) [![hacs][hacs-shield]][hacs]

<p align="center">
  <img src="fulloch.png" alt="Fulloch Logo" width="200">
</p>

_The **Ful**ly **Loc**al **H**ome Voice Assistant — private, conversational, 100% on-device._

A voice assistant with agentic memory, web research, and smart-home control running entirely on your own GPU. Speech recognition, the language model, and the spoken voice never leave your machine.

Fulloch sits on top of your existing setup as the conversational brain — it drives **[Home Assistant](https://github.com/home-assistant)** for smart-home control and reads and writes plain **Markdown notes**, so it plugs straight into an **[Obsidian](https://github.com/obsidianmd/obsidian-releases)** vault or any Markdown workflow.

## Features

- **Conversational** — holds context across a turn; follow-ups like *"and tomorrow?"* just work
- **Barge-in** — interrupt the assistant mid-sentence with the wakeword
- **Agentic memory** — facts persist across restarts and build up over time
- **Markdown notes** — read, write, append, and search `.md` files by voice; point it at your Obsidian vault
- **Semantic search** — *"what did I write about the car service?"* via a local embedding model
- **Conversation recall** — *"what did we talk about yesterday afternoon?"* — summarises past turns from Home Assistant history
- **Web research** — live answers from a self-hosted search engine, summarised into a short spoken reply
- **Smart-home control** — lights, climate, media, calendar, weather, scenes, and entity history via Home Assistant
- **Music search & play** — *"play the Beatles"* via [SpotifyPlus](https://github.com/thlucas1/homeassistantcomponent_spotifyplus) (required for search-by-name; basic playback control works with the built-in Spotify integration)
- **Calendar reminders** — creates events on a dedicated HA calendar and speaks them at the right time
- **Thinking mode** — *"think about X"* for a slower, deeper answer; interrupt to get a partial summary
- **Cloned voice** — speaks in a voice cloned from a few seconds of reference audio
- **Web dashboard** *(optional)* — type from your phone; voice and text share the same brain and history

## The Local Stack

| Model | Role |
| -- | -- |
| Qwen3.5-9B GGUF Q5_K_M (llama.cpp) | Language model |
| Qwen3-ASR-1.7B | Speech recognition |
| Qwen3-TTS-12Hz-1.7B-Base | Text-to-speech |
| bge-small-en-v1.5 | Semantic note search |

**Minimum hardware:** 16GB VRAM GPU (e.g. RTX 5060 Ti). The full pipeline is GPU-resident; there is no CPU fallback.

## Home Assistant Integration

This integration will set up the following platforms.

Platform | Description
-- | --
`sensor` | Status, last utterance, last response
`switch` | Microphone mute / unmute
`text` | Speak and chat text input entities

### Optional HA integrations

| Integration | Purpose |
| -- | -- |
| [SpotifyPlus](https://github.com/thlucas1/homeassistantcomponent_spotifyplus) | Required for *"play the Beatles"* style music queries. Basic Spotify playback control (pause, skip, volume) works without it. |

### HA HACS Installation

Use the following link to open (and add) the Fulloch repository in HACS:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=liampetti&repository=fulloch&category=Integration)

If the above does not work, add it manually:
- Go to HACS → three-dot menu → **Custom repositories**
- Paste `https://github.com/liampetti/fulloch`, select **Integration**, click **Add**
- Find the Fulloch entry and click **Download**
- Restart Home Assistant

### HA Manual Installation

- Copy `custom_components/fulloch/` into your HA `config/custom_components/` directory
- Restart Home Assistant
- Go to **Settings → Integrations → Add → Fulloch** and enter your Fulloch host and dashboard port

### HA Entities

| Entity | Type | Description |
| -- | -- | -- |
| `sensor.fulloch_status` | Sensor | `idle` / `thinking` / `speaking` |
| `sensor.fulloch_last_utterance` | Sensor | Last thing the user said |
| `sensor.fulloch_last_response` | Sensor | Last thing Fulloch said (`full_text` attribute has the full string) |
| `switch.fulloch_mic` | Switch | Mute / unmute the microphone |
| `text.fulloch_speak` | Text | Submit text → Fulloch speaks it |
| `text.fulloch_chat` | Text | Submit a query → full agent loop, speaks the result |

### HA Actions

| Action | Field | Description |
| -- | -- | -- |
| `fulloch.speak` | `text` | Speak a message through the cloned voice |
| `fulloch.chat` | `text` | Run a full agent-loop query and speak the result |
| `fulloch.mic` | `enabled` | Turn the mic on or off |

### HA Events

| Event | When |
| -- | -- |
| `fulloch_wakeword_detected` | Wakeword heard or voice turn starts |
| `fulloch_turn_ended` | Fulloch finishes speaking a response |

Use these in automations — e.g. dim lights on `fulloch_wakeword_detected`, restore on `fulloch_turn_ended`.

### Proactive speech from automations

```yaml
action: fulloch.speak
data:
  text: "The front door just opened."
```

```yaml
action: fulloch.chat
data:
  text: "Tell me today's calendar and weather."
```

## Fulloch Installation

### Linux (Docker)

```bash
git clone https://github.com/liampetti/fulloch.git
cd fulloch
./launch.sh
```

On first run, creates `data/config.yml` and `.env` from templates, pauses for editing, downloads models, loads PulseAudio echo cancellation, and starts via Docker Compose.

### Windows

```bat
git clone https://github.com/liampetti/fulloch.git
cd fulloch
pip install -r requirements.txt
launch.bat
```

SearXNG runs in Docker; the assistant runs natively in Python — no audio-passthrough complexity. Requires Python 3.10+, Docker Desktop, and an NVIDIA GPU with CUDA.

### Dependencies

`requirements.txt` pins every direct dependency to an exact version for reproducible installs. The GPU stack (CUDA `torch`, plus the git-installed `qwen-tts` / `flash-attn`) can't be PyPI-locked cleanly, so there's no transitive lockfile — snapshot a known-good environment by freezing the built image:

```bash
docker compose run --rm app pip freeze > requirements.lock
```

## Configuration

Everything lives in `data/config.yml`. Key settings:

```yaml
general:
  wakeword: "hey atticus"         # activation phrase
  barge_in: "wakeword"            # "off" | "wakeword" (needs AEC)
  follow_up_time: "5s"            # wakeword-free reply window after TTS ends
  voice_clone: "atticus"          # data/voices/<name>.{wav,txt}
  dashboard_port: 8765            # web chat UI; remove to disable
  dashboard_host: "127.0.0.1"     # local-only; "0.0.0.0" to reach it from other devices
  # dashboard_ssl_certfile: "./data/certs/fulloch.pem"   # optional HTTPS (both keys required)
  # dashboard_ssl_keyfile:  "./data/certs/fulloch-key.pem"
  use_vad: true                   # drop non-speech buffers (coughs, taps) before ASR
  asr_context_hint: true          # bias ASR decoder toward wakeword spelling
  asr_context_terms:              # optional extra terms to bias (max 10)
    - "phoebe bridgers"

home_assistant:
  url: "http://192.168.1.50:8123"
  token: "your_long_lived_access_token"  # or set FULLOCH_HA_TOKEN in .env (preferred)
  calendar: "Fulloch"             # HA calendar for voice reminders (optional)

notes:
  path: "/path/to/obsidian/vault" # defaults to data/notes/
  daily_subdir: "daily"           # optional daily journal subfolder

search:
  searxng_url: "http://localhost:8080/search"
```

Full reference: [`data/config.example.yml`](data/config.example.yml)

Secrets and tokens live in `.env` (not `config.yml`): `SEARXNG_SECRET`,
`FULLOCH_DASHBOARD_TOKEN` (dashboard auth), and `FULLOCH_HA_TOKEN` (Home Assistant,
overrides `home_assistant.token`). See [`.env.example`](.env.example).

### Exposing the dashboard

The dashboard can read and write your notes, toggle the microphone, speak through
your speakers, and drive Home Assistant — so it is **unauthenticated and bound to
`127.0.0.1` (local-only) by default**.

To reach it from your phone or another device on your network:

1. **Bind to your network** — set `general.dashboard_host: "0.0.0.0"` in `data/config.yml`.
2. **Require a token** — set `FULLOCH_DASHBOARD_TOKEN` in `.env` (generate one with
   `openssl rand -hex 32`). This is mandatory: without it, anyone on your LAN gets
   full notes/mic/speech/Home-Assistant control, and Fulloch logs a warning at startup.
3. **Open it once** at `http://<this-host-ip>:<port>/?token=<your-token>` (find the IP
   with `hostname -I`). The browser stores the token and strips it from the address
   bar, so you can bookmark the bare `http://<host>:<port>/` afterwards. If the token
   is missing or wrong, the page prompts you for it.

#### Optional: HTTPS

Over plain HTTP the token travels in clear text — fine on a trusted home LAN, but if
you want the traffic encrypted, point Fulloch at a TLS cert/key pair in `data/config.yml`:

```yaml
general:
  dashboard_ssl_certfile: "./data/certs/fulloch.pem"
  dashboard_ssl_keyfile:  "./data/certs/fulloch-key.pem"
```

Both keys are required (set only one and TLS is skipped with a warning), and the URL
becomes `https://...`. For a certificate that doesn't trip browser warnings on your
devices, generate one with [mkcert](https://github.com/FiloSottile/mkcert) and install
its root CA on each device. In Docker the cert/key must live under `./data` (the only
host folder mounted into the container), so keep them at `./data/certs/...` as above.
Do not expose this to the public internet without a proper reverse proxy doing TLS.

## Reporting a Problem

Submit a [Bug Report](https://github.com/liampetti/fulloch/issues/new?labels=bug) to bring the issue to my attention.

## Request a New Feature

Submit a [Feature Request](https://github.com/liampetti/fulloch/issues/new?labels=enhancement) to get your idea into the queue.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tools and submit changes.

## Credits

Voices in `data/voices/`:
- **`atticus` / `tulloch`** — generated with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (Apache-2.0) from text descriptions; synthetic, not clones of real people
- **`cori`** — sample from [Piper](https://github.com/rhasspy/piper) `en_GB/cori/high` by Bryce Beattie, trained on LibriVox recordings (MIT / public domain)

## License

MIT — see [LICENSE](LICENSE).

***

[releases-shield]: https://img.shields.io/github/release/liampetti/fulloch.svg?style=for-the-badge
[releases]: https://github.com/liampetti/fulloch/releases
[license-shield]: https://img.shields.io/github/license/liampetti/fulloch.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacs-shield]: https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge
