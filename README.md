# Fulloch

[![GitHub Release][releases-shield]][releases] [![License][license-shield]](LICENSE) [![hacs][hacs-shield]][hacs]

<p align="center">
  <img src="fulloch.png" alt="Fulloch Logo" width="200">
</p>

_The **Ful**ly **Loc**al **H**ome Voice Assistant - private, conversational, 100% on-device._

A voice assistant with agentic memory, web research, and smart-home control running entirely on your own GPU. Speech recognition, the language model, and the spoken voice never leave your machine.

Fulloch is the conversational brain on top of your existing setup: it drives **[Home Assistant](https://github.com/home-assistant)** for smart-home control and reads/writes plain **Markdown notes**, so it plugs straight into an **[Obsidian](https://github.com/obsidianmd/obsidian-releases)** vault or any Markdown workflow.

**Minimum hardware:** 16GB VRAM GPU (e.g. RTX 5060 Ti). The full pipeline is GPU-resident.

## Features

- **Conversational** - holds context across a turn; follow-ups like *"and tomorrow?"* just work
- **Barge-in** - interrupt mid-sentence with the wakeword
- **Agentic memory** - facts persist across restarts and build up over time
- **Markdown notes** - read, write, append, and search `.md` files by voice; point it at your Obsidian vault
- **Semantic search** - *"what did I write about the car service?"* via a local embedding model
- **Conversation recall** - *"what did we talk about yesterday afternoon?"*, summarised from Home Assistant history
- **Web research** - live answers from a self-hosted search engine, summarised into a short spoken reply
- **Smart-home control** - lights, climate, media, calendar, weather, scenes, and entity history via Home Assistant; any entity can be switched off for voice control from the dashboard
- **Music search & play** - *"play the Beatles"* via [SpotifyPlus](https://github.com/thlucas1/homeassistantcomponent_spotifyplus)
- **Calendar reminders** - creates events on a dedicated HA calendar and speaks them at the right time
- **Thinking mode** - *"think about X"* for a slower, deeper answer; interrupt to get a partial summary
- **Cloned voice** - speaks in a voice cloned from a few seconds of reference audio
- **Web dashboard** *(optional)* - type from your phone; voice and text share the same brain and history

## Installation

Requires an NVIDIA GPU with CUDA (16GB VRAM) and Docker.

### Linux (Docker)

```bash
git clone https://github.com/liampetti/fulloch.git
cd fulloch
./launch.sh
```

`launch.sh` creates `data/config.yml` and `.env` from templates (pausing for you to edit them), downloads the models, and starts Docker Compose.

### Windows

```bat
git clone https://github.com/liampetti/fulloch.git
cd fulloch
pip install -r requirements.txt
launch.bat
```

Requires Python 3.10+ and Docker Desktop. SearXNG runs in Docker; the assistant runs natively in Python, with no audio-passthrough setup.

> **Reproducible installs:** `requirements.txt` pins direct dependencies, but the CUDA stack (`torch`, git-installed `qwen-tts` / `flash-attn`) can't be PyPI-locked cleanly. Freeze a known-good image with `docker compose run --rm app pip freeze > requirements.lock`.

## Configuration

Everything lives in `data/config.yml`; secrets live in `.env`. Key settings:

```yaml
general:
  wakeword: "hey atticus"         # activation phrase
  barge_in: "wakeword"            # "off" | "wakeword" (interrupt mid-response)
  follow_up_time: "5s"            # wakeword-free reply window after TTS ends
  voice_clone: "atticus"          # data/voices/<name>.{wav,txt}
  dashboard_port: 8765            # web chat UI; remove to disable
  dashboard_host: "127.0.0.1"     # local-only; "0.0.0.0" to reach it from other devices
  use_vad: true                   # drop non-speech buffers (coughs, taps) before ASR
  asr_context_hint: true          # bias ASR decoder toward the wakeword spelling
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

Secrets in `.env`: `SEARXNG_SECRET` (required), `FULLOCH_DASHBOARD_TOKEN` (dashboard auth), `FULLOCH_HA_TOKEN` (overrides `home_assistant.token`). Full reference: [`data/config.example.yml`](data/config.example.yml) and [`.env.example`](.env.example).

## Web Dashboard

Setting `general.dashboard_port` serves a phone-friendly chat UI; voice and text share the same brain and history. While Fulloch is thinking or speaking, whether the turn started by voice or text, the send button becomes a **Stop** button that halts it instantly and silently. It also exposes **Facts**, **Notes**, and **Entities** management tabs.

Because it can read/write notes, toggle the mic, speak through your speakers, and drive Home Assistant, it is **unauthenticated and bound to `127.0.0.1` by default**. To reach it from another device:

1. **Bind to your network** - `general.dashboard_host: "0.0.0.0"`.
2. **Require a token** - set `FULLOCH_DASHBOARD_TOKEN` in `.env` (`openssl rand -hex 32`). Mandatory: without it anyone on your LAN gets full control, and Fulloch warns at startup.
3. **Open it once** at `http://<host-ip>:<port>/?token=<token>` (find the IP with `hostname -I`). The browser stores the token and strips it from the URL, so you can bookmark the bare address.

> **HTTPS (optional):** over plain HTTP the token is sent in clear text, which is fine on a trusted LAN. To encrypt, set `dashboard_ssl_certfile` and `dashboard_ssl_keyfile` (both required) in `config.yml`. For certs that don't trip browser warnings, use [mkcert](https://github.com/FiloSottile/mkcert). In Docker the cert/key must live under `./data` (the only mounted host folder). Don't expose this to the public internet without a reverse proxy doing TLS.

## Home Assistant Integration

A HACS-installable integration that connects Home Assistant to a running Fulloch dashboard, for status sensors, mic control, proactive speech, and automation triggers.

### Installation

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=liampetti&repository=fulloch&category=Integration)

Or manually: HACS → three-dot menu → **Custom repositories** → paste `https://github.com/liampetti/fulloch`, category **Integration**, **Add** → **Download** the Fulloch entry → restart HA. Then **Settings → Integrations → Add → Fulloch** and enter your Fulloch host and dashboard port.

### Entities, Actions & Events

| Entity | Description |
| -- | -- |
| `sensor.fulloch_status` | `idle` / `thinking` / `speaking` |
| `sensor.fulloch_last_utterance` | Last thing the user said |
| `sensor.fulloch_last_response` | Last thing Fulloch said (full string in the `full_text` attribute) |
| `switch.fulloch_mic` | Mute / unmute the microphone |
| `text.fulloch_speak` | Submit text → Fulloch speaks it |
| `text.fulloch_chat` | Submit a query → full agent loop, speaks the result |

| Action | Field | Description |
| -- | -- | -- |
| `fulloch.speak` | `text` | Speak a message through the cloned voice |
| `fulloch.chat` | `text` | Run a full agent-loop query and speak the result |
| `fulloch.mic` | `enabled` | Turn the mic on or off |

| Event | When |
| -- | -- |
| `fulloch_wakeword_detected` | Wakeword heard or voice turn starts |
| `fulloch_turn_ended` | Fulloch finishes speaking a response |

Use the events in automations, e.g. dim lights on `fulloch_wakeword_detected`, restore on `fulloch_turn_ended`. Use the actions for proactive speech:

```yaml
action: fulloch.speak
data:
  text: "The front door just opened."
```

### Optional: SpotifyPlus

[SpotifyPlus](https://github.com/thlucas1/homeassistantcomponent_spotifyplus) is required for *"play the Beatles"* style search-by-name queries. Basic Spotify playback control (pause, skip, volume) works without it.

### Restricting voice control

Some entities, door locks, alarms, you may want usable from the dashboard but **not by voice**. The dashboard's **Entities** tab toggles any entity off voice control: Fulloch then refuses to act on it by voice (it says *"that isn't available for voice control"*) while it stays fully usable from the dashboard and in your own HA automations. Changes apply immediately, no restart, and persist in `data/voice_denylist.json`.

> This is **separate from Home Assistant's "Expose to Assist"** setting, which only governs HA's built-in Assist and isn't readable by Fulloch.

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

## The Local Stack

| Model | Role |
| -- | -- |
| Qwen3.5-9B GGUF Q5_K_M (llama.cpp) | Language model |
| Qwen3-ASR-1.7B | Speech recognition |
| Qwen3-TTS-12Hz-1.7B-Base | Text-to-speech |
| bge-small-en-v1.5 | Semantic note search |

## Reporting a Problem

Submit a [Bug Report](https://github.com/liampetti/fulloch/issues/new?labels=bug) or a [Feature Request](https://github.com/liampetti/fulloch/issues/new?labels=enhancement).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tools and submit changes.

## Credits

Voices in `data/voices/`:
- **`atticus` / `tulloch`** - generated with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (Apache-2.0) from text descriptions; synthetic, not clones of real people
- **`cori`** - sample from [Piper](https://github.com/rhasspy/piper) `en_GB/cori/high` by Bryce Beattie, trained on LibriVox recordings (MIT / public domain)

## License

MIT - see [LICENSE](LICENSE).

***

[releases-shield]: https://img.shields.io/github/release/liampetti/fulloch.svg?style=for-the-badge
[releases]: https://github.com/liampetti/fulloch/releases
[license-shield]: https://img.shields.io/github/license/liampetti/fulloch.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacs-shield]: https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge
