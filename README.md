# Fulloch

[![GitHub Release][releases-shield]][releases] [![License][license-shield]](LICENSE) [![hacs][hacs-shield]][hacs]

<p align="center">
  <img src="fulloch.png" alt="Fulloch Logo" width="200">
</p>

_The **Ful**ly **Loc**al **H**ome Voice Assistant - private, conversational, 100% on-device._

A voice assistant with agentic memory, web research, and smart-home control running entirely on your own home computer or server. Speech recognition, the language model, and the spoken voice never needs to leave your machine.

Fulloch is the conversational brain on top of your existing setup: it drives **[Home Assistant](https://github.com/home-assistant)** for smart-home control and reads/writes plain **Markdown notes**, so it plugs straight into an **[Obsidian](https://github.com/obsidianmd/obsidian-releases)** vault or any Markdown workflow, it can also search the web and return information summaries using **[SearXNG](https://github.com/searxng/searxng)**.

**Hardware:** the **GPU Tier** all-in-one stack needs a GPU (tested on an NVIDIA RTX 5060 Ti, 16GB VRAM); the **CPU Tier** stacks run on a standard CPU-only PC (tested on an AMD Ryzen 9 7900, 32GB RAM), for simple regex commands fully locally, or with an OpenAI-compatible LLM on another server (e.g. your GPU box) for full conversation.

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
- **Cloned voice** *(Full GPU stack)* - speaks in a voice cloned from a few seconds of reference audio; the CPU stacks use fast built-in named voices instead
- **Web dashboard** - setup wizard, settings console, and a phone-friendly chat UI in one place; voice and text share the same brain and history

## Quick installation

### Docker Compose
- **GPU (no SearXNG)**
```bash
docker compose up -d
```
- **GPU + SearXNG**
```bash
docker compose -f compose.yml -f compose.searxng.yml up -d
```
- **CPU (no SearXNG)**
```bash
docker compose -f compose.yml -f compose.cpu.yml up -d
```
- **CPU + SearXNG**
```bash
docker compose -f compose.yml -f compose.cpu.yml -f compose.searxng.yml up -d
```

### Windows
- **Directsound access for Windows users**
```bat
git clone https://github.com/liampetti/fulloch.git
cd fulloch
pip install -r requirements.txt
launch.bat
```

### Once launched
- Open `http://localhost:8765` and follow the wizard.

## Configuration

The **setup wizard** configures Fulloch on first boot, and the **settings console** (gear icon, the same web UI) edits every option afterwards. Everything is reachable from the UI: wakeword, barge-in, voice, the Home Assistant connection, notes path, and web search. But if you'd rather hand-edit, the full annotated reference is [`data/config.example.yml`](data/config.example.yml).

Secrets live in `.env`, all optional. See [`.env.example`](.env.example).

## Web Dashboard

The web dashboard is **unauthenticated and bound to `127.0.0.1` by default**. To reach it from another device you need to set dashboard host to 0.0.0.0 and set a `FULLOCH_DASHBOARD_TOKEN` in `.env`. 

### OpenAI Endpoint

<img src="parloch.png" alt="Fulloch being Parloch" width="180" align="right">

The moment you point the language model at an OpenAI-compatible endpoint, the avatar and favicon swap to a travelling version of the character (Let's call him *Parloch*: The **Par**tially **loc**al **h**ome voice assistant), and the tagline reads *"language model is off-device."* Pick a local model again (**None** or the GPU 9B) and Fulloch comes home. It triggers as soon as a remote endpoint is *configured*, even if it is on your home network.

> **Important:** If you are using an OpenAI endpoint you will lose the GBNF enforced JSON formatting that comes with the built-in LLM. This means there is a higher risk of tool call errors when running the LLM remotely.

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

## Reporting a Problem

Submit a [Bug Report](https://github.com/liampetti/fulloch/issues/new?labels=bug) or a [Feature Request](https://github.com/liampetti/fulloch/issues/new?labels=enhancement).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tools and submit changes.

## Credits

Voices in `data/voices/`:
- **`atticus` / `tulloch`** - generated with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (Apache-2.0) from text descriptions; synthetic, not clones of real people
- **`cori`** - sample from [Piper](https://github.com/rhasspy/piper) `en_GB/cori/high` by Bryce Beattie, trained on LibriVox recordings (MIT / public domain)
- **`All Kokoro voices`** - generated with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0)

## License

MIT - see [LICENSE](LICENSE).

***

[releases-shield]: https://img.shields.io/github/release/liampetti/fulloch.svg?style=for-the-badge
[releases]: https://github.com/liampetti/fulloch/releases
[license-shield]: https://img.shields.io/github/license/liampetti/fulloch.svg?style=for-the-badge
[hacs]: https://github.com/hacs/integration
[hacs-shield]: https://img.shields.io/badge/HACS-Custom-41BDF5.svg?style=for-the-badge
