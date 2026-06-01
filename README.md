# Fulloch

<p align="center">
  <img src="fulloch.png" alt="Fulloch Logo" width="200">
</p>

> The **Ful**ly **Loc**al **H**ome Voice Assistant

A private, conversational voice assistant with **agentic memory, web research, and note-taking** — all running on your own GPU, with nothing sent to the cloud. Speech recognition, the language model, and the spoken voice are 100% local.

Think of Fulloch as a **"brain" that sits on top of your existing local home setup**. It drives your **[Home Assistant](https://github.com/home-assistant)** for smart-home control, and it reads and writes plain **Markdown notes** — so it plugs straight into an **[Obsidian](https://github.com/obsidianmd/obsidian-releases)** vault (or any other Markdown-based workflow). Ask it to remember things, look things up, take notes, control the house, and hold a back-and-forth conversation about all of it.

## What it does

- **Conversational** — holds context across a turn and across the conversation, so follow-ups like *"and tomorrow?"* or *"make those warmer"* just work. No need to repeat the wakeword for a quick reply.
- **Barge-in** — interrupt the assistant mid-sentence by saying the wakeword again.
- **Agentic memory** — tell it *"remember that bin night is Thursday"* and it keeps that fact across restarts. It builds up a long-term picture of you over time.
- **Markdown notes** — read, write, append, and search a folder of `.md` files by voice, including daily-journal entries. Point it at your **Obsidian vault** and it becomes a hands-free front-end to your existing notes.
- **Searchable notes** — full-text *and* semantic search ("what did I write about the car service?") using a local embedding model. No cloud, no external index.
- **Web research** — pulls live answers from a self-hosted search engine and summarises them into a short spoken reply.
- **Smart-home control** — lights, climate, media, calendar, weather, scenes, and more, all through **Home Assistant**. Whatever you've named an entity in HA is what you say out loud.
- **Thinking mode** — say *"think about X"* for a slower, deeper answer, with the option to interrupt and ask *"what have you got so far?"*
- **Multi-step plans** — one request can fan out to several actions (*"dim the lights, play some music, and drop the AC"*) and the assistant chains steps together when one result feeds the next.
- **Cloned voice** — speaks in a voice cloned from a few seconds of reference audio. Design your own voice from a text description (see [Custom voice](#custom-voice)).
- **Web dashboard** *(optional)* — type to the assistant from your phone or laptop; voice and text share the same brain and history.

## The Local Stack
* **LLM Engine:** Llama.cpp running Qwen3.5-9B GGUF Q5_K_M
* **Speech-to-Text:** Qwen3-ASR-1.7B
* **Text-to-Speech:** Qwen3-TTS-12Hz-1.7B-Base
* **Vector Search:** bge-small-en-v1.5

## Hardware Requirements
* **Minimum:** 16GB VRAM GPU (e.g., RTX 5060ti)

## How it fits your setup

```
        You (voice or text)
                │
        ┌───────▼────────┐
        │    Fulloch     │   local ASR → language model → local TTS
        │   (the brain)  │
        └───┬────────┬───┘
            │        │
   ┌────────▼──┐  ┌──▼───────────────┐
   │   Home    │  │  Markdown notes  │
   │ Assistant │  │ (e.g. Obsidian)  │
   └───────────┘  └──────────────────┘
     your house      your knowledge
```

Fulloch doesn't replace Home Assistant or your note app — it sits on top of them and gives you a single conversational interface to both.

## Requirements

- A **CUDA-capable GPU with 16GB VRAM** (developed on an RTX 5060 Ti 16GB). The full pipeline is GPU-resident; there is no CPU fallback.
- A **microphone and speakers**.
- ~15GB free disk for the model files (downloaded automatically on first run).
- **Docker** + the Docker Compose plugin (for SearXNG web search).

**Linux** is the primary platform. **Windows** is supported via a hybrid approach: SearXNG runs in Docker, and the app runs directly in a Python venv — no audio routing through Docker needed.

## Quick start

### Linux

The launch script checks dependencies, downloads the models, sets up echo cancellation, and starts everything.

```bash
git clone https://github.com/liampetti/fulloch.git
cd fulloch
./launch.sh
```

On first run it creates `data/config.yml` and `.env` from the templates and pauses so you can edit them (see [Configuration](#configuration)). It then downloads all four models, loads PulseAudio echo cancellation for barge-in, and starts the app and SearXNG together via Docker Compose.

### Windows

The Windows launcher runs SearXNG in Docker and the app natively in Python — this avoids the audio-device-passthrough complexity that comes with running inside a container.

**Prerequisites:** Python 3.10+, Docker Desktop, and a CUDA-capable GPU with the NVIDIA driver installed.

```bat
git clone https://github.com/liampetti/fulloch.git
cd fulloch
pip install -r requirements.txt
launch.bat
```

`launch.bat` checks dependencies, downloads all four models, starts SearXNG in Docker, then runs `app.py` in the foreground. Closing the terminal (or Ctrl+C) stops the app; SearXNG keeps running in Docker until you `docker compose -f compose.searxng.yml down`.

> **Note:** the two packages that need special install flags (`qwen-tts` and `flash-attn`) have prebuilt Windows wheels — install them with the same `pip install --no-deps` commands shown in `requirements.txt`, just without the `git+` source flag if a wheel is available for your CUDA version.

> **Tip:** out of the box Fulloch works as a standalone conversational assistant with notes and (if configured) web search. Add the `home_assistant:` block when you want it to control your house.

## Configuration

Everything lives in `data/config.yml`. Credentials (if any) go in `.env`. Both are created from `*.example` templates on first launch.

### General

```yaml
general:
  wakeword: "hey fulloch"      # Activation phrase (case-insensitive; multi-word ok; s↔z tolerant), also acts as fallback if regex option is enabled and fails
  wakeword_pattern: '\b(?:hey|hay|hi)\W+(?:ph|[ft])[uoi]ll?(?:[aeiou]\w*|\W+loo?[ck])' # Optional regex override (case-insensitive); tolerant matcher for ASR mishearings of "fulloch"/"tulloch"
  voice_clone: "atticus"       # which voice in data/voices/ to speak with
  barge_in: "wakeword"        # "off" | "wakeword" (interrupt mid-response)
  follow_up_time: "5s"        # wakeword-free reply window after it finishes speaking

  dashboard_port: 8765        # web chat UI; remove to disable
  dashboard_host: "0.0.0.0"   # "127.0.0.1" = local-only
```

`barge_in` and `follow_up_time` make conversations feel natural — you can cut the assistant off, or fire back a quick follow-up without re-saying the wakeword. Reliable barge-in needs acoustic echo cancellation so the assistant doesn't hear itself; `launch.sh` sets this up automatically (opt out with `FULLOCH_SKIP_AEC=1`).

### Home Assistant (smart-home control)

Smart-home actions run entirely through Home Assistant. Add the block below — `url` and `token` are all you need, everything else is auto-detected from your HA install on startup.

```yaml
home_assistant:
  url: "http://192.168.1.50:8123"
  token: "your_long_lived_access_token"
```

1. Create a Long-Lived Access Token in your HA profile (`http://<your-ha>:8123/profile`).
2. Paste it into the block above.

Entity friendly names are pulled from HA, so whatever you've named a light or media player in the HA UI is what you say out loud. Rename an entity in HA, then restart Fulloch to pick it up. (Leave the block out entirely and the smart-home tools simply don't load.)

### Notes (and Obsidian)

Notes are plain Markdown in a folder Fulloch can read, write, append to, and search. By default they live in `data/notes/`. Notes and facts can not be deleted or modified by the AI assistant, the Web UI or a text/markdown editor is the only way to do this.

To use your **existing Obsidian vault** (or any Markdown folder), add a `notes:` block:

```yaml
notes:
  path: "/path/to/your/obsidian/vault"
  daily_subdir: "daily"   # optional: daily-journal notes go here as YYYY-MM-DD.md
```

Because they're just `.md` files, anything Fulloch writes shows up in Obsidian and vice versa — it reads notes you've written by hand, and your notes sync wherever your vault already syncs. Facts you ask it to remember are stored here too and reloaded every startup.

> **Linux (Docker):** the notes path must be reachable inside the container. The whole `data/` folder is already mounted; to point at a vault elsewhere on the host, add a volume mount for it in `compose.yml`. **Windows (native):** any local path works directly — no volume mount needed.

### Web search

```yaml
search:
  searxng_url: "http://localhost:8080/search"
```

A self-hosted [SearXNG](https://github.com/searxng/searxng) instance is started for you by Docker Compose, so this works out of the box. Remove the block to disable web research.

## Talking to it

Some examples (the assistant figures out intent — you don't need exact phrasing):

| You say | What happens |
|---|---|
| "Play some music" / "Pause" / "Skip" | Media control |
| "Turn on the kitchen lights" / "Dim the bedroom to 50%" | Lights |
| "Set the office to 22 degrees" | Climate |
| "What's on today?" / "What's the weather this week?" | Calendar / weather |
| "Set a timer for 10 minutes" / "What time is it?" | Timers / time |
| "Search for the latest news about…" | Web research |
| "When is bin night?" / "Add milk to my shopping list" | Notes |
| "Remember that my wife is allergic to peanuts" | Long-term memory |
| "Think about how I should plan the trip" | Thinking mode |

## Custom voice

The active voice is a reference pair in `data/voices/`: a `<name>.wav` (a few seconds of clean speech) plus a matching `<name>.txt` transcript. Set `general.voice_clone: "<name>"` to switch.

To create a brand-new voice from a text description (e.g. *"a warm, calm British woman in her 30s"*), run:

```bash
# Linux
./voice_design.sh

# Windows
voice_design.bat
```

Either script downloads the Qwen3 VoiceDesign model (~3.4GB, one-time), then drops into an interactive loop: describe a voice, listen to the result, and save it as a reference pair when you're happy.

## Web dashboard

If `general.dashboard_port` is set, a browser chat UI runs alongside the voice loop — handy for typing from your phone while the assistant keeps listening on the mic. Voice and text share the same conversation and history.

- From the host: `http://localhost:8765/`
- From another device on your network: `http://<host-ip>:8765/` (`hostname -I` shows the host IP)

It scales to a phone screen — add it to your home screen for a one-tap client.

## Home Assistant integration

Fulloch ships a HACS-installable HA integration in `custom_components/fulloch/`. It turns Fulloch into a proper HA device — you can trigger it from automations, react to its events, and control it from any HA dashboard.

**Setup:** copy `custom_components/fulloch/` into your HA config directory, restart HA, then go to Settings → Integrations → Add → Fulloch and enter your Fulloch host and dashboard port.

### Entities

| Entity | Type | Description |
|--------|------|-------------|
| `sensor.fulloch_status` | Sensor | `idle` / `thinking` / `speaking` |
| `sensor.fulloch_last_utterance` | Sensor | Last thing the user said |
| `sensor.fulloch_last_response` | Sensor | Last thing Fulloch said (full text in `full_text` attribute) |
| `switch.fulloch_mic` | Switch | Mute / unmute the microphone |
| `text.fulloch_speak` | Text | Type a message and submit → Fulloch speaks it |
| `text.fulloch_chat` | Text | Type a query → runs the full agent loop and speaks the result |

### Actions

| Action | Fields | Description |
|--------|--------|-------------|
| `fulloch.speak` | `text` | Speak a message through the cloned voice |
| `fulloch.chat` | `text` | Run a full agent-loop query and speak the result |
| `fulloch.mic` | `enabled` | Turn the mic on or off |

### HA events

| Event | Payload | When |
|-------|---------|------|
| `fulloch_wakeword_detected` | `utterance`, `source` | Wakeword heard or voice turn starts |
| `fulloch_turn_ended` | `response`, `source` | Fulloch finishes speaking a response |

Use these in automations — e.g. dim lights when `fulloch_wakeword_detected` fires, restore them on `fulloch_turn_ended`.

### Proactive speech from automations

Any HA automation can make Fulloch speak:

```yaml
action: fulloch.speak
data:
  message: "The front door just opened."
```

For agent-loop queries (calendar, weather, news briefings):

```yaml
action: fulloch.chat
data:
  text: "Tell me today's calendar and weather."
```

If Fulloch is mid-conversation when an automation fires, the proactive message waits politely for the current turn to finish rather than talking over it.

### Lovelace

Embed the existing Fulloch chat dashboard in any HA dashboard with the built-in **Webpage card**:

```yaml
type: webpage
url: http://<fulloch-host>:8765
title: Fulloch
```

## Troubleshooting

**No audio detected** — check the mic is the default input device; lower `SILENCE_THRESHOLD` in `core/audio.py` if it's not picking up speech.

**Models won't load** — confirm ~15GB free disk and a working CUDA install; the model files should be under `data/models/`.

**Home Assistant not responding** — verify the token is current (HA tokens can expire) and that `url` is reachable from inside the container (`curl -H "Authorization: Bearer $TOKEN" $URL/api/`). Set `general.log_level: debug` in `data/config.yml` and look for the "Fetched N entity aliases" line on startup.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tools and submit changes.

## Credits

The bundled voice-clone reference pairs in `data/voices/` are derived from third-party models:

- **`atticus` and `tulloch`** were generated with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (Apache-2.0) from text descriptions — synthetic voices, not clones of any real person.
- **`cori`** is a sample from the [Piper](https://github.com/rhasspy/piper) `en_GB/cori/high` voice by [Bryce Beattie](https://brycebeattie.com/files/tts/), trained on public-domain [LibriVox](https://librivox.org) recordings and distributed in [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) (MIT). Released to the public domain.

## License

MIT — see [LICENSE](LICENSE).
