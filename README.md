# Fulloch

[![GitHub Release][releases-shield]][releases] [![License][license-shield]](LICENSE) [![hacs][hacs-shield]][hacs]

<p align="center">
  <img src="fulloch.png" alt="Fulloch Logo" width="200">
</p>

_The **Ful**ly **Loc**al **H**ome Voice Assistant, a private voice layer for your notes, your home, and the web._

Fulloch is your fully private, local voice assistant running on your own PC or Mac. Ask questions, capture thoughts, and search your **[Obsidian](https://github.com/obsidianmd/obsidian-releases)** vault by voice. Control your home via **[Home Assistant](https://github.com/home-assistant)**. Pull live answers from the web with **[SearXNG](https://github.com/searxng/searxng)**. All fully private and running on your home PC.

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
- **Cloned voice** *(GPU override)* - speaks in a voice cloned from a few seconds of reference audio; the default CPU stack uses fast built-in named voices (Kokoro) instead

## Quick installation

The default stack runs on **CPU (mac/linux/windows)**. Audio runs through the browser dashboard. The LLM is either regex-only (simple commands) or off-box via an OpenAI-compatible endpoint you configure in the wizard (e.g. Ollama / LM Studio / another machine on your LAN). The dashboard avatar swaps to Parloch, the **Par**tially-**loc**al **h**ome voice assistant, when the LLM is running off-device.

### CPU (mac/linux/windows)

```bash
docker run -d \
  --name fulloch-ai \
  --restart unless-stopped \
  -p 8765:8765 \
  -e DASHBOARD_HOST=0.0.0.0 \
  -v ./data:/app/data:rw \
  # -v /path/to/your/ObsidianVault:/vault:rw \
  ghcr.io/liampetti/fulloch:cpu
```

### GPU (Linux/Windows + NVIDIA)

Swap `:cpu` for `:latest` (the CUDA image with Qwen3-TTS voice cloning and the on-GPU 9B SLM) and add `--gpus all`:

```bash
docker run -d \
  --name fulloch-ai \
  --restart unless-stopped \
  --gpus all \
  -p 8765:8765 \
  -e DASHBOARD_HOST=0.0.0.0 \
  -v ./data:/app/data:rw \
  # -v /path/to/your/ObsidianVault:/vault:rw \
  ghcr.io/liampetti/fulloch:latest
```

> The commented-out `-v` line exposes your Obsidian vault to Fulloch so voice notes can read/write it. Uncomment it, edit the host path, and add a matching `obsidian.path_translation` entry in `data/config.yml`, see the Obsidian section below.
>
> **Named volumes** (e.g. `-v fulloch-data:/app/data:rw` in compose) work too — the image's entrypoint chowns the volume to the container's user on first boot, so no separate `chown` init container is needed. The bind mount above (the default in this README) inherits the host's UID and is even simpler.

### SearXNG sidecar (optional, for live web answers)

Create a shared network, then run SearXNG on it and point Fulloch at it via `search.searxng_url`:

```bash
docker network create fulloch
docker run -d \
  --name searxng \
  --network fulloch \
  --restart unless-stopped \
  -p 8080:8080 \
  -e SEARXNG_SECRET=change-me \
  searxng/searxng
```

Then in the Fulloch setup wizard, set **Web search URL** to `http://searxng:8080/search`. (Skip the `-p 8080:8080` if you don't need to reach SearXNG from the host, the Fulloch container only needs the in-network name.)

### Once launched
- Open `https://localhost:8765` and follow the wizard. See [First 2 minutes](#first-2-minutes) below for a step-by-step walkthrough of what to expect.

## First 2 minutes

What the first two minutes of a fresh install actually look like, so
nothing in the timeline surprises you:

1. **`docker run` returns in a second.** The container starts and the
   entrypoint chowns the data dir (a no-op on bind mounts; fixes named
   volumes on first boot). The image is ~2 GB for the CPU stack and ~6
   GB for the GPU stack; only download size, not memory.
2. **Open `https://localhost:8765`.** The dashboard's first render
   shows the URL banner with a Copy button and a one-line note about
   the self-signed-cert warning. Click through the warning; it's
   expected for a private LAN install.
3. **Walk the wizard.** Four steps: pick a thinking mode (regex
   simple / full GPU / remote LLM), pick a name and voice, optionally
   connect Home Assistant and SearXNG, optionally connect Obsidian.
   The wizard's preflight (disk + network + GPU) blocks the download
   with a clear error if anything is missing — you'll never see a
   half-finished download.
4. **Models download.** This is the long bit. The CPU stack pulls
   roughly 1.5 GB (Qwen3-ASR 0.6B ONNX + Kokoro 82M); the GPU
   stack pulls roughly 6 GB (Qwen3-ASR 1.7B + Qwen3-TTS 1.7B + the
   9B language model). Plan on **3–10 minutes** on a residential
   connection, longer on a phone hotspot. The download is silent on
   the dashboard but you can watch it live with
   `docker logs -f fulloch-ai` — the assistant's `setup` step
   streams each model's progress.
5. **First wake.** Once the progress bar fills, the assistant
   transitions to `READY` and the mic button on the dashboard
   lights up. The default wakeword is **"hey atticus"**; say it out
   loud and the assistant should reply with a chime. If it doesn't,
   check that the browser tab has microphone permission (the URL bar
   shows a mic icon when granted).

If anything in those five steps goes wrong, the most useful single
command is:

```bash
docker logs -f fulloch-ai
```

…which streams the same logs the dashboard's loading screen renders.
Permission errors, model download failures, and the wizard's
preflight errors all surface there.

## Configuration

The **setup wizard** configures Fulloch on first boot, and the **settings console** (gear icon, the same web UI) edits every option afterwards. Everything is reachable from the UI: wakeword, barge-in, voice, the Home Assistant connection, notes path, and web search. But if you'd rather hand-edit, the full annotated reference is [`data/config.example.yml`](data/config.example.yml).

Secrets (HA token, LLM API key, dashboard password) are stored in `data/credentials.json`, written by the setup wizard. Copying `data/` to a new machine transfers everything. To configure headlessly, copy [`data/credentials.example.json`](data/credentials.example.json) to `data/credentials.json` and fill in the values, or set the equivalent env vars (`HA_TOKEN`, `LLM_API_KEY`, `DASHBOARD_PASSWORD`, `OBSIDIAN_TOKEN`) in `.env`.

## Web Dashboard

The web dashboard is **unauthenticated and bound to `127.0.0.1` by default**. To reach it from another device, set `dashboard_host: "0.0.0.0"` in config and set a password in the setup wizard's finish step.

### OpenAI Endpoint

<img src="parloch.png" alt="Fulloch being Parloch" width="180" align="right">

The moment you point the language model at an OpenAI-compatible endpoint, the avatar and favicon swap to a travelling version of the character (Let's call him *Parloch*: The **Par**tially **loc**al **h**ome voice assistant), and the tagline reads *"language model is off-device."* Pick a local model again (**None** or the GPU 9B) and Fulloch comes home. It triggers as soon as a remote endpoint is *configured*, even if it is on your home network.

> **Note:** GBNF grammar is passed through to llama.cpp-family servers (llama-server, LM Studio, Unsloth Studio), so the remote path enforces the same action/reply shape as the local one. Other OpenAI-compatible endpoints ignore the undocumented `grammar` field and fall back to `response_format: json_object` plus a one-shot repair round-trip, so tool-call errors are more likely there.

## Home Assistant Integration

A HACS-installable integration for status sensors, mic control, proactive speech, and automation triggers.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=liampetti&repository=fulloch&category=Integration)

Or manually: HACS → **Custom repositories** → paste `https://github.com/liampetti/fulloch`, category **Integration** → **Download** → restart HA → **Settings → Integrations → Add → Fulloch**.

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

Use events in automations, e.g. dim lights on `fulloch_wakeword_detected`, restore on `fulloch_turn_ended`. Use `fulloch.speak` for proactive notifications from your home.

The dashboard's **Entities** tab blocks specific entities (locks, alarms) from voice control without affecting dashboard or automation access. Changes apply immediately.

> Search-by-name music queries (*"play the Beatles"*, *"play jazz in the kitchen"*, *"play music everywhere"*) need a `spotify:` block in `config.yml`, Fulloch searches Spotify directly via the Web API, then hands the resolved track/playlist off to a Home Assistant `media_player` entity to actually play. Requires a one-time manual OAuth step to get a refresh token. See the `spotify:` section in `data/config.example.yml` for the config keys and `data/credentials.json` fields. Without a `spotify:` block, there's no music search, pause/resume/skip still work through Home Assistant regardless (so they keep working for AVR/TV too).

## Obsidian Integration

Connect your Obsidian vault so Fulloch reads, writes, appends, and searches your notes by voice, and knows which note you have open. The first time you connect, Fulloch offers to copy your existing notes into the vault as `Inbox/fulloch-import/`. Cloud sync (Remotely Save, etc.) is unchanged, Fulloch only sees the local vault.

**Setup (about 2 minutes):**

1. **In the Fulloch setup wizard**, the "Connect Obsidian" step lets you auto-detect your vault or paste its path. Click **Save and continue**, or **Skip** to do it later.
2. **Open the dashboard's Obsidian tab** and click **Show install instructions** to get a download link and your auth token.
3. **In Obsidian**, extract the downloaded `plugin.zip` into `<your-vault>/.obsidian/plugins/fulloch/`, then enable the Fulloch plugin in **Settings → Community plugins** and paste the token.

That's it. Once the plugin connects, the dashboard flips to **Connected** and voice notes go straight to your vault. Closing Obsidian doesn't break anything, Fulloch remembers the vault and voice keeps working.

To point Fulloch at a different vault later, use the **Switch vault** section on the Obsidian tab.

**Running Fulloch in Docker?** The plugin runs on the host, so it reports host paths (e.g. `/Users/you/Documents/MyVault`). Fulloch inside the container can't see those. Two edits to wire it up:

1. Add your vault's directory as a volume when launching Docker container `-v /Users/you/Documents/MyVault:/vault:rw`
2. Add the same mapping under `obsidian.path_translation` in `data/config.yml`:

   ```yaml
   obsidian:
     path_translation:
       "/Users/you/Documents/MyVault": "/vault"
   ```

Restart Fulloch. The Obsidian wizard's "Auto-detect" scans the container filesystem for a vault, so it'll find `/vault` (or wherever you mounted it) without further config, path translation above is still needed so the plugin's *host* paths resolve correctly once connected.

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
