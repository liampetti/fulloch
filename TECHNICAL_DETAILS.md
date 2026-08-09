## llama.cpp experimental options
> Native MTP speculative decoding and llama.cpp's built-in Flash Attention are
> off by default for every GPU. Enable them individually in the setup wizard or
> Settings only after stability-testing the selected model, driver, and hardware.
> This does not affect the Python FlashAttention 2 dependency used by the GPU
> speech models.

## Docker Data Directory Permissions

Fulloch starts its entrypoint as root only long enough to make `/app/data`
owned by the in-container `appuser` (UID/GID `1000`), then launches the app as
that non-root user. This repairs a fresh Docker named volume automatically.

For a bind mount, the source directory must be writable by the container. If
the entrypoint logs `chown failed` or Fulloch reports `PermissionError`, fix the
host directory, then restart the container:

```bash
sudo chown -R 1000:1000 /path/to/fulloch-data
```

Do not use `--user` or replace the image entrypoint unless the mounted data
directory is already writable by the user you select; either bypasses the
automatic ownership repair. Read-only, root-squashed network mounts, and host
security policies can also prevent the repair.

## Obsidian Information
> The optional Obsidian mount exposes your vault to Fulloch so voice notes can read/write it. Add it before the image name, edit the host path, and add a matching `obsidian.path_translation` entry in `data/config.yml`, see the Obsidian section below.

## SearXNG sidecar (optional, for live web answers)

Create a shared network before starting Fulloch, add `--network fulloch` to the
Fulloch `docker run` command, then run SearXNG on that network and point
Fulloch at it via `search.searxng_url`:

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
4. **Models download.** This is the long bit. The CPU stack pulls
    roughly 5 GB (Qwen3 1.7B ONNX + Kokoro 82M); choosing Pocket TTS instead
    adds its roughly 250 MB English ONNX voice-cloning bundle. The GPU
    stack pulls roughly 13 GB (Qwen3 1.7B PyTorch ASR + Qwen3 1.7B PyTorch TTS + the
   9B language model).
5. **Startup** Once models are downloaded the assistant can go through startup, this will happen everytime you stop and restart Fulloch. This should only take about a minute or two with the Qwen3 TTS taking the longest to warm-up. Once everything is loaded you are presented with the chat screen. You can type text or click to activate voice mode. Selecting "Always Listen" starts exclusive Conversation mode: it skips the wakeword, keeps the mic live during replies, and disconnects other voice satellites until you turn it off (default wakeword is 'Hey Atticus').

## Trusted LAN HTTPS

The default certificate is self-signed, which is enough for browser microphone
permission after accepting the warning once. To remove that warning on home
network devices, create a private Fulloch CA and dashboard certificate:

```bash
python scripts/create_local_ca.py --force
```

The script writes the dashboard certificate to `data/certs/dashboard.crt`, so it
uses the existing HTTPS configuration. It prints the one-time trust command for
Linux, macOS, Windows, iOS/iPadOS, and Android. Install only
`data/certs/fulloch-home-ca.crt` on client devices; never distribute
`fulloch-home-ca.key`. Restart Fulloch after running the script. Add extra names
or addresses before clients use them, for example:

```bash
python scripts/create_local_ca.py --force --host fulloch.home --ip 192.168.1.20
```

## Configuration

The **setup wizard** configures Fulloch on first boot, and the **settings console** (gear icon, the same web UI) edits every option afterwards. Everything is reachable from the UI: wakeword, barge-in, voice, the Home Assistant connection, notes path, and web search. But if you'd rather hand-edit, the full annotated reference is [`data/config.example.yml`](data/config.example.yml).

Secrets are stored in `data/credentials.json`. The setup wizard writes the HA token, LLM API key, dashboard password, and Obsidian token; add integration API tokens manually. Copying `data/` to a new machine transfers everything. To configure headlessly, copy [`data/credentials.example.json`](data/credentials.example.json) to `data/credentials.json` and fill in the values, or set the equivalent env vars (`HA_TOKEN`, `LLM_API_KEY`, `DASHBOARD_PASSWORD`, `OBSIDIAN_TOKEN`) in `.env`.

## Web Dashboard

The web dashboard is **unauthenticated and bound to `127.0.0.1` by default**. To reach it from another device, set `dashboard_host: "0.0.0.0"` in config and set a password in the setup wizard's finish step.

## Voice Satellites

The dashboard is itself a browser voice satellite: open it on any trusted LAN
device, enable Voice mode, and its microphone and speaker connect to the
Fulloch server over WebSocket. Expose the dashboard with
`general.dashboard_host: "0.0.0.0"`, use the server's HTTPS URL, and set a
dashboard password before using it from another device.

For a dedicated Raspberry Pi, laptop, or other native microphone/speaker
device, use [fulloch-satellite](https://github.com/liampetti/fulloch-satellite).
Configure that client with the Fulloch server's LAN address and dashboard port
(normally `8765`), and leave TLS enabled unless the server is deliberately
configured without HTTPS. Add its `server.token` to the server's optional
`satellite_tokens:` list in `data/config.yml` to require satellite
authentication. See the satellite repository for installation and its client
configuration reference.

Conversation mode is exclusive: enabling it disconnects other voice satellites
and rejects their reconnections until Conversation mode is turned off.
You can also ask Fulloch to announce a message through a named connected
satellite, for example: "Tell downstairs that dinner is ready." Browser
satellites use their selected Home Assistant area as their name; native
satellites use their configured room name.

## OpenAI Endpoint Grammar
> **Note:** GBNF grammar is passed through to llama.cpp-family servers (llama-server, LM Studio, Unsloth Studio), so the remote path enforces the same action/reply shape as the local one. Other OpenAI-compatible endpoints ignore the undocumented `grammar` field and fall back to `response_format: json_object` plus a one-shot repair round-trip, so tool-call errors are more likely there.

## HACS Integration
Fulloch exposes a token-authenticated plain-HTTP integration API at port `8766` by default for HACS and future trusted native integrations. Add one or more tokens to `data/credentials.json` before configuring HACS:

```json
{"integration_tokens": ["a-long-random-token"]}
```

Enter the Fulloch host, integration port, and that token in the HACS config flow. This API is separate from dashboard authentication and `satellite_tokens`; set `general.integration_api_enabled: false` to disable it, or change `general.integration_api_port` to use another port. It uses `general.dashboard_host` as its bind address.

Manual install to HACS → **Custom repositories** → paste `https://github.com/liampetti/fulloch`, category **Integration** → **Download** → restart HA → **Settings → Integrations → Add → Fulloch**.

Use events in automations, e.g. dim lights on `fulloch_wakeword_detected`, restore on `fulloch_turn_ended`. Use `fulloch.speak` for proactive notifications from your home.

The dashboard's **Entities** tab blocks specific entities (locks, alarms) from voice control without affecting dashboard or automation access. Changes apply immediately.

> Search-by-name music queries (*"play the Beatles"*, *"play jazz in the kitchen"*, *"play music everywhere"*) need a `spotify:` block in `config.yml`, Fulloch searches Spotify directly via the Web API, then hands the resolved track/playlist off to a Home Assistant `media_player` entity to actually play. Requires a one-time manual OAuth step to get a refresh token. See the `spotify:` section in `data/config.example.yml` for the config keys and `data/credentials.json` fields. Without a `spotify:` block, there's no music search, pause/resume/skip still work through Home Assistant regardless (so they keep working for AVR/TV too).

## Obsidian Integration

Connect your Obsidian vault so Fulloch reads, writes, appends, and searches your notes by voice, and knows which note you have open. The first time you connect, Fulloch offers to copy your existing notes into the vault as `Inbox/fulloch-import/`. Cloud sync (Remotely Save, etc.) is unchanged, Fulloch only sees the local vault.

**Setup (about 2 minutes):**

1. **In the Fulloch setup wizard**, the "Connect Obsidian" step lets you auto-detect your vault or paste its path. Click **Save and continue**, or **Skip** to do it later.
2. **Open the dashboard's Obsidian tab** and click **Show install instructions** to get a download link and your auth token.
3. **In Obsidian**, extract the downloaded `fulloch.zip` into `<your-vault>/.obsidian/plugins/fulloch/`, then enable the Fulloch plugin in **Settings → Community plugins** and paste the token.

   In the plugin settings, use the HTTPS dashboard URL (for example, `https://localhost` with port `8765`). A plain `localhost` value creates an insecure `ws://` connection, which Fulloch redirects and WebSockets cannot follow. The default self-signed dashboard certificate must be trusted on the computer running Obsidian; use the private-CA instructions above for a permanent trust setup.

That's it. Once the plugin connects, the dashboard flips to **Connected** and voice notes go straight to your vault. Closing Obsidian doesn't break anything, Fulloch remembers the vault and voice keeps working.

By default, voice can only create and append notes. While the plugin is connected, the Obsidian tab has an explicit **Enable edit/delete** control for active-editor actions: insert at the live cursor, replace currently selected text, rename the active note, or move it to Obsidian trash. The dashboard tab pulses red while this mode is enabled; disable it again when you are finished.

To point Fulloch at a different vault later, use the **Switch vault** section on the Obsidian tab.

The plugin runs on the host, so it reports host paths (e.g. `/Users/you/Documents/MyVault`). Fulloch inside the container can't see those. Two edits to wire it up:

1. Add your vault's directory as a volume when launching Docker container `-v /Users/you/Documents/MyVault:/vault:rw`
2. Add the same mapping under `obsidian.path_translation` in `data/config.yml`:

   ```yaml
   obsidian:
     path_translation:
       "/Users/you/Documents/MyVault": "/vault"
   ```
