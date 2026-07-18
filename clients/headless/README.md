# Headless satellite client

A thin Python client that turns any device with a mic and speaker (Raspberry Pi,
thin client, laptop) into a Fulloch satellite — no browser needed.

## Requirements

```
pip install sounddevice websockets pyyaml numpy
```

## Setup

1. Copy `example.config.yml` to `config.yml` and edit:
   - `server.host` — the IP of the machine running Fulloch
   - `server.token` — one of the tokens from the server's `satellite_tokens:` config list
   - `satellite.room` — a name for this satellite (shows in the busy banner on other devices)
   - `satellite.ha_area` — optional, scopes bare "turn off the lights" to a specific HA room
2. Make sure the server has `satellite_tokens:` configured (in `data/config.yml`) if you're using auth

## Usage

```bash
python satellite.py --config config.yml
```

Ctrl+C to stop. Reconnects automatically on connection loss.

## Config reference

| Key | Default | Description |
|---|---|---|
| `server.host` | `localhost` | Fulloch server address |
| `server.port` | `8765` | Dashboard port |
| `server.ssl` | `true` | Use `wss://` vs `ws://` |
| `server.token` | (none) | `satellite_tokens` entry from server config |
| `satellite.room` | (none) | Label for the busy banner |
| `satellite.ha_area` | (none) | HA area_id for bare light commands |
| `satellite.conversation_mode` | `false` | Exclusive full-duplex mode without a wakeword |
| `satellite.server_vad` | `true` | Server handles endpointing |
| `audio.mic_device` | default | sounddevice input device name/index |
| `audio.speaker_device` | default | sounddevice output device name/index |

Hardware switches can send `{"type":"conversation_mode.set","enabled":true|false}` on the active satellite-v2 socket.
