# Headless satellite client

A thin Python client that turns any device with a mic and speaker (Raspberry Pi,
thin client, laptop) into a Fulloch satellite — no browser needed. It is also
the reference implementation of the ESP32-S3 `satellite-v2` wire protocol.

## Requirements

```
pip install sounddevice websockets pyyaml numpy
```

## Setup

1. Copy `example.config.yml` to `config.yml` and edit:
    - `server.host` — the IP of the machine running Fulloch
    - `server.token` — one of the tokens from the server's `satellite_tokens:` config list
    - `satellite.room` — a name for this satellite (shows in the busy banner on other devices)
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
| `audio.mic_device` | default | sounddevice input device name/index |
| `audio.speaker_device` | default | sounddevice output device name/index |
| `audio.full_duplex` | `false` | Keep mic live during TTS; requires reliable local AEC |

## Satellite-v2 protocol

The client sends `satellite.hello` first. It requires the server's
`satellite.welcome` contract to be protocol major version 2 with these fixed
audio settings:

| Direction | Encoding | Rate | Channels | Framing |
|---|---|---:|---:|---|
| Uplink | `pcm_s16le` | 16 kHz | 1 | 640 bytes, exactly 20 ms |
| Downlink | `pcm_s16le` | 16 kHz | 1 | Server frames are at most 4 KiB |

The client sends `satellite.health` at the `heartbeat_interval_ms` received in
the welcome message. It reports bounded-capture-queue drops, downlink frames
received before playback starts, and sounddevice capture status events.

Server lifecycle messages are `assistant.state` with one of
`wake_detected`, `listening`, `thinking`, `speaking`, `follow_up`, or `idle`.
The client logs every transition, starts playback on `speaking`, and releases
the playback stream on `follow_up`, `idle`, or `tts.cancel`. In conversation
mode, the server may additionally send `conversation.transcript` and
`conversation.response`; the client logs both.

With the default half-duplex configuration, `speaking` sends
`{"type":"satellite.mute","muted":true}` and stops mic uplink. `follow_up`,
`idle`, and `tts.cancel` unmute it. Set `audio.full_duplex: true` only on a
satellite that performs local AEC, such as the ESP32-S3 integration below.

The server also accepts `satellite.mute`, `satellite.stop`,
`conversation_mode.disable`, and `satellite.health` JSON controls. This
reference client sends health automatically; hardware controls can send the
other supported messages over its active socket.

## ESP32-S3 AEC integration

`satellite-v2` uplink packets are 640-byte, mono `pcm_s16le` frames: 320
samples at 16 kHz, or 20 ms. ESP-SR's SR and FD AEC modes consume 16 kHz,
signed-16 microphone and speaker-reference blocks returned by
`aec_get_chunksize()` (currently 512 samples / 32 ms). The two sizes are
compatible, but the ESP32 satellite must bridge them locally:

- Buffer captured mic samples until a complete AEC block is available, then
  call `aec_process()` with an equally sized, time-aligned speaker reference.
- Queue AEC output and emit it as consecutive 320-sample satellite-v2 frames.
  Do not pad or truncate 32 ms AEC blocks to fit the 20 ms wire packet.
- The downlink is `pcm_s16le` at 16 kHz. Tap the same audio that is queued for
  I2S playback as the AEC reference, accounting for the speaker/I2S queue
  delay. Using received WebSocket timing alone does not provide a time-aligned
  reference.

This adds a small, fixed local buffering delay but requires no server protocol
change. AEC is only useful when the microphone stays live during playback;
the normal half-duplex satellite mode can continue to mute its uplink while
TTS plays.
