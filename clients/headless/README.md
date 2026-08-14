# Raspberry Pi satellite

Turn a Raspberry Pi and USB conference speaker into a Fulloch voice satellite.

## What you need

- Raspberry Pi 2 or newer recommended, network connection, and microSD card
- USB conference speaker with microphone (preferably with echo cancellation built-in)
- A running Fulloch server

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to install
**Raspberry Pi OS Lite (32-bit)**. In the Imager customisation screen, set a
username, Wi-Fi if needed, and enable SSH. This is the compatible choice for
an original Raspberry Pi 2.

Plug the conference speaker into USB, then SSH into the Pi and run:

```bash
sudo apt update
sudo apt install -y git python3-venv portaudio19-dev
git clone https://github.com/liampetti/fulloch.git
cd fulloch/clients/headless
python3 -m venv .venv
. .venv/bin/activate
pip install sounddevice websockets pyyaml numpy
cp example.config.yml config.yml
```

On the Fulloch server, add a token to `data/config.yml` and restart it:

```yaml
satellite_tokens:
  - "choose-a-long-random-token"
```

Copy the server certificate to the Pi:

```bash
scp user@fulloch-server:/path/to/fulloch/data/certs/dashboard.crt ~/fulloch.crt
```

Edit `config.yml`:

```yaml
server:
  host: "192.168.1.50"       # Fulloch server IP
  ca_cert: "/home/<user>/fulloch.crt"
  token: "choose-a-long-random-token"

satellite:
  room: "kitchen"
```

Start it:

```bash
.venv/bin/python satellite.py
```

USB conference speakers normally become the default microphone and speaker.
If yours does not, run `python -c "import sounddevice as sd; print(sd.query_devices())"`
and set its name for both `audio` devices in `config.yml`. Keep
`audio.full_duplex` disabled unless the device provides echo cancellation.

## Other hardware

The same client works on any small Linux computer with Python 3, a network connection,
and USB audio:

- Raspberry Pi 3, 4, or 5.
- Raspberry Pi Zero 2 W, with a USB OTG adapter. Use a powered hub if the
  conference speaker needs more power than the Pi can supply.
- An old x86 mini PC, thin client, or laptop running Debian or Ubuntu.
- An Orange Pi, ODROID, or similar board running a supported Linux image.

An unlocked or modified second-generation Amazon Echo Dot could potentially also run this client but has not been tested yet.
