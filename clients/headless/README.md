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
sudo apt install -y git python3-venv portaudio19-dev libopenblas0
git clone https://github.com/liampetti/fulloch.git
cd fulloch/clients/headless
python3 -m venv .venv
. .venv/bin/activate
pip install sounddevice websockets pyyaml numpy rpi_ws281x
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

If USB audio is not the default device, either set `audio.mic_device` and
`audio.speaker_device` in `config.yml`, or configure ALSA with the card ID from
`python -c "import sounddevice as sd; print(sd.query_devices())"`:

```bash
nano ~/.asoundrc

pcm.!default {
    type plug
    slave.pcm "hw:<USB_CARD_ID>,0"
}

ctl.!default {
    type hw
    card <USB_CARD_ID>
}
```

## Waveshare RGB LED HAT

The client automatically detects the 32-pixel Waveshare RGB LED HAT and shows
a Fulloch-themed simulated spectrum display: a startup wave, then eight
four-pixel bars while the assistant is active. It turns off at idle and does
nothing when the HAT or its driver is unavailable.

Install `rpi_ws281x` with the other Python dependencies. The driver requires
GPIO access, so start the satellite with `sudo` while retaining the user's ALSA
configuration:

```bash
sudo env HOME="$HOME" .venv/bin/python satellite.py
```

### PWM Audio Conflict

The HAT drives its data signal through PWM0 on GPIO 18. Before using it, disable
the Pi's built-in PWM audio driver, which conflicts with that hardware even when
the satellite's microphone and speaker are USB devices:

```bash
sudo sed -i 's/^dtparam=audio=on$/dtparam=audio=off/' /boot/firmware/config.txt
printf 'blacklist snd_bcm2835\n' | sudo tee /etc/modprobe.d/blacklist-snd-bcm2835.conf
sudo reboot
```

This disables the Pi's built-in analog/HDMI audio, not USB audio. The client
refuses to initialize the HAT while `snd_bcm2835` remains loaded, preventing
corrupted LED output. Power down and unplug the Pi before fitting or removing
the HAT.

## Optional: run in the background and start on boot

On Raspberry Pi OS or another systemd-based Linux distribution, install a service
after confirming the manual command works. Stop the manual process with Ctrl+C,
then run the following **on the satellite**, from the directory containing
`satellite.py`, `config.yml`, and `.venv`:

```bash
sudo tee /etc/systemd/system/fulloch-satellite.service >/dev/null <<EOF
[Unit]
Description=Fulloch voice satellite
Wants=network-online.target
After=network-online.target sound.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$(id -un)
Environment="HOME=$HOME"
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=$PWD
ExecStart="$PWD/.venv/bin/python" "$PWD/satellite.py"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

For the **RGB LED HAT**, change `User=...` to `User=root` in the service file
(`sudo nano /etc/systemd/system/fulloch-satellite.service`). Keep `HOME` and the
paths pointing to your installation, matching the manual `sudo env HOME=...`
command above. Otherwise, the service runs as your normal user.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fulloch-satellite.service
```

The satellite now runs independently of SSH, starts on boot, and restarts five
seconds after an unexpected exit. View status and logs with:

```bash
sudo systemctl status fulloch-satellite.service
sudo journalctl -u fulloch-satellite.service -n 100 -f
```

Ctrl+C exits the log viewer without stopping the satellite. To manage the service:

```bash
# Stop now; it will still start on the next boot
sudo systemctl stop fulloch-satellite.service

# Stop now and prevent startup on boot
sudo systemctl disable --now fulloch-satellite.service

# Start again and re-enable startup on boot
sudo systemctl enable --now fulloch-satellite.service
```

To remove the service entirely (keeping the satellite files and configuration):

```bash
sudo systemctl disable --now fulloch-satellite.service
sudo rm /etc/systemd/system/fulloch-satellite.service
sudo systemctl daemon-reload
```

## Other hardware

The same client works on any small Linux computer with Python 3, a network connection,
and USB audio:

- Raspberry Pi 3, 4, or 5.
- Raspberry Pi Zero 2 W, with a USB OTG adapter. Use a powered hub if the
  conference speaker needs more power than the Pi can supply.
- An old x86 mini PC, thin client, or laptop running Debian or Ubuntu.
- An Orange Pi, ODROID, or similar board running a supported Linux image.

An unlocked or modified second-generation Amazon Echo Dot could potentially also run this client but has not been tested yet.
