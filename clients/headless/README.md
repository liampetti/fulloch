# Raspberry Pi satellite

Turn a Raspberry Pi and USB conference speaker into a Fulloch voice satellite.
You need a running Fulloch server, a network-connected Pi (Pi 2 or newer
recommended), and a USB speaker with microphone, preferably with echo
cancellation. Install [Raspberry Pi OS Lite](https://www.raspberrypi.com/software/)
(32-bit for an original Pi 2), configure networking and enable SSH.

## Configure and provision

From a computer with this repository and SSH access to the Pi:

1. Add a long random token to `satellite_tokens` in the server's
   `data/config.yml` and restart the server.
2. Copy `clients/headless/example.config.yml` to `clients/headless/config.yml`.
   Set the server address, matching token, and room. If using a private CA,
   set `server.ca_cert` to the **remote** path
   `/home/<pi-user>/fulloch-client/certs/fulloch-home-ca.crt` (adjust for your
   Pi user and `SATELLITE_DIR`).
3. Put the server's trusted CA certificate at
   `clients/headless/certs/fulloch-home-ca.crt`. The script requires and copies
   this file even if TLS is disabled. The config and certificates are ignored
   by Git.
4. Connect the USB speaker and find its ALSA card ID on the Pi with
   `aplay -l`. Keep any RGB LED HAT detached during setup.
5. From the repository root, run:

   ```bash
   bash clients/headless/provision_raspberry_pi_satellite.sh <ssh-host> <alsa-card> provision
   ```

The script copies the configuration, certificate and client over SSH, installs
dependencies, and configures the USB card as the default ALSA device. It backs
up an existing `~/.asoundrc` to `~/.asoundrc.before-fulloch`, disables the Pi's
built-in audio, and blacklists `snd_bcm2835` to avoid the RGB HAT's PWM conflict.
This affects built-in analog/HDMI audio, not USB audio. After provisioning,
power down and unplug the Pi before fitting the HAT; reboot to apply the audio
changes. The optional service runs as root for GPIO access.

The defaults are SSH host `raspberry-pi`, ALSA card `A300`, and remote directory
`~/fulloch-client`. Set `SATELLITE_DIR` to change the directory relative to the
remote user's home. If USB audio needs further adjustment, set `audio.mic_device`
and `audio.speaker_device` in `config.yml`. Leave `audio.full_duplex` disabled
unless the device provides echo cancellation.

## Manage the satellite

Run the script without an action for an interactive menu, or specify an action:

```bash
bash clients/headless/provision_raspberry_pi_satellite.sh <ssh-host> <alsa-card> start
bash clients/headless/provision_raspberry_pi_satellite.sh <ssh-host> <alsa-card> status
bash clients/headless/provision_raspberry_pi_satellite.sh <ssh-host> <alsa-card> logs
```

`start` installs or restarts the systemd service and enables it on boot. `stop`
stops it until the next boot; `disable` stops it and disables startup; `remove`
removes the service but keeps the satellite files. Press Ctrl+C to exit the log
viewer. Run the script with `--help` for usage.

The client detects a Waveshare 32-pixel RGB LED HAT when available. Other Linux
computers can also run the client, but this provisioning script is intended for
Raspberry Pi OS.
