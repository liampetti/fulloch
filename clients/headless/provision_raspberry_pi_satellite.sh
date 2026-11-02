#!/usr/bin/env bash
# Provision or manage a Fulloch headless satellite on a Raspberry Pi.
# Usage: bash clients/headless/provision_raspberry_pi_satellite.sh [ssh-host] [alsa-card] [action]
# Actions: menu (default), provision, start, stop, disable, remove, status, logs.
# SATELLITE_DIR overrides the directory relative to the remote user's home.
set -euo pipefail

REMOTE_HOST="${1:-raspberry-pi}"
ALSA_CARD="${2:-A300}"
HEADLESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SATELLITE_DIR:-fulloch-client}"
ACTION="${3:-menu}"

usage() {
    printf 'Usage: bash %s [ssh-host] [alsa-card] [action]\n' "$0"
    printf 'Actions: menu, provision, start, stop, disable, remove, status, logs\n'
    printf 'Defaults: raspberry-pi, A300, menu; remote directory: ~/fulloch-client\n'
    printf 'Set SATELLITE_DIR to use another directory under the remote user home.\n'
}

if [[ "$REMOTE_HOST" == "--help" || "$REMOTE_HOST" == "-h" ]]; then
    usage
    exit 0
fi

# These values are embedded in the remote SSH command; keep them shell-safe.
if [[ ! "$TARGET_DIR" =~ ^[a-zA-Z0-9_-][a-zA-Z0-9_./-]*$ || "/$TARGET_DIR/" == *"/../"* ]]; then
    printf 'SATELLITE_DIR must be a relative directory without spaces or .. components.\n' >&2
    exit 1
fi
if [[ ! "$ALSA_CARD" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Invalid ALSA card name: %s\n' "$ALSA_CARD" >&2
    exit 1
fi

provision() {
local remote_script remote_command

for path in "$HEADLESS_DIR/satellite.py" "$HEADLESS_DIR/config.yml" "$HEADLESS_DIR/certs/fulloch-home-ca.crt"; do
    if [[ ! -f "$path" ]]; then
        printf 'Required file is missing: %s\n' "$path" >&2
        exit 1
    fi
done

printf 'Provisioning %s with USB ALSA card %s.\n' "$REMOTE_HOST" "$ALSA_CARD"
printf 'Keep the RGB LED HAT detached until the script completes and the Pi is powered down.\n'

ssh "$REMOTE_HOST" "mkdir -p ~/$TARGET_DIR/certs"
scp -p \
    "$HEADLESS_DIR/satellite.py" \
    "$HEADLESS_DIR/config.yml" \
    "$REMOTE_HOST:~/$TARGET_DIR/"
scp -p "$HEADLESS_DIR/certs/fulloch-home-ca.crt" "$REMOTE_HOST:~/$TARGET_DIR/certs/"

remote_script="$(cat <<'REMOTE'
set -euo pipefail

target_dir="$1"
alsa_card="$2"
target="$HOME/$target_dir"

sudo apt update
sudo apt install -y python3-venv portaudio19-dev libopenblas0

if [[ ! -x "$target/.venv/bin/python" ]]; then
    python3 -m venv "$target/.venv"
fi
"$target/.venv/bin/pip" install --upgrade pip
"$target/.venv/bin/pip" install sounddevice websockets pyyaml numpy rpi_ws281x

if [[ -f "$HOME/.asoundrc" ]]; then
    cp "$HOME/.asoundrc" "$HOME/.asoundrc.before-fulloch"
fi
cat > "$HOME/.asoundrc" <<EOF
pcm.!default {
    type plug
    slave.pcm "hw:${alsa_card},0"
}

ctl.!default {
    type hw
    card ${alsa_card}
}
EOF

boot_config="/boot/firmware/config.txt"
if [[ ! -f "$boot_config" ]]; then
    boot_config="/boot/config.txt"
fi
if grep -q '^dtparam=audio=' "$boot_config"; then
    sudo sed -i 's/^dtparam=audio=.*/dtparam=audio=off/' "$boot_config"
else
    printf '\ndtparam=audio=off\n' | sudo tee -a "$boot_config" >/dev/null
fi
printf 'blacklist snd_bcm2835\n' | sudo tee /etc/modprobe.d/blacklist-snd-bcm2835.conf >/dev/null

"$target/.venv/bin/python" -m py_compile "$target/satellite.py"
printf '\nProvisioning complete. Power down and unplug the Pi before fitting the RGB LED HAT.\n'
printf 'After fitting the HAT, boot the Pi and run:\n'
printf '  cd ~/%s && sudo env HOME="$HOME" .venv/bin/python satellite.py\n' "$target_dir"
REMOTE
)"
printf -v remote_command '%q' "$remote_script"
ssh -t "$REMOTE_HOST" "bash -c $remote_command -- $TARGET_DIR $ALSA_CARD"
}

manage_service() {
    local remote_script remote_command
    # Pass the script as an argument so stdin remains available for sudo prompts.
    remote_script="$(cat <<'REMOTE'
set -euo pipefail
target="$HOME/$1"
action="$2"
service="fulloch-satellite.service"
unit="/etc/systemd/system/$service"
sudo -v

case "$action" in
    start)
        if [[ ! -f "$target/satellite.py" || ! -x "$target/.venv/bin/python" || ! -f "$target/config.yml" ]]; then
            printf 'Missing satellite.py, config.yml, or .venv/bin/python in %s. Provision first or set SATELLITE_DIR.\n' "$target" >&2
            exit 1
        fi
        # Quote systemd paths and reject its expansion/escape characters.
        if [[ "$target" == *[\%\$\"\\]* || "$target" == *$'\n'* || "$target" == *$'\r'* ]]; then
            printf 'Unsupported characters in remote service path: %s\n' "$target" >&2
            exit 1
        fi
        sudo -n tee "$unit" >/dev/null <<EOF
[Unit]
Description=Fulloch headless voice satellite
Wants=network-online.target
After=network-online.target sound.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=root
Environment="HOME=$HOME"
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=$target
ExecStart="$target/.venv/bin/python" "$target/satellite.py"
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF
        sudo -n systemctl daemon-reload
        sudo -n systemd-analyze verify "$unit"
        sudo -n systemctl enable "$service"
        sudo -n systemctl restart "$service"
        printf '\nService started and enabled on boot (root, HOME=%s).\n' "$HOME"
        sudo -n systemctl --no-pager --full status "$service"
        ;;
    stop)
        sudo -n systemctl stop "$service"
        printf 'Stopped. Still enabled on boot; use disable or remove to prevent that.\n'
        ;;
    disable)
        sudo -n systemctl disable --now "$service"
        printf 'Stopped and disabled on boot. Use start to enable it again.\n'
        ;;
    remove)
        if [[ -f "$unit" ]]; then
            sudo -n systemctl disable --now "$service"
            sudo -n rm -- "$unit"
            sudo -n systemctl daemon-reload
            sudo -n systemctl reset-failed "$service" 2>/dev/null || true
        fi
        printf 'Service removed. Satellite files and virtual environment retained in %s.\n' "$target"
        ;;
    status) sudo -n systemctl --no-pager --full status "$service" ;;
    logs) sudo -n journalctl -u "$service" -n 100 -f ;;
esac
REMOTE
)"
    printf -v remote_command '%q' "$remote_script"
    ssh -t "$REMOTE_HOST" "bash -c $remote_command -- $TARGET_DIR $ACTION"
}

if [[ "$ACTION" == "menu" ]]; then
    printf '\nFulloch satellite: %s (~/%s)\n' "$REMOTE_HOST" "$TARGET_DIR"
    printf '  1) Provision/setup\n  2) Install/start service and enable on boot\n'
    printf '  3) Stop service (until next boot or start)\n  4) Stop and disable on boot\n'
    printf '  5) Remove service (keep satellite files)\n  6) Service status\n  7) Follow logs\n  q) Quit\n'
    read -r -p 'Choose an action: ' choice
    case "$choice" in
        1) ACTION=provision ;; 2) ACTION=start ;; 3) ACTION=stop ;;
        4) ACTION=disable ;; 5) ACTION=remove ;; 6) ACTION=status ;;
        7) ACTION=logs ;; q|Q) exit 0 ;;
        *) printf 'Unknown choice.\n' >&2; exit 1 ;;
    esac
fi

case "$ACTION" in
    provision)
        provision
        start_now=n
        read -r -p 'Install/start the service now and enable it on boot? [y/N] ' start_now || true
        if [[ "$start_now" == "y" || "$start_now" == "Y" ]]; then
            ACTION=start
            manage_service
        fi
        ;;
    start|stop|disable|remove|status|logs) manage_service ;;
    *) usage >&2; exit 1 ;;
esac
