#!/usr/bin/env python3
"""Interactive authenticated USB fallback for the Waveshare Rev 2 satellite."""

import argparse
import getpass
import hashlib
import http.cookiejar
import importlib.util
import json
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlsplit

BASE = "/api/satellite-provisioning"
RECOVERY = "USB writes cannot be undone; re-enter secrets and flash again if needed."


def ask(label, default=""):
    value = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return value or default


class Client:
    def __init__(self, url, ca_file):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Dashboard URL must be an HTTPS origin without credentials")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("Dashboard URL must not contain a path, query, or fragment")
        self.url = url.rstrip("/")
        context = ssl.create_default_context(cafile=ca_file or None)
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
            urllib.request.HTTPSHandler(context=context),
        )

    def request(self, path, *, data=None, method=None, provisioning=True):
        headers = {"X-Fulloch-Provisioning": "1"} if provisioning else {}
        if data is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(data).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        with self.opener.open(request, timeout=30) as response:
            return response.read()

    def json(self, path, **kwargs):
        return json.loads(self.request(path, **kwargs))


def connect():
    """Authenticate without saving the password or session cookie to disk."""
    url = ask("Dashboard HTTPS origin (e.g. https://fulloch.local:8765)")
    ca = ask("Public CA certificate file for HTTPS trust (blank for system trust)")
    client = Client(url, ca)
    password = getpass.getpass("Dashboard password: ")
    try:
        client.json("/auth/login", data={"password": password}, provisioning=False)
        password = None
        if not client.json("/status", provisioning=False).get("auth_enabled"):
            raise ValueError("Enable dashboard password authentication before provisioning")
    except BaseException:
        password = None
        logout(client)
        raise
    return client, url


def logout(client):
    try:
        client.request("/auth/logout", method="POST", provisioning=False)
    except (OSError, urllib.error.URLError):
        pass


def run():
    print("Waveshare ESP32-S3 Touch LCD 1.85C Box Rev 2 USB setup")
    print("All existing tokenless satellites must be reprovisioned after migration.")
    print("Use a trusted HTTPS dashboard and a LAN address reachable from the satellite.")
    client, url = connect()
    try:
        return provision(client, url)
    finally:
        logout(client)


def provision(client, url):
    if ask("Type REPROVISION to enable mandatory tokens") != "REPROVISION":
        print("Cancelled; no changes made.")
        return 1
    device = ask("Device ID")
    name = ask("Device name")
    timezone = ask("POSIX timezone", "UTC0")
    ssid = ask("Wi-Fi network")
    wifi_password = getpass.getpass("Wi-Fi password: ")
    host = urlsplit(url).netloc
    server_url = ask("Satellite WSS URL", f"wss://{host}/ws/satellite-v2")
    parsed = urlsplit(server_url)
    if (
        parsed.scheme != "wss"
        or not parsed.hostname
        or parsed.path != "/ws/satellite-v2"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Satellite URL must be wss://HOST/ws/satellite-v2")
    cert_file = ask(
        "Public satellite server certificate PEM file (blank for active dashboard certificate)"
    )
    certificate = Path(cert_file).read_text(encoding="utf-8") if cert_file else ""
    volume = int(ask("Speaker volume 0–100", "50"))
    rotation = int(ask("Display rotation 0 or 180", "0"))
    player = ask("Music service: none, sonos, spotify", "none").lower()
    config = {
        "schema_version": 1,
        "device": {"id": device, "name": name, "timezone": timezone},
        "server": {"url": server_url, "certificate_pem": certificate},
        "audio": {"speaker_volume_percent": volume, "aec_enabled": True},
        "display": {"rotation_degrees": rotation},
        "music": {"player": player},
    }
    credentials = {"schema_version": 1, "wifi": {"ssid": ssid, "password": wifi_password}}
    wifi_password = None
    if player == "sonos":
        config["sonos"] = {"host": ask("Sonos host")}
    elif player == "spotify":
        credentials["spotify"] = {
            "client_id": ask("Spotify client ID"),
            "client_secret": getpass.getpass("Spotify client secret: "),
            "refresh_token": getpass.getpass("Spotify refresh token: "),
        }
    elif player != "none":
        raise ValueError("Unknown music service")
    port = ask("USB serial port (e.g. /dev/ttyACM0 or COM3)")
    if not port:
        raise ValueError("Select a USB serial port")
    if importlib.util.find_spec("esptool") is None:
        print("Install esptool first: python -m pip install esptool==4.11.0", file=sys.stderr)
        raise ValueError("Install esptool before starting provisioning")
    transaction = None
    try:
        client.json(BASE + "/migration", data={"confirm_tokenless_reprovisioning": True})
        response = client.json(
            BASE + "/waveshare/transactions", data={"config": config, "credentials": credentials}
        )
        transaction = response["transaction_id"]
        credentials = None
        manifest = client.json(BASE + "/waveshare/manifest")
        with tempfile.TemporaryDirectory(prefix="fulloch-satellite-") as directory:
            paths = []
            for image in manifest["images"]:
                data = client.request(BASE + "/waveshare/images/" + quote(image["file"], safe=""))
                if (
                    len(data) != image["size"]
                    or hashlib.sha256(data).hexdigest() != image["sha256"]
                ):
                    raise ValueError("Firmware integrity check failed")
                path = Path(directory) / image["file"]
                path.write_bytes(data)
                paths += [hex(image["offset"]), str(path)]
            nvs = client.request(BASE + f"/transactions/{quote(transaction, safe='')}/nvs")
            if len(nvs) != manifest["nvs"]["size"]:
                raise ValueError("NVS image does not match manifest")
            nvs_path = Path(directory) / "nvs.bin"
            nvs_path.write_bytes(nvs)
            paths += [hex(manifest["nvs"]["offset"]), str(nvs_path)]
            print("Flashing; do not unplug the board…", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "esptool",
                    "--chip",
                    "esp32s3",
                    "--port",
                    port,
                    "--baud",
                    "115200",
                    "write_flash",
                    *paths,
                ],
                check=True,
            )
        print("Waiting for the expected authenticated device…", flush=True)
        while True:
            state = client.json(BASE + f"/transactions/{quote(transaction, safe='')}")
            if state["state"] == "confirmed":
                transaction = None
                print("Setup complete: the expected device authenticated.")
                return 0
            if state["state"] != "ready":
                raise TimeoutError("The device did not connect before the setup deadline")
            time.sleep(1.5)
    finally:
        if transaction:
            try:
                client.request(
                    BASE + f"/transactions/{quote(transaction, safe='')}", method="DELETE"
                )
            except (OSError, urllib.error.URLError):
                pass


def revoke():
    print("Revoke one device token. The device will be disconnected and must be reprovisioned.")
    client, _ = connect()
    try:
        device = ask("Device ID to revoke")
        if not device:
            raise ValueError("Device ID is required")
        if ask(f"Type REVOKE to revoke {device}") != "REVOKE":
            print("Cancelled; no changes made.")
            return 1
        result = client.json(BASE + f"/devices/{quote(device, safe='')}/token", method="DELETE")
        if not result.get("revoked"):
            print("No owned token or active provisioning candidate found for that device.")
            return 1
        print("Device token revoked. Mandatory authentication remains enabled.")
        return 0
    finally:
        logout(client)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revoke", action="store_true", help="interactively revoke a device token")
    args = parser.parse_args(argv)
    try:
        return revoke() if args.revoke else run()
    except (KeyboardInterrupt, EOFError):
        print(f"\nCancelled. {RECOVERY}", file=sys.stderr)
    except urllib.error.HTTPError as error:
        advice = {
            400: "Check the device fields, server hostname, public certificate, and migration state.",
            401: "Check the dashboard password and sign in again.",
            403: "Enable dashboard password authentication and sign in again.",
            404: "The resource is unavailable or the provisioning attempt has expired.",
            503: "The server firmware release is unavailable or invalid.",
        }.get(error.code, "Check the dashboard connection and try again.")
        print(f"Dashboard request failed (HTTP {error.code}). {advice}", file=sys.stderr)
        if not args.revoke:
            print(RECOVERY, file=sys.stderr)
    except ssl.SSLError:
        print(
            "TLS verification failed. Use the matching HTTPS hostname and public CA certificate.",
            file=sys.stderr,
        )
    except urllib.error.URLError as error:
        if isinstance(error.reason, ssl.SSLError):
            print(
                "TLS verification failed. Use the matching HTTPS hostname and public CA certificate.",
                file=sys.stderr,
            )
        else:
            print(
                "Cannot reach the dashboard. Check the HTTPS address and connection.",
                file=sys.stderr,
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Setup failed ({type(error).__name__}). {RECOVERY}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
