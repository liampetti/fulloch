#!/usr/bin/env python3
"""Validate a device bundle and generate its ESP-IDF NVS image."""

import argparse
import csv
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


class BundleError(ValueError):
    pass


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path} must contain a JSON object")
    return value


def require_keys(value: dict, keys: set[str], name: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = ", ".join(sorted(keys - actual))
        extra = ", ".join(sorted(actual - keys))
        details = ", ".join(
            part
            for part in (
                f"missing: {missing}" if missing else "",
                f"unknown: {extra}" if extra else "",
            )
            if part
        )
        raise BundleError(f"{name} has invalid fields ({details})")


def require_string(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise BundleError(f"{name} must be a non-empty string of at most {maximum} UTF-8 bytes")
    return value


def require_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise BundleError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def validate(config: dict, credentials: dict) -> list[tuple[str, str, str]]:
    config_keys = {"schema_version", "device", "server", "audio", "music"}
    credential_keys = {"schema_version", "wifi", "fulloch"}
    optional_config_keys = {key for key in ("display", "sonos") if key in config}
    require_keys(config, config_keys | optional_config_keys, "config")
    require_keys(
        credentials,
        credential_keys | ({"spotify"} if "spotify" in credentials else set()),
        "credentials",
    )
    if config["schema_version"] != 1 or credentials["schema_version"] != 1:
        raise BundleError("bundle schema_version must be 1")

    device = config["device"]
    server = config["server"]
    audio = config["audio"]
    display = config.get("display")
    wifi = credentials["wifi"]
    fulloch = credentials["fulloch"]
    for value, keys, name in (
        (device, {"id", "name", "timezone"}, "device"),
        (server, {"url", "certificate_pem"}, "server"),
        (wifi, {"ssid", "password"}, "wifi"),
        (fulloch, {"device_token"}, "fulloch"),
    ):
        if not isinstance(value, dict):
            raise BundleError(f"{name} must be an object")
        require_keys(value, keys, name)
    if not isinstance(audio, dict):
        raise BundleError("audio must be an object")
    require_keys(
        audio,
        {"speaker_volume_percent", "aec_enabled"}
        | ({"uplink_channels"} if "uplink_channels" in audio else set()),
        "audio",
    )
    if display is not None:
        if not isinstance(display, dict):
            raise BundleError("display must be an object")
        require_keys(display, {"rotation_degrees"}, "display")

    if "spotify" in credentials:
        spotify_credentials = credentials["spotify"]
        if not isinstance(spotify_credentials, dict):
            raise BundleError("spotify credentials must be an object")
        require_keys(
            spotify_credentials,
            {"client_id", "client_secret", "refresh_token"},
            "spotify credentials",
        )
        require_string(spotify_credentials["client_id"], "spotify.client_id", 256)
        require_string(spotify_credentials["client_secret"], "spotify.client_secret", 256)
        require_string(spotify_credentials["refresh_token"], "spotify.refresh_token", 512)
    music = config["music"]
    if not isinstance(music, dict):
        raise BundleError("music must be an object")
    require_keys(music, {"player"}, "music")
    if not isinstance(music["player"], str) or music["player"] not in {"spotify", "sonos", "none"}:
        raise BundleError("music.player must be spotify, sonos or none")
    if ("sonos" in config) != ("music" in config and config["music"]["player"] == "sonos"):
        raise BundleError("sonos configuration requires music.player set to sonos")
    if ("spotify" in credentials) != (music["player"] == "spotify"):
        raise BundleError("Spotify credentials require music.player set to spotify")
    if "sonos" in config:
        sonos = config["sonos"]
        if not isinstance(sonos, dict):
            raise BundleError("sonos must be an object")
        require_keys(sonos, {"host"}, "sonos")
        require_string(sonos["host"], "sonos.host", 63)

    server_url = require_string(server["url"], "server.url", 256)
    parsed_url = urlsplit(server_url)
    if (
        parsed_url.scheme != "wss"
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
    ):
        raise BundleError("server.url must be a wss:// URL without user credentials")
    certificate = server["certificate_pem"]
    if not isinstance(certificate, str) or len(certificate.encode("utf-8")) > 3999:
        raise BundleError("server.certificate_pem must be a PEM certificate of at most 3999 bytes")
    if not re.fullmatch(
        r"-----BEGIN CERTIFICATE-----\n[A-Za-z0-9+/=\n]+-----END CERTIFICATE-----\n?", certificate
    ):
        raise BundleError("server.certificate_pem must contain one public PEM certificate")
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=certificate)
    except (ValueError, ssl.SSLError) as error:
        raise BundleError("server.certificate_pem is not a valid X.509 certificate") from error
    if not isinstance(audio["aec_enabled"], bool):
        raise BundleError("audio.aec_enabled must be a boolean")
    uplink_channels = require_integer(
        audio.get("uplink_channels", 1), "audio.uplink_channels", 1, 2
    )
    if uplink_channels == 2 and not audio["aec_enabled"]:
        raise BundleError("audio.uplink_channels requires audio.aec_enabled")
    rotation_degrees = require_integer(
        display.get("rotation_degrees", 0) if display else 0, "display.rotation_degrees", 0, 180
    )
    if rotation_degrees not in (0, 180):
        raise BundleError("display.rotation_degrees must be 0 or 180")

    values = [
        ("device", "id", require_string(device["id"], "device.id", 64)),
        ("device", "name", require_string(device["name"], "device.name", 64)),
        ("device", "timezone", require_string(device["timezone"], "device.timezone", 64)),
        ("server", "url", server_url),
        ("server", "certificate", certificate),
        (
            "audio",
            "speaker_volume",
            str(
                require_integer(
                    audio["speaker_volume_percent"], "audio.speaker_volume_percent", 0, 100
                )
            ),
        ),
        ("audio", "aec_enabled", str(audio["aec_enabled"]).lower()),
        ("audio", "uplink_channels", str(uplink_channels)),
        ("display", "rotation", str(rotation_degrees)),
        ("wifi", "ssid", require_string(wifi["ssid"], "wifi.ssid", 32)),
        ("wifi", "password", require_string(wifi["password"], "wifi.password", 63)),
        (
            "fulloch",
            "device_token",
            require_string(fulloch["device_token"], "fulloch.device_token", 256),
        ),
    ]
    if "spotify" in credentials:
        values.extend(
            [
                ("spotify", "client_id", spotify_credentials["client_id"]),
                ("spotify", "client_secret", spotify_credentials["client_secret"]),
                ("spotify", "refresh_token", spotify_credentials["refresh_token"]),
            ]
        )
    if "music" in config:
        values.append(("music", "player", config["music"]["player"]))
    if "sonos" in config:
        values.append(("sonos", "host", config["sonos"]["host"]))
    return [(namespace, key, value) for namespace, key, value in values]


def write_csv(entries: list[tuple[str, str, str]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("key", "type", "encoding", "value"))
        namespace = None
        for entry_namespace, key, value in entries:
            if entry_namespace != namespace:
                writer.writerow((entry_namespace, "namespace", "", ""))
                namespace = entry_namespace
            writer.writerow((key, "data", "string", value))


def generate_nvs(
    config: dict, credentials: dict, *, size: int = 0x10000, nvs_generator: Path | None = None
) -> bytes:
    """Generate an NVS partition without leaving secret files behind."""
    if size != 0x10000:
        raise BundleError("NVS size must match the 0x10000-byte firmware partition")
    entries = validate(config, credentials)
    if nvs_generator is not None and not nvs_generator.is_file():
        raise BundleError("NVS generator not found")
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "nvs.csv"
        image_path = Path(directory) / "nvs.bin"
        write_csv(entries, csv_path)
        generator = (
            [str(nvs_generator)]
            if nvs_generator is not None
            else ["-m", "esp_idf_nvs_partition_gen"]
        )
        result = subprocess.run(
            [sys.executable, *generator, "generate", str(csv_path), str(image_path), hex(size)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0 or not image_path.is_file():
            raise BundleError("NVS generation failed; check that the pinned generator is installed")
        data = image_path.read_bytes()
        if len(data) != size:
            raise BundleError("NVS generator produced an unexpected image size")
        return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "stage"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True, type=Path)
        command_parser.add_argument("--credentials", required=True, type=Path)
        if command == "stage":
            command_parser.add_argument("--output", required=True, type=Path)
            command_parser.add_argument(
                "--nvs-generator",
                type=Path,
                help="ESP-IDF generator script (defaults to installed Python package)",
            )
            command_parser.add_argument("--size", required=True)
    args = parser.parse_args()

    try:
        config = load_json(args.config)
        credentials = load_json(args.credentials)
        if args.command == "stage":
            try:
                size = int(args.size, 0)
            except ValueError as error:
                raise BundleError("NVS size must be an integer") from error
            image = generate_nvs(config, credentials, size=size, nvs_generator=args.nvs_generator)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(image)
        else:
            validate(config, credentials)
    except (BundleError, OSError) as error:
        print(f"provisioning error: {error}", file=sys.stderr)
        return 1
    print("provisioning bundle is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
