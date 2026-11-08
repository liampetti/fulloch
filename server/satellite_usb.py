"""Validated public flash artifacts for the Waveshare Rev 2 USB deployment."""

import hashlib
import json
import re
import struct
from pathlib import Path


class ArtifactError(ValueError):
    """The installed public firmware artifacts are missing or inconsistent."""


_IMAGES = (
    ("bootloader.bin", 0, 0x8000),
    ("partition-table.bin", 0x8000, 0x1000),
    ("ota-data.bin", 0x19000, 0x2000),
    ("application.bin", 0x20000, 0x300000),
)
DEFAULT_RELEASE = Path(__file__).parent / "firmware" / "waveshare-rev2" / "0.1.0"


def load_release(directory: Path = DEFAULT_RELEASE) -> tuple[dict, dict[str, bytes]]:
    """Load and validate a release once; returned bytes cannot change between checks and use."""
    directory = Path(directory)
    try:
        with (directory / "manifest.json").open("rb") as source:
            raw = source.read(16385)
        if len(raw) > 16384:
            raise ArtifactError("firmware manifest is too large")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "board",
            "firmware_version",
            "flash_size",
            "images",
            "nvs",
        }:
            raise ArtifactError("invalid firmware manifest")
        version = manifest["firmware_version"]
        nvs = manifest["nvs"]
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["board"] != "waveshare_touch_lcd_185c_box_rev2"
            or type(manifest["flash_size"]) is not int
            or manifest["flash_size"] != 0x1000000
            or not isinstance(version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,30}", version)
            or not isinstance(nvs, dict)
            or set(nvs) != {"offset", "size", "bundle_schema_version"}
            or any(type(value) is not int for value in nvs.values())
            or nvs != {"offset": 0x9000, "size": 0x10000, "bundle_schema_version": 1}
        ):
            raise ArtifactError("firmware manifest is not a supported Rev 2 release")
        images = manifest["images"]
        if not isinstance(images, list) or len(images) != len(_IMAGES):
            raise ArtifactError("firmware manifest has an invalid image list")
        contents = {}
        for image, (name, offset, maximum) in zip(images, _IMAGES, strict=True):
            if (
                not isinstance(image, dict)
                or set(image) != {"file", "offset", "size", "sha256"}
                or image["file"] != name
                or type(image["offset"]) is not int
                or image["offset"] != offset
                or type(image["size"]) is not int
                or not 0 < image["size"] <= maximum
                or not isinstance(image["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", image["sha256"])
            ):
                raise ArtifactError("invalid firmware image metadata")
            with (directory / name).open("rb") as source:
                data = source.read(maximum + 1)
            if len(data) != image["size"] or hashlib.sha256(data).hexdigest() != image["sha256"]:
                raise ArtifactError(f"firmware image {name} failed integrity checks")
            contents[name] = data
        table = contents["partition-table.bin"]
        partitions = {}
        for position in range(0, len(table) - 31, 32):
            magic, typ, subtype, offset, size, name, _ = struct.unpack_from(
                "<HBBII16sI", table, position
            )
            if magic != 0x50AA:
                break
            partitions[name.split(b"\0", 1)[0]] = (typ, subtype, offset, size)
        if (
            partitions.get(b"nvs") != (1, 2, 0x9000, 0x10000)
            or partitions.get(b"otadata") != (1, 0, 0x19000, 0x2000)
            or partitions.get(b"factory") != (0, 0, 0x20000, 0x300000)
        ):
            raise ArtifactError("firmware partition table does not match the Rev 2 flash layout")
        app = contents["application.bin"]
        if (
            len(app) < 80
            or struct.unpack_from("<I", app, 32)[0] != 0xABCD5432
            or app[48:80].split(b"\0", 1)[0] != version.encode("ascii")
        ):
            raise ArtifactError("firmware application version does not match the manifest")
        return manifest, contents
    except ArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError, struct.error) as error:
        raise ArtifactError("cannot load the installed Rev 2 firmware release") from error
