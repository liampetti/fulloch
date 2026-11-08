"""Public USB release integrity and flash-layout checks."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from server import satellite_usb


@pytest.fixture
def release(tmp_path):
    path = tmp_path / "release"
    shutil.copytree(satellite_usb.DEFAULT_RELEASE, path)
    return path


def _edit_manifest(path, edit):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    edit(manifest)
    manifest_path.write_text(json.dumps(manifest))


def test_packaged_release_is_valid():
    manifest, images = satellite_usb.load_release()
    assert manifest["firmware_version"] == "0.1.0"
    assert set(images) == {
        "bootloader.bin",
        "partition-table.bin",
        "ota-data.bin",
        "application.bin",
    }
    assert "nvs.bin" not in images


def test_firmware_release_matches_satellite_repository_when_available():
    sibling = Path(__file__).resolve().parents[2] / "local-satellite"
    if not sibling.is_dir():
        pytest.skip("satellite checkout not available")
    source = sibling / "firmware" / "waveshare-rev2" / "0.1.0"
    if not source.is_dir():
        pytest.fail("satellite checkout is missing the expected firmware release")
    manifest, images = satellite_usb.load_release()
    assert json.loads((source / "manifest.json").read_text()) == manifest
    for name, content in images.items():
        assert (source / name).read_bytes() == content


def test_corruption_and_missing_image_are_rejected(release):
    (release / "bootloader.bin").write_bytes(b"corrupt")
    with pytest.raises(satellite_usb.ArtifactError, match="integrity"):
        satellite_usb.load_release(release)
    (release / "bootloader.bin").unlink()
    with pytest.raises(satellite_usb.ArtifactError):
        satellite_usb.load_release(release)


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["images"][0].update(file="../../secret"),
        lambda m: m["images"][0].update(offset=1),
        lambda m: m["nvs"].update(size=4096),
        lambda m: m.update(firmware_version="9.9.9"),
        lambda m: m["images"].append(m["images"][0]),
    ],
)
def test_invalid_metadata_is_rejected(release, edit):
    _edit_manifest(release, edit)
    with pytest.raises(satellite_usb.ArtifactError):
        satellite_usb.load_release(release)


def test_partition_layout_checked_even_with_matching_checksum(release):
    path = release / "partition-table.bin"
    data = bytearray(path.read_bytes())
    data[8] ^= 1
    path.write_bytes(data)
    _edit_manifest(
        release, lambda m: m["images"][1].update(sha256=hashlib.sha256(data).hexdigest())
    )
    with pytest.raises(satellite_usb.ArtifactError, match="partition table"):
        satellite_usb.load_release(release)
