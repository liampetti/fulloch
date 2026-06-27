#!/usr/bin/env python3
"""Bump the Fulloch version in both files that carry it, in one shot.

The version lives in two places that can't reference each other —
`pyproject.toml` (Python packaging) and `custom_components/fulloch/manifest.json`
(HACS / Home Assistant). This script sets both to the same value so they never
drift; `tests/test_version_consistency.py` enforces that they agree.

Usage:
    python scripts/bump_version.py 2.1.6        # set an explicit version
    python scripts/bump_version.py --show       # print the current version

After bumping, commit both files and tag the release, e.g.:
    git commit -am "Release 2.1.6" && git tag v2.1.6
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
MANIFEST = ROOT / "custom_components" / "fulloch" / "manifest.json"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def read_pyproject_version(text: str) -> str:
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        sys.exit('error: no top-level `version = "..."` in pyproject.toml')
    return match.group(1)


def set_pyproject_version(text: str, version: str) -> str:
    return re.sub(
        r'^(version\s*=\s*)["\'][^"\']+["\']',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )


def set_manifest_version(text: str, version: str) -> str:
    # Targeted replace (not a json round-trip) so the rest of the manifest's
    # formatting — e.g. inline `codeowners` arrays — is preserved byte-for-byte.
    new_text, count = re.subn(
        r'("version"\s*:\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
    )
    if count != 1:
        sys.exit('error: no `"version": "..."` in manifest.json')
    return new_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump the Fulloch version.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("version", nargs="?", help="new version, e.g. 2.1.6")
    group.add_argument("--show", action="store_true", help="print the current version and exit")
    args = parser.parse_args()

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    manifest_text = MANIFEST.read_text(encoding="utf-8")

    if args.show:
        print(f"pyproject.toml: {read_pyproject_version(pyproject_text)}")
        print(f"manifest.json:  {json.loads(manifest_text)['version']}")
        return

    new_version = args.version.lstrip("v")
    if not _SEMVER.match(new_version):
        sys.exit(f"error: {new_version!r} is not a MAJOR.MINOR.PATCH version")

    PYPROJECT.write_text(set_pyproject_version(pyproject_text, new_version), encoding="utf-8")
    MANIFEST.write_text(set_manifest_version(manifest_text, new_version), encoding="utf-8")

    print(f"Set version to {new_version} in pyproject.toml and manifest.json.")
    print("Next: commit both files and tag the release (e.g. git tag v" + new_version + ").")


if __name__ == "__main__":
    main()
