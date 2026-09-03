"""Guard against version drift between the Python package and the HA component.

The same product release is described by two files that can't reference each
other: `pyproject.toml` (read by Python packaging) and
`custom_components/fulloch/manifest.json` (read by HACS / Home Assistant). This
test fails if they disagree, so bumping one without the other is caught in CI.
When releasing, update both — and tag the git release to the same number.
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
MANIFEST = PROJECT_ROOT / "custom_components" / "fulloch" / "manifest.json"


def _pyproject_version() -> str:
    # Regex rather than a TOML parser keeps this small test dependency-free.
    # Anchored at line start so it can't match `target-version = "py311"`.
    match = re.search(
        r'^version\s*=\s*["\'](?P<v>[^"\']+)["\']',
        PYPROJECT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, 'no top-level `version = "..."` found in pyproject.toml'
    return match.group("v")


def _manifest_version() -> str:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["version"]


def test_pyproject_and_manifest_versions_match():
    pyproject = _pyproject_version()
    manifest = _manifest_version()
    assert pyproject == manifest, (
        f"version mismatch: pyproject.toml={pyproject!r} but "
        f"manifest.json={manifest!r}. Bump both to the same value."
    )
