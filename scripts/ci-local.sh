#!/usr/bin/env bash
# Mirror the Python matrix and commands in .github/workflows/ci.yml.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'uv is required to create the local CI environments.' >&2
    exit 1
fi

for version in 3.11 3.12; do
    venv="$PWD/.venv-ci-$version"
    python="$venv/bin/python"
    if [[ ! -e "$venv" ]]; then
        uv venv --seed --python "$version" "$venv"
    fi
    if [[ ! -x "$python" ]]; then
        printf 'Not a usable virtual environment: %s\n' "$venv" >&2
        exit 1
    fi
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
        "$python" -c 'import sys; assert ".".join(map(str, sys.version_info[:2])) == sys.argv[1], "wrong Python version"' "$version"
    printf '\nRunning CI checks with Python %s\n' "$version"
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
        "$python" -m pip install -r requirements-ci.txt
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
        "$python" -m ruff check .
    env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
        "$python" -m pytest tests/
done
