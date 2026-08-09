#!/bin/sh
# Fulloch container entrypoint.
#
# Fixes the ownership of $FULLOCH_DATA_DIR (default /app/data) so the
# non-root `appuser` can write to it. Necessary because:
#   - bind mounts inherit the host user's UID — usually already correct,
#     the find below no-ops in microseconds;
#   - Docker-named volumes are created root-owned and stay that way
#     until something chowns them. Without this, the first run on a
#     fresh `fulloch-data` volume fails to write config.yml / models.
#
# Idempotent: an already-correct data dir hits the early-return and is
# silent. A wrong-owned dir is fixed and logged. A read-only mount
# (or any other chown failure) is logged on stderr and the container
# continues — the app's first write will fail with a clearer
# PermissionError naming the actual file.
#
# Runs as root, then drops to `appuser` before starting the command.
set -eu

DATA_DIR="${FULLOCH_DATA_DIR:-/app/data}"
TARGET_USER="${FULLOCH_ENTRYPOINT_USER:-appuser}"

run_app() {
    # Keep local shell/test invocations usable; the release image always has
    # appuser, so this branch is never taken in a container.
    if ! TARGET_UID="$(id -u "$TARGET_USER" 2>/dev/null)"; then
        exec "$@"
    fi
    if [ "$(id -u)" = "$TARGET_UID" ]; then
        exec "$@"
    fi
    # `su -c` makes the first command argument shell `$0`; include it when
    # re-executing so `python app.py` does not become `exec app.py`.
    # `su` sanitizes PATH, which hides Conda's Python in the GPU image. Resolve
    # bare command names before changing users, then run that absolute path.
    APP_COMMAND="$1"
    case "$APP_COMMAND" in
        */*) ;;
        *) APP_COMMAND="$(command -v "$APP_COMMAND")" ;;
    esac
    shift
    exec su -s /bin/sh "$TARGET_USER" -c 'exec "$0" "$@"' -- "$APP_COMMAND" "$@"
}

# Bootstrap hasn't run yet — the dir might not exist. In that case
# `core/bootstrap.py:ensure_scaffolding` will create it correctly,
# owned by the running UID (appuser), so there's nothing to fix.
if [ ! -d "$DATA_DIR" ]; then
    run_app "$@"
fi

TARGET_UID="$(id -u "$TARGET_USER")"
TARGET_GID="$(id -g "$TARGET_USER")"
CURRENT="$(stat -c '%u:%g' "$DATA_DIR")"
WANT="${TARGET_UID}:${TARGET_GID}"

if [ "$CURRENT" = "$WANT" ]; then
    # Already correct — the common case for bind mounts and for
    # named volumes after the first boot. Skip both the chown and
    # the log line so this stays a no-op in `docker logs`.
    run_app "$@"
fi

# Wrong owner. Fix it. `find \! -user` prunes already-correct
# subtrees, which matters on a multi-GB data dir on every boot.
echo "Entrypoint: fixing $DATA_DIR ownership to $WANT (was $CURRENT)..."
if find "$DATA_DIR" \! -user "$TARGET_UID" -exec chown -R "$TARGET_UID:$TARGET_GID" {} +; then
    echo "Entrypoint: ownership fixed."
else
    echo "Entrypoint: chown failed (likely a read-only mount); container will continue and fail later with a clearer error." >&2
fi

run_app "$@"
