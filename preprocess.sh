#!/usr/bin/env bash
# Public preprocessing entrypoint: strip every repository with the pinned
# lean-strip, verify it with the Comparator, and publish the dataset.

set -euo pipefail
cd "$(dirname "$0")"

# Prefer a rootless Docker daemon when DOCKER_HOST is not set.
if [[ -z "${DOCKER_HOST:-}" ]]; then
    rootless_socket="/run/user/$(id -u)/docker.sock"
    if [[ -S "$rootless_socket" ]]; then
        export DOCKER_HOST="unix://$rootless_socket"
    fi
    unset rootless_socket
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run python scripts/preprocess.py "$@"
fi
exec python3 scripts/preprocess.py "$@"
