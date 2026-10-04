#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

# Prefer a rootless Docker daemon when DOCKER_HOST is not set.
if [[ -z "${DOCKER_HOST:-}" ]]; then
  rootless_socket="/run/user/$(id -u)/docker.sock"
  if [[ -S "$rootless_socket" ]]; then
    export DOCKER_HOST="unix://$rootless_socket"
  fi
  unset rootless_socket
fi

exec uv run python scripts/postprocess.py "$@"
