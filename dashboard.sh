#!/usr/bin/env bash
set -euo pipefail
# Never echo sourced credentials, even when launched with bash -x.
set +x
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f secret.sh ]]; then
    set -a
    if ! source ./secret.sh >/dev/null 2>&1; then
        printf '%s\n' 'Could not load secret.sh; check its shell syntax.' >&2
        exit 1
    fi
    set +a
fi
# The container probes talk to the rootless daemon. DOCKER_HOST does not
# survive a fresh shell, so a dashboard launched outside tmux would otherwise
# report every run as having zero containers.
if [[ -z "${DOCKER_HOST:-}" ]]; then
    rootless_socket="/run/user/$(id -u)/docker.sock"
    if [[ -S "$rootless_socket" ]]; then
        export DOCKER_HOST="unix://$rootless_socket"
    fi
    unset rootless_socket
fi

exec .venv/bin/python run_dashboard.py --root "$PWD" "$@"
