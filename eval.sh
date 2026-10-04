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

if [[ -f secret.sh ]]; then
  source secret.sh
else
  eval_common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  eval_shared_secret="$(dirname "$eval_common_git_dir")/secret.sh"
  if [[ -n "$eval_common_git_dir" && -f "$eval_shared_secret" ]]; then
    source "$eval_shared_secret"
  fi
  unset eval_common_git_dir eval_shared_secret
fi

# Select a named credential only after loading secret.sh. The selector, unlike
# the credential itself, is safe to carry through the tmux launch command.
if [[ -n "${LEANLEAN_CLAUDE_TOKEN_ENV:-}" ]]; then
  if [[ ! "$LEANLEAN_CLAUDE_TOKEN_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    printf '%s\n' 'LEANLEAN_CLAUDE_TOKEN_ENV is not a valid environment variable name' >&2
    exit 2
  fi
  if [[ -z "${!LEANLEAN_CLAUDE_TOKEN_ENV:-}" ]]; then
    printf 'Selected Claude credential %s is not loaded\n' "$LEANLEAN_CLAUDE_TOKEN_ENV" >&2
    exit 2
  fi
  export CLAUDE_CODE_OAUTH_TOKEN="${!LEANLEAN_CLAUDE_TOKEN_ENV}"
fi

exec uv run python scripts/eval.py "$@"
