#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

if [[ $# -lt 1 ]]; then
  echo "usage: bash resume_eval.sh configs/evaluation/<run>.yaml" >&2
  exit 2
fi

exec uv run python scripts/resume_eval.py "$@"
