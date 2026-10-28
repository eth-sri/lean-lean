#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$build_dir"
}
trap cleanup EXIT

cc -O2 -static -s -Wall -Wextra -Werror   "$repo_root/docker/model-relay/relay.c"   -o "$build_dir/relay"
chmod 0555 "$build_dir/relay"

relay_iid_args=()
if [[ -n "${LEANLEAN_RELAY_IID_FILE:-}" ]]; then
  relay_iid_args=(--iidfile "$LEANLEAN_RELAY_IID_FILE")
fi

docker build "${relay_iid_args[@]}"   --tag leanlean-model-relay:latest   --file "$repo_root/docker/model-relay/Dockerfile"   "$build_dir"

docker image inspect   --format 'Built {{.RepoTags}} {{.Id}}'   leanlean-model-relay:latest
