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

docker build   --tag leanlean-model-relay:latest   --file "$repo_root/docker/model-relay/Dockerfile"   "$build_dir"

docker image inspect   --format 'Built {{.RepoTags}} {{.Id}}'   leanlean-model-relay:latest
