#!/usr/bin/env bash
# Build one prebuilt Docker image per LeanLean instance.
#
# Each image clones the target repo, installs elan + the right Lean toolchain
# (auto-resolved from the repo's lean-toolchain file), warms the mathlib cache
# if applicable, and runs a full `lake build` — so .lake/ is baked into the
# image and every subsequent `docker run` starts warm.
#
# This is slow on first invocation (minutes-to-tens-of-minutes per image).
# Docker layer caching makes re-running this cheap when nothing changed.
#
# Usage:
#     bash scripts/build_leanlean_images.sh              # build all
#     bash scripts/build_leanlean_images.sh degiorgi     # build one
#
# Keep the INSTANCE list below in sync with LEANLEAN_REPOS in
# src/leanlean/benchmarks/leanlean.py. Commits are optional (empty =
# track upstream HEAD); pin them when you want reproducibility.

set -euo pipefail

cd "$(dirname "$0")/.."
DOCKERFILE="docker/leanlean/Dockerfile"

# instance_id | repo_url | pinned_commit | build targets | source paths to remove
INSTANCES=(
    "degiorgi|https://github.com/scottnarmstrong/DeGiorgi|4c1b3077d3782b24065184df4ba59501b2e56fc7||"
    "strongpnt|https://github.com/math-inc/strongpnt|2f5835c322314f55f1026ec2f139d704b7c45c69||"
    "langlib|https://github.com/nielstron/langlib|4c6de7625d1b9756dfcf844618f1e91343096952||"
    "physlib|https://github.com/leanprover-community/physlib|c6e61dce0a80e9b1139af2d81cac4dac886c4c29|Physlib|QuantumInfo/Capacity QuantumInfo/Channels QuantumInfo/ClassicalInfo QuantumInfo/Entropy QuantumInfo/Measurements QuantumInfo/Operators QuantumInfo/Regularized.lean QuantumInfo/ResourceTheory QuantumInfo/States QuantumInfo.lean PhyslibAlpha PhyslibAlpha.lean"
    "quantuminfo|https://github.com/leanprover-community/physlib|c6e61dce0a80e9b1139af2d81cac4dac886c4c29|QuantumInfo|Physlib/ClassicalFieldTheory Physlib/ClassicalMechanics Physlib/CondensedMatter Physlib/Cosmology Physlib/Electromagnetism Physlib/FluidDynamics Physlib/Mathematics Physlib/Optics Physlib/Particles Physlib/QFT Physlib/QuantumMechanics Physlib/Relativity Physlib/SpaceAndTime Physlib/StatisticalMechanics Physlib/StringTheory Physlib/Thermodynamics Physlib/Units Physlib.lean PhyslibAlpha PhyslibAlpha.lean"
)

build_one() {
    local id="$1"
    local url="$2"
    local commit="${3:-}"
    local default_build_targets="${4:-}"
    local default_remove_paths="${5:-}"
    local build_targets="${BUILD_TARGETS:-${default_build_targets}}"
    local remove_paths="${REMOVE_SOURCE_PATHS:-${default_remove_paths}}"
    local tag="leanlean-${id}:latest"
    echo
    echo "=== building ${tag} (REPO_URL=${url}, REPO_COMMIT=${commit:-HEAD}, BUILD_TARGETS=${build_targets:-<all>}, REMOVE_SOURCE_PATHS=${remove_paths:-<none>}) ==="
    docker build \
        --ulimit nofile=1048576:1048576 \
        -f "${DOCKERFILE}" \
        --build-arg "REPO_URL=${url}" \
        --build-arg "REPO_COMMIT=${commit}" \
        --build-arg "BUILD_TARGETS=${build_targets}" \
        --build-arg "REMOVE_SOURCE_PATHS=${remove_paths}" \
        -t "${tag}" \
        .
    echo
    local upstream init
    upstream=$(docker run --rm --network=none --cap-drop=ALL --security-opt=no-new-privileges:true "${tag}" cat /.upstream_commit 2>/dev/null || echo "unknown")
    init=$(docker run --rm --network=none --cap-drop=ALL --security-opt=no-new-privileges:true "${tag}" cat /.init_commit 2>/dev/null || echo "unknown")
    echo "  upstream_commit (pin in LEANLEAN_REPOS): ${upstream}"
    echo "  init_commit     (/.init_commit in image):       ${init}"
}

target="${1:-}"

for entry in "${INSTANCES[@]}"; do
    IFS="|" read -r id url commit build_targets remove_paths <<< "${entry}"
    if [ -n "${target}" ] && [ "${target}" != "${id}" ]; then
        continue
    fi
    build_one "${id}" "${url}" "${commit}" "${build_targets}" "${remove_paths}"
done

echo
echo "images:"
docker images --filter 'reference=leanlean-*'
