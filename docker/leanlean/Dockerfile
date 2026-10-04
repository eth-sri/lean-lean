# Prebuilt per-instance image for the LeanLean benchmark.
#
# Clones a target Lean 4 repo, installs elan (which auto-resolves the right
# toolchain from the repo's lean-toolchain file on the first lake command),
# warms the mathlib cache if the project depends on mathlib, runs a full
# `lake build` to populate .lake/, then resets git to a clean single-commit
# state. Every subsequent `docker run` starts from this warm state, so
# applied patches trigger only an incremental recompilation.
#
# Build one image with:
#
#   docker build \
#       -f docker/leanlean/Dockerfile \
#       --build-arg REPO_URL=https://github.com/scottnarmstrong/DeGiorgi \
#       -t leanlean-degiorgi:latest \
#       .
#
# Or build all instances at once via scripts/build_leanlean_images.sh.

FROM ubuntu:22.04

ARG REPO_URL
ARG REPO_COMMIT=""
# Optional space-separated lake build targets. Empty = build the repo's default
# target(s); naming targets builds only a subset of a large repository.
ARG BUILD_TARGETS=""
# Optional Lean runtime thread cap for source builds. Empty preserves the toolchain default.
ARG LEAN_BUILD_THREADS=""
# Optional space-separated top-level source paths to remove before building.
# This physically isolates multiple libraries that share one upstream monorepo.
ARG REMOVE_SOURCE_PATHS=""

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.elan/bin:${PATH}"

# Base deps. elan needs curl + ca-certificates, lake clones deps via git, and
# some Lean projects' native build hooks expect a working C toolchain.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        python3 && \
    rm -rf /var/lib/apt/lists/*

# Install elan. --default-toolchain none means no Lean version is installed
# up front; elan auto-resolves the right one from the repo's lean-toolchain
# file during the `lake build` below.
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /tmp/elan-init.sh && \
    sh /tmp/elan-init.sh -y --default-toolchain none && \
    rm /tmp/elan-init.sh

# Clone the target repo and (optionally) pin to a specific commit.
RUN git clone "${REPO_URL}" /testbed
WORKDIR /testbed
RUN if [ -n "${REPO_COMMIT}" ]; then git checkout "${REPO_COMMIT}"; fi
RUN for path in ${REMOVE_SOURCE_PATHS}; do rm -rf -- "${path}"; done

# Warm caches. `lake exe cache get` downloads prebuilt mathlib .olean files
# when the project depends on mathlib; it's a no-op otherwise, which we
# swallow. The subsequent `lake build` populates .lake/ for the project's
# own sources and is the slow step on first build.
RUN lake exe cache get || true
RUN if [ -n "${LEAN_BUILD_THREADS}" ]; then \
        LEAN_NUM_THREADS="${LEAN_BUILD_THREADS}" lake build ${BUILD_TARGETS}; \
    else \
        lake build ${BUILD_TARGETS}; \
    fi

# Reset git history so the model sees a clean single-commit state and has no
# upstream to rebase against.
RUN git rev-parse HEAD > /.upstream_commit && \
    rm -rf .git && \
    git init && \
    git config user.name "A U Thor" && \
    git config user.email "author@example.com" && \
    printf '.lake/\n' > .gitignore && \
    git add . && \
    git commit -m "init" && \
    git rev-parse HEAD > /.init_commit

# leanlean's DockerEnvironment runs the container with `sleep <container_timeout>`
# and issues commands via `docker exec`; CMD is only hit if the image is run
# standalone, so just sleep forever.
CMD ["sleep", "infinity"]
