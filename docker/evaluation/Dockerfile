# Assemble one hash-pinned source tree before an agent is started.
# The build context excludes local Lake artifacts and Git history.
FROM ubuntu:22.04

ARG REPO_VARIANT=optimized
ARG INSTANCE_ID
ARG SOURCE_TREE_SHA256
ARG BUILD_TARGET=""
ARG LEAN_BUILD_THREADS=8
ARG ELAN_VERSION=v4.2.2

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.elan/bin:${PATH}"
ENV ELAN_NO_OVERRIDE_NOTICE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential python3 zstd && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 4 --retry-delay 2 \
      "https://github.com/leanprover/elan/releases/download/${ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz" \
      -o /tmp/elan.tar.gz && \
    tar -xzf /tmp/elan.tar.gz -C /tmp && \
    /tmp/elan-init -y --default-toolchain none && \
    rm -f /tmp/elan.tar.gz /tmp/elan-init && \
    elan --version

COPY . /testbed
WORKDIR /testbed

RUN test -s lean-toolchain && \
    { test -s lakefile.toml || test -s lakefile.lean; }
RUN lake exe cache get || true
RUN ulimit -n 1048576 && \
    if [ -n "${BUILD_TARGET}" ]; then \
        LEAN_NUM_THREADS="${LEAN_BUILD_THREADS}" lake build ${BUILD_TARGET}; \
    else \
        LEAN_NUM_THREADS="${LEAN_BUILD_THREADS}" lake build; \
    fi

RUN rm -rf .git && \
    git init -q && \
    git config user.name "A U Thor" && \
    git config user.email "author@example.com" && \
    printf '.lake/\n' > .gitignore && \
    git add . && \
    git commit -qm init && \
    git rev-parse HEAD > /.init_commit

LABEL org.openai.leanlean.instance_id="${INSTANCE_ID}" \
      org.openai.leanlean.repo_variant="${REPO_VARIANT}" \
      org.openai.leanlean.source_isolation="standalone_v2" \
      org.openai.leanlean.materialization="build_before_agent_v1" \
      org.openai.leanlean.source_tree_sha256="${SOURCE_TREE_SHA256}"

CMD ["sleep", "infinity"]
