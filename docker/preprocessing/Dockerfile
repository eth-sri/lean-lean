# Source image for one repository: the Lean toolchain, the upstream project at
# its pinned commit (unpacked from the hash-verified archive), and its Lake
# dependencies. Nothing is built, isolated or removed here: lean-strip does all
# of that later, in a container without network access.
FROM ubuntu:22.04

ARG LEAN_TOOLCHAIN_URL
ARG LEAN_TOOLCHAIN_SHA256

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/lean/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential python3 python3-tomli zstd && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 4 --retry-delay 2 "${LEAN_TOOLCHAIN_URL}" \
      -o /tmp/lean.tar.zst && \
    printf "%s  %s\n" "${LEAN_TOOLCHAIN_SHA256}" /tmp/lean.tar.zst \
      | sha256sum -c - && \
    mkdir /opt/lean && \
    tar --use-compress-program=unzstd -xf /tmp/lean.tar.zst \
      -C /opt/lean --strip-components=1 && \
    rm /tmp/lean.tar.zst && \
    lean --version && lake --version

ARG ARCHIVE_MEMBER
ARG ARCHIVE_STRIP_COMPONENTS
ARG REPO_COMMIT
COPY source.tar.gz /tmp/source.tar.gz
RUN mkdir /testbed && \
    tar -xzf /tmp/source.tar.gz -C /testbed \
      --strip-components="${ARCHIVE_STRIP_COMPONENTS}" "${ARCHIVE_MEMBER}" && \
    rm /tmp/source.tar.gz && \
    printf '%s\n' "${REPO_COMMIT}" > /.upstream_commit

WORKDIR /testbed
RUN test -f lean-toolchain && { test -f lakefile.toml || test -f lakefile.lean; }
# Fetch the pinned dependencies, retrying because GitHub throttles bursts of
# anonymous clones, then Mathlib's prebuilt cache where there is one.
RUN for attempt in 1 2 3 4 5 6; do \
      lake env true && exit 0; \
      echo "dependency fetch failed (attempt ${attempt}); retrying" >&2; \
      sleep $((attempt * 60)); \
    done; exit 1
RUN lake exe cache get || true
# Track every archive file, as upstream does, even ones its .gitignore matches
# (some repositories force-add them); lean-strip only sees tracked files.
RUN git init -q && \
    printf '.lake/\n' >> .git/info/exclude && \
    git add -A --force -- . ':(exclude).lake' && \
    git -c user.name=lean-strip -c user.email=lean-strip@example.com \
      commit -qm upstream

ARG INSTANCE_ID
ARG SOURCE_RUN_ID
LABEL org.openai.leanlean.instance_id="${INSTANCE_ID}" \
      org.openai.leanlean.repo_variant="source" \
      org.openai.leanlean.run_id="${SOURCE_RUN_ID}" \
      org.openai.leanlean.source_commit="${REPO_COMMIT}"

CMD ["sleep", "infinity"]
