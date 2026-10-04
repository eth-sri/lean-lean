#!/usr/bin/env bash
set -euo pipefail

# 2.1.269 is the benchmark default; Opus 5.5 needs 2.1.280.
version="${1:-2.1.269}"
case "${version}/$(uname -m)" in
  2.1.269/x86_64|2.1.269/amd64)
    asset="claude-linux-x64.tar.gz"
    sha256="2749b0fc61cb5ff9a998431d30e9e447a2214b4c5393708db0800245dd3733eb"
    ;;
  2.1.269/aarch64|2.1.269/arm64)
    asset="claude-linux-arm64.tar.gz"
    sha256="b6834b006482a17618ee41d790dd39eaed9aac10ea6efef190252bd029c84d2b"
    ;;
  2.1.280/x86_64|2.1.280/amd64)
    asset="claude-linux-x64.tar.gz"
    sha256="4239c476881f46f5bd2cccee97c10b1c8073f4ab4ae40e3c829493210e757764"
    ;;
  *)
    echo "No pinned Claude Code ${version} archive for $(uname -m)" >&2
    exit 2
    ;;
esac

releases="${CLAUDE_STANDALONE_RELEASES:-${HOME}/.local/share/claude/versions}"
target="${releases}/${version}"
if [[ -x "${target}" ]]; then
  actual="$("${target}" --version)"
  if [[ "${actual}" == "${version} "* ]]; then
    echo "Claude Code ${version} is already installed at ${target}"
    exit 0
  fi
fi
if [[ -e "${target}" ]]; then
  echo "Refusing to overwrite incomplete or mismatched binary: ${target}" >&2
  exit 1
fi

temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT
archive="${temporary}/${asset}"
url="https://github.com/anthropics/claude-code/releases/download/v${version}/${asset}"
curl -fsSL "${url}" -o "${archive}"
printf '%s  %s\n' "${sha256}" "${archive}" | sha256sum -c -
tar -xzf "${archive}" -C "${temporary}"
if [[ ! -x "${temporary}/claude" ]]; then
  echo "Official Claude archive is missing its executable" >&2
  exit 1
fi
actual="$("${temporary}/claude" --version)"
if [[ "${actual}" != "${version} "* ]]; then
  echo "Expected Claude Code ${version}, found: ${actual}" >&2
  exit 1
fi

install -d -m 0755 "${releases}"
install -m 0555 "${temporary}/claude" "${target}"
echo "Installed Claude Code ${version} at ${target}"
echo "The eval runner copies this binary offline; no container download is used."
