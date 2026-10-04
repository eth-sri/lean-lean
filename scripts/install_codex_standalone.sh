#!/usr/bin/env bash
set -euo pipefail

# 0.154.0 is the benchmark default; GPT-6.1 Sol runs on 0.160.0.
version="${1:-0.154.0}"
tag="rust-v${version}"
case "${version}/$(uname -m)" in
  0.154.0/x86_64|0.154.0/amd64)
    triple="x86_64-unknown-linux-musl"
    sha256="fc6e3e3b85f2cf7d664520ee5c66a7fe4aa12bae7d46834f47e2f165fd0d6f78"
    ;;
  0.154.0/aarch64|0.154.0/arm64)
    triple="aarch64-unknown-linux-musl"
    sha256="97d93e11df72d3c26772db019e6ea8bb72c246500d46b98c760839f3240355e6"
    ;;
  0.160.0/x86_64|0.160.0/amd64)
    triple="x86_64-unknown-linux-musl"
    sha256="4fcc47ab57f52ff75363951a8761146cd10c8288bd86fed45487dbb204a16b71"
    ;;
  *)
    echo "No pinned Codex ${version} package for $(uname -m)" >&2
    exit 2
    ;;
esac

asset="codex-package-${triple}.tar.gz"
releases="${CODEX_STANDALONE_RELEASES:-${HOME}/.codex/packages/standalone/releases}"
target="${releases}/${version}-${triple}"
if [[ -x "${target}/bin/codex" ]]; then
  actual="$("${target}/bin/codex" --version)"
  if [[ "${actual}" == "codex-cli ${version}" ]]; then
    echo "Codex CLI ${version} is already installed at ${target}"
    exit 0
  fi
fi
if [[ -e "${target}" ]]; then
  echo "Refusing to overwrite incomplete or mismatched bundle: ${target}" >&2
  exit 1
fi

temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT
archive="${temporary}/${asset}"
url="https://github.com/openai/codex/releases/download/${tag}/${asset}"
curl -fsSL "${url}" -o "${archive}"
printf '%s  %s\n' "${sha256}" "${archive}" | sha256sum -c -
install -d -m 0755 "${temporary}/release"
tar -xzf "${archive}" -C "${temporary}/release"
for path in bin/codex bin/codex-code-mode-host codex-resources/bwrap codex-path/rg; do
  if [[ ! -x "${temporary}/release/${path}" ]]; then
    echo "Official Codex package is missing ${path}" >&2
    exit 1
  fi
done
actual="$("${temporary}/release/bin/codex" --version)"
if [[ "${actual}" != "codex-cli ${version}" ]]; then
  echo "Expected Codex CLI ${version}, found: ${actual}" >&2
  exit 1
fi

install -d -m 0755 "${releases}"
mv "${temporary}/release" "${target}"
echo "Installed Codex CLI ${version} at ${target}"
echo "The eval runner copies this package offline; no container download is used."
