#!/usr/bin/env bash
set -euo pipefail

version="2.25.0"
case "$(uname -m)" in
  x86_64|amd64)
    asset="vibe-linux-x86_64-${version}.zip"
    sha256="2d7b59165143c93e10fd739d9e3b01e413464e2ded6e9ad7a1edc44c773f796f"
    ;;
  aarch64|arm64)
    asset="vibe-linux-aarch64-${version}.zip"
    sha256="7a24bb4c63d9db891613dc152ecf1ecde687d73118605a9959ac19c01a7a0f09"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 2
    ;;
esac

data_root="${XDG_DATA_HOME:-${HOME}/.local/share}"
releases="${VIBE_STANDALONE_RELEASES:-${data_root}/leanlean/vibe/releases}"
target="${releases}/${version}"
if [[ -x "${target}/vibe" && -d "${target}/_internal" ]]; then
  actual="$("${target}/vibe" --version)"
  if [[ "${actual}" == "vibe ${version}" ]]; then
    echo "Mistral Vibe ${version} is already installed at ${target}"
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
url="https://github.com/mistralai/mistral-vibe/releases/download/v${version}/${asset}"
curl -fsSL "${url}" -o "${archive}"
printf '%s  %s\n' "${sha256}" "${archive}" | sha256sum -c -
install -d -m 0755 "${temporary}/release"
unzip -q "${archive}" -d "${temporary}/release"
if [[ ! -x "${temporary}/release/vibe" || ! -d "${temporary}/release/_internal" ]]; then
  echo "Official Vibe archive is missing the executable or bundled runtime" >&2
  exit 1
fi
actual="$("${temporary}/release/vibe" --version)"
if [[ "${actual}" != "vibe ${version}" ]]; then
  echo "Expected vibe ${version}, found: ${actual}" >&2
  exit 1
fi

install -d -m 0755 "${releases}"
mv "${temporary}/release" "${target}"
echo "Installed Mistral Vibe ${version} at ${target}"
echo "The eval runner copies this bundle offline; no container package download is used."
