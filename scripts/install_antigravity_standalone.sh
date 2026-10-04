#!/usr/bin/env bash
set -euo pipefail

version="1.1.26"
case "$(uname -m)" in
  x86_64|amd64)
    platform="linux-x64"
    asset="agy_cli_linux_x64.tar.gz"
    sha256="c47c0726266b3513660b7094bceceecbd03d8ae907786aa269c507ceb7e4ee54"
    ;;
  aarch64|arm64)
    platform="linux-arm64"
    asset="agy_cli_linux_arm64.tar.gz"
    sha256="f595d2f1ae23001afffab9cb9012d054f0e8a02a1e848537f73239ae8d3fbd6d"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 2
    ;;
esac

data_root="${XDG_DATA_HOME:-${HOME}/.local/share}"
releases="${ANTIGRAVITY_STANDALONE_RELEASES:-${data_root}/leanlean/antigravity/releases}"
target="${releases}/${version}-${platform}"
if [[ -x "${target}/agy" ]]; then
  actual="$("${target}/agy" --version)"
  if [[ "${actual}" == "${version}" ]]; then
    echo "Antigravity CLI ${version} is already installed at ${target}"
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
url="https://github.com/google-antigravity/antigravity-cli/releases/download/${version}/${asset}"
curl -fsSL "${url}" -o "${archive}"
printf '%s  %s\n' "${sha256}" "${archive}" | sha256sum -c -
install -d -m 0755 "${temporary}/release"
tar -xzf "${archive}" -C "${temporary}/release"
if [[ ! -x "${temporary}/release/antigravity" ]]; then
  echo "Official Antigravity archive is missing its executable" >&2
  exit 1
fi
actual="$("${temporary}/release/antigravity" --version)"
if [[ "${actual}" != "${version}" ]]; then
  echo "Expected Antigravity CLI ${version}, found: ${actual}" >&2
  exit 1
fi
mv "${temporary}/release/antigravity" "${temporary}/release/agy"

install -d -m 0755 "${releases}"
mv "${temporary}/release" "${target}"
echo "Installed Antigravity CLI ${version} at ${target}"
echo "The eval runner copies this binary offline; no container package download is used."
