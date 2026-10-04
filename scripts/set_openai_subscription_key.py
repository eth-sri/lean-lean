#!/usr/bin/env python3
"""Explicitly import a Codex login token into secret.sh without displaying it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from leanlean.subscription_auth import (  # noqa: E402
    SUBSCRIPTION_ACCOUNT_ENV,
    SUBSCRIPTION_KEY_ENV,
    subscription_auth_record,
)


def import_key(auth_file: Path, secret_file: Path) -> None:
    if secret_file.is_symlink():
        raise ValueError("secret.sh must be a regular file, not a symlink")
    try:
        auth = json.loads(auth_file.read_text())
    except (OSError, ValueError):
        raise ValueError("Cannot read the Codex login file; run codex login first") from None
    if not isinstance(auth, dict):
        raise ValueError("The Codex login file must contain a JSON object")
    tokens = auth.get("tokens", auth)
    if not isinstance(tokens, dict):
        raise ValueError("The Codex login file contains no subscription token")
    token = tokens.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("No subscription access token found; use codex login with ChatGPT")
    values = {SUBSCRIPTION_KEY_ENV: token}
    account = tokens.get("account_id")
    if isinstance(account, str) and account:
        values[SUBSCRIPTION_ACCOUNT_ENV] = account
    subscription_auth_record(values)

    content = secret_file.read_text() if secret_file.exists() else ""
    # Always replace the account selection alongside the token so an older
    # explicit account cannot silently survive importing another login.
    values.setdefault(SUBSCRIPTION_ACCOUNT_ENV, "")
    for name, value in values.items():
        pattern = re.compile(rf"(?m)^[ \t]*(?:export[ \t]+)?{name}=.*$")
        if len(pattern.findall(content)) > 1:
            raise ValueError(f"Multiple {name} assignments in secret.sh; keep only one")
        assignment = f"export {name}={shlex.quote(value)}"
        if pattern.search(content):
            content = pattern.sub(lambda _match: assignment, content)
        else:
            content = content.rstrip("\n") + "\n" + assignment + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".subscription-secret-", dir=secret_file.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        temporary.chmod(0o600)
        temporary.replace(secret_file)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-codex-login", action="store_true", required=True)
    parser.add_argument("--auth-file", type=Path, default=Path.home() / ".codex/auth.json")
    parser.add_argument("--secret-file", type=Path, default=REPO_ROOT / "secret.sh")
    args = parser.parse_args()
    try:
        import_key(args.auth_file, args.secret_file)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print("Saved OPENAI_SUBSCRIPTION_KEY in secret.sh (mode 0600); token not displayed.")
    print("Run this command again when you choose to replace the token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
