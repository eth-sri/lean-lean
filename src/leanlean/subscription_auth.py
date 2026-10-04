"""Explicit, user-managed ChatGPT subscription credentials for the gateway."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections.abc import Mapping, MutableMapping
from typing import Any


SUBSCRIPTION_KEY_ENV = "OPENAI_SUBSCRIPTION_KEY"
SUBSCRIPTION_ACCOUNT_ENV = "OPENAI_SUBSCRIPTION_ACCOUNT_ID"
EXPLICIT_AUTH_ENV = "LEANLEAN_CHATGPT_EXPLICIT_AUTH"
NAMED_SUBSCRIPTION_KEY_ENVS = tuple(
    f"OPENAI_SUBSCRIPTION_KEY_NEW{index}" for index in range(1, 11)
)
_SUBSCRIPTION_KEY_ENV_PATTERN = re.compile(
    r"OPENAI_SUBSCRIPTION_KEY(?:_NEW[1-9][0-9]*)?"
)


def validate_subscription_key_env(name: str) -> str:
    """Validate a non-secret selector accepted in authored model configs."""
    if not isinstance(name, str) or _SUBSCRIPTION_KEY_ENV_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "subscription credential_env must be OPENAI_SUBSCRIPTION_KEY or "
            "OPENAI_SUBSCRIPTION_KEY_NEW followed by a positive integer"
        )
    return name


def subscription_account_env(key_env: str) -> str:
    key_env = validate_subscription_key_env(key_env)
    return key_env.replace("_KEY", "_ACCOUNT_ID", 1)


def subscription_provider_name(key_env: str) -> str:
    key_env = validate_subscription_key_env(key_env)
    if key_env == SUBSCRIPTION_KEY_ENV:
        return "Codex · secret.sh account"
    return "Codex · " + key_env.removeprefix("OPENAI_SUBSCRIPTION_KEY_")


def strip_subscription_credentials(env: MutableMapping[str, str]) -> None:
    """Remove every host-side ChatGPT credential from a child environment."""
    for name in tuple(env):
        if _SUBSCRIPTION_KEY_ENV_PATTERN.fullmatch(name) is not None:
            env.pop(name, None)
        elif name == SUBSCRIPTION_ACCOUNT_ENV or name.startswith(
            SUBSCRIPTION_ACCOUNT_ENV + "_NEW"
        ):
            env.pop(name, None)


def subscription_auth_record(
    env: Mapping[str, str] | None = None,
    *,
    key_env: str = SUBSCRIPTION_KEY_ENV,
) -> dict[str, Any]:
    """Resolve an explicitly supplied bearer token; never read desktop auth."""
    values = os.environ if env is None else env
    key_env = validate_subscription_key_env(key_env)
    token = values.get(key_env, "").strip()
    if not token:
        raise RuntimeError(
            f"Set {key_env} in secret.sh and launch through eval.py "
            "or eval.sh. The subscription runner does not use the desktop login."
        )
    if any(character.isspace() for character in token):
        raise RuntimeError(f"{key_env} must contain only the bearer token")
    if token.startswith("sk-"):
        raise RuntimeError(
            f"{key_env} requires a ChatGPT subscription bearer token, "
            "not an OpenAI Platform API key."
        )
    record: dict[str, Any] = {"access_token": token}
    claims: dict[str, Any] = {}
    try:
        payload = token.split(".")[1]
        decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if isinstance(decoded, dict):
            claims = decoded
    except (ValueError, IndexError, UnicodeError):
        pass
    expiry = claims.get("exp")
    if isinstance(expiry, (int, float)) and not isinstance(expiry, bool):
        if expiry <= time.time() + 60:
            raise RuntimeError(f"{key_env} has expired; update it in secret.sh")
        record["expires_at"] = expiry
    account = values.get(subscription_account_env(key_env), "").strip()
    auth_claims = claims.get("https://api.openai.com/auth")
    if not account and isinstance(auth_claims, dict):
        account = auth_claims.get("chatgpt_account_id")
    if isinstance(account, str) and account:
        record["account_id"] = account
    return record


def activate_subscription_credential(
    key_env: str,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select one named token for legacy gateway consumers in this process."""
    values = os.environ if env is None else env
    record = subscription_auth_record(values, key_env=key_env)
    values[SUBSCRIPTION_KEY_ENV] = record["access_token"]
    account = record.get("account_id")
    if isinstance(account, str) and account:
        values[SUBSCRIPTION_ACCOUNT_ENV] = account
    else:
        values.pop(SUBSCRIPTION_ACCOUNT_ENV, None)
    return record


def install_explicit_chatgpt_auth() -> None:
    """Keep LiteLLM from refreshing credentials or starting a device login."""
    if os.environ.get(EXPLICIT_AUTH_ENV) != "1":
        return
    from litellm.llms.chatgpt.authenticator import Authenticator
    from litellm.llms.chatgpt.common_utils import GetAccessTokenError

    def get_access_token(self: Authenticator) -> str:
        record = self._read_auth_file() or {}
        token = record.get("access_token")
        if not isinstance(token, str) or not token:
            raise GetAccessTokenError(401, "OPENAI_SUBSCRIPTION_KEY is missing; update secret.sh")
        expiry = record.get("expires_at")
        if isinstance(expiry, (int, float)) and expiry <= time.time() + 60:
            raise GetAccessTokenError(401, "OPENAI_SUBSCRIPTION_KEY has expired; update secret.sh")
        return token

    def get_account_id(self: Authenticator) -> str | None:
        account = (self._read_auth_file() or {}).get("account_id")
        return account if isinstance(account, str) and account else None

    Authenticator.get_access_token = get_access_token
    Authenticator.get_account_id = get_account_id
