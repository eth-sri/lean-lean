"""Verify optional timeout wiring without Docker or provider calls."""
import logging
import subprocess

import pytest

from leanlean.environments.docker import DockerEnvironment, DockerEnvironmentConfig


@pytest.fixture
def gateway(monkeypatch):
    env = object.__new__(DockerEnvironment)
    env.config = DockerEnvironmentConfig(image="fixture", network_policy="model_proxy_only")
    env.container_id = "worker"
    env._model_gateway_target = None
    env.logger = logging.getLogger("relay-test")
    monkeypatch.setattr(env, "_local_ipv4_addresses", lambda: {"192.0.2.1"})
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="relay-id\n", stderr="")

    monkeypatch.setattr("leanlean.environments.docker.subprocess.run", run)
    return env, calls


@pytest.mark.parametrize("timeout", [None, 900])
def test_timeout_reaches_relay_and_repeated_attach_is_idempotent(gateway, timeout):
    env, calls = gateway
    url = "http://192.0.2.1:1234/lc-accounting/scope/v1"
    result = env.attach_model_gateway(url, idle_timeout_seconds=timeout)
    assert result == "http://model-gateway:18080/lc-accounting/scope/v1"
    create = next(c for c in calls if c[1] == "create")
    suffix = ["192.0.2.1", "1234", "18080"] + ([] if timeout is None else ["900"])
    assert create[-len(suffix):] == suffix
    count = len(calls)
    assert env.attach_model_gateway(url, idle_timeout_seconds=timeout) == result
    assert len(calls) == count
    with pytest.raises(RuntimeError, match="cannot change the idle timeout"):
        env.attach_model_gateway(url, idle_timeout_seconds=600)


@pytest.mark.parametrize("timeout", [True, 0, -1, 86401, "900", 1.5])
def test_invalid_timeout_fails_before_any_docker_mutation(gateway, timeout):
    env, calls = gateway
    with pytest.raises(ValueError, match="relay idle timeout"):
        env.attach_model_gateway("http://192.0.2.1:1234", idle_timeout_seconds=timeout)
    assert calls == []


@pytest.mark.parametrize("model,version,expected", [
    ("gemini/gemini-4-argon", "1.4.0", {"idle_timeout_seconds": 900}),
    ("gemini-4-argon", "1.4.0", {"idle_timeout_seconds": 900}),
    ("gemini/gemini-3.8-flash", "1.4.0", {}),
    ("gemini/gemini-4-argon", "1.1.26", {}),
])
def test_timeout_default_is_scoped_to_pinned_argon(model, version, expected):
    from leanlean.antigravity_config import antigravity_gateway_options
    assert antigravity_gateway_options(model, version) == expected


def test_relay_image_pin_survives_shared_tag_changes(monkeypatch):
    pin = "sha256:" + "a" * 64
    monkeypatch.setenv("LEANLEAN_MODEL_RELAY_IMAGE", pin)
    first = DockerEnvironmentConfig(image="fixture")
    second = DockerEnvironmentConfig(image="fixture")
    assert first.model_relay_image == second.model_relay_image == pin


@pytest.mark.parametrize("pin", ["latest", "sha256:bad", "sha256:" + "z" * 64])
def test_relay_pin_rejects_mutable_or_invalid_values(monkeypatch, pin):
    monkeypatch.setenv("LEANLEAN_MODEL_RELAY_IMAGE", pin)
    with pytest.raises(ValueError, match="immutable sha256"):
        DockerEnvironmentConfig(image="fixture")
