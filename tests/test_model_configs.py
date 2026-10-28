"""Validate portable model pins and context selection without a model or Docker."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from configs.model_constants import get_model_prices
from leanlean.generators.cli_agent import CLIAgent, CLIAgentConfig
from leanlean.pipeline import evaluation

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted(ROOT.glob("configs/models/*/*.yaml"))


@pytest.mark.parametrize("path", MODELS, ids=lambda p: str(p.relative_to(ROOT / "configs/models")))
def test_named_model_contract(path, monkeypatch):
    config = yaml.safe_load(path.read_text())
    captured = {}

    class Composed(Exception):
        pass

    def capture(*args, **kwargs):
        captured.update(kwargs["_config"])
        raise Composed

    # All public config, prompt, catalog, pricing and access validation happens
    # before repository materialization, which is outside this test's scope.
    monkeypatch.setattr(evaluation, "resolve_config", capture)
    with pytest.raises(Composed):
        evaluation.resolve_named_config("demo", path.relative_to(ROOT / "configs/models").with_suffix(""), repo_root=ROOT)
    assert captured["reasoning_effort"] == config["reasoning_effort"]
    assert captured.get("context_window") == config.get("context_window")
    request = config["generation_parameters"]["request"]
    effort = request.get("reasoning", {}).get("effort", request.get("output_config", {}).get("effort"))
    if effort is not None:
        assert effort == config["reasoning_effort"]


def _agent(context_window):
    agent = object.__new__(CLIAgent)
    agent.config = CLIAgentConfig(
        cli_name="claude_code_sub", context_window=context_window,
        launch_command="", install_commands=[], post_install_commands=[], post_exec_commands=[],
    )
    agent.model = SimpleNamespace(config=SimpleNamespace(model_kwargs={"reasoning_effort": "high"}))
    return agent


@pytest.mark.parametrize("window,expected", [(None, "claude-opus-5-5"), (1000000, "claude-opus-5-5[1m]")])
def test_claude_context_window_is_opt_in(window, expected):
    args = _agent(window)._normalize_openai_args({
        "model": "anthropic/claude-opus-5-5", "base_url": "http://localhost:1234/v1",
    })
    assert args["model"] == expected
    assert args["base_url"] == "http://localhost:1234"
    assert args["reasoning_effort"] == "high"
    assert get_model_prices(expected) == get_model_prices("claude-opus-5-5")
    assert get_model_prices(expected) is not None


@pytest.mark.parametrize("harness,window", [("codex_sub", 1000000), ("claude_code_sub", 200000), ("claude_code_sub", "1000000"), ("claude_code_sub", True)])
def test_unsupported_context_override_fails_before_materialization(harness, window):
    config = yaml.safe_load((ROOT / "configs/dataset/demo.yaml").read_text())
    config.pop("postprocessing", None)
    config.update(kind=evaluation.CONFIG_KIND, run_id="test-window", model="opus-5.5",
                  harness=harness, reasoning_effort="high", context_window=window,
                  parallelism={"workers": 1, "start_interval_seconds": 0})
    with pytest.raises(ValueError, match="context_window"):
        evaluation.resolve_config(ROOT / "configs/dataset/demo.yaml", repo_root=ROOT, _config=config)


def test_runner_forwards_and_restores_context_window(monkeypatch):
    from scripts import run_codex_sub_yaml as runner
    from copy import deepcopy

    spec = {**runner.FORWARDED_DEFAULTS, **runner.FIXED_SETTINGS,
            "model": "opus-5.5", "generator": "claude_code_sub", "reasoning_effort": "high",
            "context_window": 1000000, "run_id": "test-window", "repos": ["example"],
            "playback": {}, "trajectory": {}}
    original = deepcopy(runner.runner.ALL_GENERATOR_CONFIGS["claude_code_sub"])

    class Launched(Exception):
        pass

    def launch(**kwargs):
        config = runner.runner.ALL_GENERATOR_CONFIGS["claude_code_sub"]
        assert config["context_window"] == 1000000
        # Window size is a client setting, not a provider request parameter.
        assert "context_window" not in kwargs["exec_model_config"]["model_kwargs"]
        raise Launched

    monkeypatch.setattr(runner.runner, "main", launch)
    with pytest.raises(Launched):
        runner.run_experiment(spec)
    restored = runner.runner.ALL_GENERATOR_CONFIGS["claude_code_sub"]
    assert restored.get("context_window") == original.get("context_window")
    assert ("context_window" in restored) == ("context_window" in original)
