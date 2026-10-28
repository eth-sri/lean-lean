"""Verify Argon routing and native capability restrictions, without paid calls."""
from pathlib import Path
import json

import pytest
import yaml

from configs.model_constants import MODEL_GEMINI_4_ARGON, get_model_prices
from leanlean.antigravity_config import (
    ANTIGRAVITY_REQUEST_TOOLS,
    antigravity_launch_command,
    antigravity_profile_files,
    validate_antigravity_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_proxy_maps_every_argon_effort_to_exact_provider_model(tmp_path):
    from leanlean.model.litellm_wrapper.litellm_server import LitellmServer
    server = LitellmServer(host="127.0.0.1", log_dir=str(tmp_path), **MODEL_GEMINI_4_ARGON)
    try:
        config = server._build_proxy_config()
        assert config["router_settings"]["num_retries"] == 0
        models = config["model_list"]
        assert {m["model_name"] for m in models} == {
            "gemini/gemini-4-argon", "gemini-4-argon",
            "gemini-4-argon-low", "gemini-4-argon-medium", "gemini-4-argon-high",
        }
        assert all(m["litellm_params"]["model"] == "gemini/gemini-4-argon" for m in models)
        assert all(m["model_info"]["input_cost_per_token"] == 2e-6 for m in models)
    finally:
        server.stop()


@pytest.mark.parametrize("override", [
    {"enable_subagents": True}, {"allowed_skills": ["anything"]},
    {"mcp_servers": ["anything"]}, {"version": "1.1.26"},
    {"version": "9.9.9"}, {"model": "gemini-typo"},
])
def test_contract_rejects_unsafe_or_unverified_runs(override):
    values = dict(model="gemini-4-argon", version="1.4.0", enable_subagents=False,
                  allowed_skills=[], mcp_servers=[])
    with pytest.raises(ValueError):
        validate_antigravity_contract(**(values | override))


def test_legacy_pin_keeps_original_launch_and_profile():
    command = "HOME=private /usr/local/bin/agy --model {model}"
    assert antigravity_launch_command(command, "1.1.26") == command
    assert list(antigravity_profile_files("1.1.26")) == [".gemini/antigravity-cli/settings.json"]


def test_argon_usage_accepts_native_effort_aliases():
    from leanlean.antigravity_usage import AntigravityUsageJournal
    journal = AntigravityUsageJournal()
    journal.ingest({"event": "init", "init": {"model": "gemini-4-argon-high"}})
    usage = dict(input_tokens=1000, output_tokens=100, thinking_tokens=50,
                 cache_read_tokens=200, total_tokens=1100)
    journal.ingest({"event": "step_update", "step_update": {
        "conversation_id": "smoke", "step_index": 1, "state": "DONE", "usage": usage}})
    journal.ingest({"event": "result", "result": {"usage": usage}})
    journal.validate()
    assert journal.cost == pytest.approx(.00302)
    assert get_model_prices("gemini/gemini-4-argon-high") == (2, 10, .1, 0)


def test_config_snapshot_and_pricing_validate():
    from leanlean.pipeline.evaluation import _resolve_native_system_prompt, _resolve_named_pricing
    config = yaml.safe_load((ROOT / "configs/models/google/gemini-4-argon-high.yaml").read_text())
    _resolve_native_system_prompt(config["system_prompt"], root=ROOT,
                                 harness_version=config["harness_version"],
                                 model_path=ROOT / "configs/models/google/gemini-4-argon-high.yaml",
                                 request_sha256=config["generation_parameters"]["request_sha256"])
    _resolve_named_pricing(config["pricing"], model=config["model"], root=ROOT)
    snapshot = json.loads((ROOT / config["system_prompt"]["snapshot"]).read_text())
    assert set(snapshot["payload_audit"]["tool_names"]) == ANTIGRAVITY_REQUEST_TOOLS


def test_pinned_binary_request_excludes_workspace_extensions():
    from leanlean.generators.cli_agent import resolve_antigravity_standalone_tool
    from scripts.capture_harness_request import capture_antigravity
    try:
        resolve_antigravity_standalone_tool("1.4.0")
    except RuntimeError as error:
        pytest.skip(str(error))
    request, count = capture_antigravity("1.4.0", "gemini-4-argon", "high", {})
    assert count >= 1
    assert request["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"

    prompt = request["systemInstruction"]["parts"][0]["text"]
    assert "<identity>" in prompt and "<conversation_transcript>" in prompt
    assert "<subagents>" not in prompt and "<skills>" not in prompt


def test_pinned_launch_disables_updater_with_native_boolean_value():
    command = antigravity_launch_command("/usr/local/bin/agy -p {prompt}", "1.4.0")
    assert "AGY_CLI_DISABLE_AUTO_UPDATE=true " in command
