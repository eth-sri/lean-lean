"""Pinned Antigravity API routing and extension-free benchmark profile.

The CLI's init.tools lists its registry, not the tools sent to the model.
Validate the actual GenAI function declarations (see the loopback test).
"""
from __future__ import annotations

import json

import yaml

ANTIGRAVITY_VERSION = "1.4.0"
ANTIGRAVITY_ARGON_IDLE_TIMEOUT_SECONDS = 15 * 60
ANTIGRAVITY_MODELS = frozenset({"gemini-3.8-flash", "gemini-4-argon"})
ANTIGRAVITY_LOCAL_TOOLS = (
    "view_file", "run_command", "write_to_file", "replace_file_content",
    "grep_search", "find_by_name", "list_dir",
)
ANTIGRAVITY_REQUEST_TOOLS = frozenset((*ANTIGRAVITY_LOCAL_TOOLS, "manage_task"))
ANTIGRAVITY_AGENT_NAME = "leanlean-local"

# Captured from the stock 1.4.0 system prompt. Custom agents do not inherit
# these two sections; retain them explicitly, omitting subagent discovery.
# The transcript heading is plain text because custom-agent H1 headings
# become XML sections. Other applicable sections remain native-generated.
ANTIGRAVITY_BASE_INSTRUCTIONS = """<identity>
You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.
You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.
The USER will send you requests, which you must always prioritize addressing. User requests are enclosed within <USER_REQUEST> tags.
</identity>
<conversation_transcript>
Transcripts are located directly at `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript.jsonl` (and `transcript_full.jsonl`).
- Start with `transcript.jsonl` (compact). When `truncated_fields` is present, read only the specific corresponding line in `transcript_full.jsonl`.
- Link conversations using `[<label>](conversation://<conversation-id>)`.

File Format
Transcripts are in JSON Lines (JSONL) format. Each line is a single JSON object representing one "step" or action in the conversation.
Each JSON object contains fields such as:
- `step_index`: The index of the step in the trajectory.
- `source`: The source of the action (e.g., `USER_EXPLICIT`, `MODEL`, `SYSTEM`).
- `type`: The type of the step. Particular steps of interest are `USER_INPUT`, which represents a user's prompt, and `PLANNER_RESPONSE`, which represents the agent's response and tool calls.
- `status`: The status of the step (e.g., `DONE`, `ERROR`).
- `created_at`: The ISO 8601 timestamp of when the step occurred.
- `content`: The text content of the step (e.g., the user's request, the model's response, or tool responses).
- `thinking`: The model's internal reasoning / chain-of-thought (for `PLANNER_RESPONSE` steps).
- `tool_calls`: An array of tool calls made in this step, including their arguments.
- `media`: An array of media (e.g. images) attached to the step, each with a `mime_type` and a `uri` referencing the media file on disk. Media bytes are not stored in the transcript; view the `uri` path to see the media.
- `truncated_fields`: An array of field names that were truncated (e.g., `["content"]`, `["thinking"]`, `["tool_calls"]`). Only present in `transcript.jsonl` when truncation occurred (never in `transcript_full.jsonl`). When present, read the corresponding line in `transcript_full.jsonl` for the complete content.

</conversation_transcript>
"""


def antigravity_profile_files(version: str) -> dict[str, str]:
    """Paths relative to private HOME; preserve the pinned native core guidance."""
    if version not in {"1.1.26", ANTIGRAVITY_VERSION}:
        raise ValueError(f"Unsupported pinned Antigravity version: {version}")
    settings = {
        "modelProvider": "gemini", "enableTelemetry": False,
        "showFeedbackSurvey": False, "notifications": False,
        "toolPermission": "always-proceed", "artifactReviewPolicy": "always-proceed",
    }
    files = {".gemini/antigravity-cli/settings.json": json.dumps(settings, sort_keys=True) + "\n"}
    if version == ANTIGRAVITY_VERSION:
        # Both inheritance switches are required: empty lists alone
        # still discover workspace skills and launch workspace MCP servers.
        agent = {
            "name": ANTIGRAVITY_AGENT_NAME,
            "description": "Native benchmark agent with local coding tools only.",
            "mainAgent": True, "subagent": False,
            "inheritCustomizations": False, "inheritMcp": False,
            "tools": list(ANTIGRAVITY_LOCAL_TOOLS),
            "skills": [], "plugins": [], "mcpServers": [],
        }
        files[f".gemini/config/agents/{ANTIGRAVITY_AGENT_NAME}.md"] = (
            "---\n" + yaml.safe_dump(agent, sort_keys=False) + "\n---\n"
            + ANTIGRAVITY_BASE_INSTRUCTIONS
        )
    return files


def antigravity_launch_command(command: str, version: str) -> str:
    """Keep historical runs pinned; route the current CLI through our relay."""
    if version == "1.1.26":
        return command
    if version != ANTIGRAVITY_VERSION:
        raise ValueError(f"Unsupported pinned Antigravity version: {version}")
    # 1.4.0's API-key catalog omits Argon. Its native GenAI gateway catalog
    # recognizes it and emits gemini-4-argon-{effort}, routed to the exact
    # provider model by our proxy. Pin all helper calls to the same model.
    controls = (
        "AGY_CLI_DISABLE_AUTO_UPDATE=true AGY_CLI_OSC7501=off "
        "AGY_LLM_GATEWAY_URL={base_url} AGY_LLM_GATEWAY_API_KEY={api_key} "
        "AGY_LLM_GATEWAY_WIRE_PROTOCOL=genai AGY_LLM_GATEWAY_MODELS={model} "
    )
    return controls + command.replace(
        "/usr/local/bin/agy ", f"/usr/local/bin/agy --agent {ANTIGRAVITY_AGENT_NAME} ", 1
    )


def validate_antigravity_contract(*, model: str, version: str,
                                  enable_subagents: bool, allowed_skills: list,
                                  mcp_servers: list) -> None:
    if model not in ANTIGRAVITY_MODELS:
        raise ValueError(f"Unsupported Antigravity model: {model}")
    if version not in {"1.1.26", ANTIGRAVITY_VERSION}:
        raise ValueError(f"Unsupported pinned Antigravity version: {version}")
    if model == "gemini-4-argon" and version != ANTIGRAVITY_VERSION:
        raise ValueError("Gemini 4 Argon requires Antigravity 1.4.0")
    if enable_subagents or allowed_skills or mcp_servers:
        raise ValueError("Antigravity benchmark runs require subagents, skills, and MCP disabled")


def antigravity_gateway_options(model: str, version: str) -> dict[str, int]:
    """Break silent Argon streams; native recovery keeps the existing budget."""
    if model.split("/")[-1] == "gemini-4-argon" and version == ANTIGRAVITY_VERSION:
        return {"idle_timeout_seconds": ANTIGRAVITY_ARGON_IDLE_TIMEOUT_SECONDS}
    return {}
