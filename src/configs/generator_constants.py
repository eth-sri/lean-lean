from configs.agent_augmentation import make_variant

CLAUDE_CODE_CLI_VERSION = "2.1.269"
CODEX_CLI_VERSION = "0.154.0"
GEMINI_CLI_VERSION = "0.58.0"
ANTIGRAVITY_CLI_VERSION = "1.1.26"
MUSE_CODE_CLI_VERSION = "1.0.3-R2198.1"
MISTRAL_VIBE_CLI_VERSION = "2.25.0"


QWEN_GENERATOR_CONFIG = {
    "launch_command": ". $HOME/.nvm/nvm.sh && OPENAI_API_KEY={api_key} OPENAI_BASE_URL={base_url} OPENAI_MODEL={model} qwen --yolo -p {prompt}",
    "install_commands": [
        "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash",
        ". ~/.nvm/nvm.sh && nvm install 24",
        ". ~/.nvm/nvm.sh && npm install -g @qwen-code/qwen-code@0.12.6",
        "sudo apt-get install -y ripgrep",
    ],
    "post_install_commands": [
        "mkdir -p .qwen",
        """cat > .qwen/settings.json << 'JSON'
{
  "sessionTokenLimit": 262144,
  "contextFileName": "AGENTS.md",
  "chatCompression": {
    "contextPercentageThreshold": 0.6
  },
  "summarizeToolOutput": {
    "run_shell_command": {
      "tokenBudget": 2000
    }
  }
}
JSON""",
        "cat .qwen/settings.json",
    ],
    "post_exec_commands": [
        "rm -rf .qwen",
    ],
    "cli_name": "qwen_code",
}

CODEX_GENERATOR_CONFIG = {
    # Codex runs with full access *inside* the disposable Docker filesystem.
    # The outer container owns isolation: no host mounts, no capabilities,
    # no-new-privileges, a PID cap, and model-relay-only networking. Nested
    # bwrap cannot create user namespaces under that boundary.
    "launch_command": "RUST_LOG=debug LITELLM_API_KEY={api_key} codex exec -c model_provider=litellm -c model_providers.litellm.name=litellm -c model_providers.litellm.base_url={base_url} -c model={model} -c model_providers.litellm.env_key=LITELLM_API_KEY -c model_providers.litellm.wire_api=responses --sandbox danger-full-access --skip-git-repo-check {prompt}",
    # Codex and its bundled ripgrep are static host binaries streamed in with
    # docker exec. Agent containers never run curl, npm, or apt.
    "install_commands": [],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "codex",
    "host_tool_bundle": "codex_standalone",
    "host_tool_version": CODEX_CLI_VERSION,
}

# ChatGPT-subscription variant. The gateway reads OPENAI_SUBSCRIPTION_KEY from
# secret.sh; the eval container receives only the gateway's ephemeral key.
# Codex JSONL remains the agent-trajectory source.
CODEX_SUB_GENERATOR_CONFIG = {
    "launch_command": (
        ". /tmp/leanlean-subscription/env && "
        "CODEX_HOME=/tmp/leanlean-codex-home "
        "codex exec --json --ephemeral --ignore-user-config "
        "--sandbox danger-full-access --skip-git-repo-check "
        "-c model_provider=litellm "
        "-c model_providers.litellm.name=litellm "
        "-c model_providers.litellm.base_url={base_url} "
        "-c model_providers.litellm.env_key=LITELLM_API_KEY "
        "-c model_providers.litellm.wire_api=responses "
        "-c model_reasoning_effort={reasoning_effort} "
        "--model {model} {prompt}"
    ),
    "install_commands": [],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "codex_sub",
    "requires_model_proxy": True,
    "subscription_transport": "chatgpt",
    "host_tool_bundle": "codex_standalone",
    "host_tool_version": CODEX_CLI_VERSION,
}

# App Server variant: the Python client owns the JSONL handshake and sends one
# ephemeral turn. The launch string is descriptive only; this generator never
# shells the prompt into a command line and never routes through LiteLLM.
CODEX_APP_SERVER_SUB_GENERATOR_CONFIG = {
    **CODEX_SUB_GENERATOR_CONFIG,
    "launch_command": "codex app-server --listen stdio://",
    "post_exec_commands": [],
    "cli_name": "codex_app_server_sub",
    "requires_model_proxy": False,
    "subscription_transport": None,
}

CLAUDE_CODE_GENERATOR_CONFIG = {
    "launch_command": "IS_SANDBOX=1 ANTHROPIC_BASE_URL={base_url} ANTHROPIC_AUTH_TOKEN={api_key} ~/.local/bin/claude --dangerously-skip-permissions --model {model} -p {prompt}",
    "install_commands": [
        f"curl -fsSL https://claude.ai/install.sh | bash -s -- {CLAUDE_CODE_CLI_VERSION}",
        "sudo apt-get install ripgrep",
    ],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "claude_code",
}


# GLM API calls use Claude Code's native stream-json harness through the local
# Anthropic-compatible gateway, with every Claude alias mapped to GLM.
CLAUDE_CODE_GLM_GENERATOR_CONFIG = {
    "launch_command": (
        "env -u CLAUDE_CODE_OAUTH_TOKEN -u ANTHROPIC_API_KEY "
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 "
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
        "ANTHROPIC_BASE_URL={base_url} ANTHROPIC_AUTH_TOKEN={api_key} "
        "ANTHROPIC_DEFAULT_OPUS_MODEL={model} "
        "ANTHROPIC_DEFAULT_SONNET_MODEL={model} "
        "ANTHROPIC_DEFAULT_HAIKU_MODEL={model} "
        "CLAUDE_CODE_SUBAGENT_MODEL={model} "
        "/usr/local/bin/claude --dangerously-skip-permissions "
        "--safe-mode --disable-slash-commands --no-chrome "
        "--no-session-persistence --strict-mcp-config --setting-sources '' "
        "--mcp-config '{{\"mcpServers\":{{}}}}' "
        "--tools Bash,Edit,Read,Write --output-format stream-json --verbose "
        "--model sonnet --effort {reasoning_effort} -p {prompt}"
    ),
    "install_commands": [],
    "post_install_commands": [],
    "post_exec_commands": [],
    # Reuse native Claude stream parsing and checkpoint hooks, not OAuth auth.
    "cli_name": "claude_code_sub",
    "requires_model_proxy": True,
    "subscription_transport": None,
    "gateway_accounting": True,
    "host_tool_bundle": "claude_standalone",
    "host_tool_version": CLAUDE_CODE_CLI_VERSION,
    "enable_subagents": False,
}

# Max/Pro subscription routed through LiteLLM. Claude's OAuth Authorization
# header is forwarded upstream; x-litellm-api-key authenticates only to the
# local gateway. Stream-json remains the authoritative agent-trajectory source.
CLAUDE_CODE_SUB_GENERATOR_CONFIG = {
    "launch_command": (
        ". /tmp/leanlean-subscription/env && "
        "env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN "
        "IS_SANDBOX=1 "
        "ANTHROPIC_BASE_URL={base_url} "
        "/usr/local/bin/claude --dangerously-skip-permissions "
        "--safe-mode --disable-slash-commands --no-chrome "
        "--no-session-persistence --strict-mcp-config "
        "--mcp-config '{{\"mcpServers\":{{}}}}' "
        "--tools Bash,Edit,Read,Write "
        "--output-format stream-json --verbose "
        "--model {model} --effort {reasoning_effort} -p {prompt}"
    ),
    "install_commands": [],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "claude_code_sub",
    "requires_model_proxy": True,
    "subscription_transport": "claude_max",
    "host_tool_bundle": "claude_standalone",
    "host_tool_version": CLAUDE_CODE_CLI_VERSION,
}


# Native Mistral Vibe harness with Vibe's bundled Leanstral prompt. The runner
# installs a checksum-pinned host bundle, creates a one-run custom agent profile,
# and points that profile only at the instance-isolated LiteLLM gateway.
MISTRAL_VIBE_LEAN_GENERATOR_CONFIG = {
    "launch_command": (
        "VIBE_HOME=/tmp/leanlean-vibe-home "
        "LITELLM_API_KEY={api_key} "
        "MISTRAL_API_KEY={api_key} "
        "/opt/leanlean/mistral-vibe/vibe "
        "--prompt {prompt} "
        "--agent leanlean-lean "
        "--auto-approve --trust --output streaming"
    ),
    "install_commands": [],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "mistral_vibe",
    "requires_model_proxy": True,
    "host_tool_bundle": "mistral_vibe_standalone",
    "host_tool_version": MISTRAL_VIBE_CLI_VERSION,
}


GEMINI_CLI_GENERATOR_CONFIG = {
    "launch_command": ". $HOME/.nvm/nvm.sh && GEMINI_API_KEY={api_key} GOOGLE_GEMINI_BASE_URL={base_url} gemini --yolo --model {model} -p {prompt}",
    "install_commands": [
        "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash",
        ". $HOME/.nvm/nvm.sh && nvm install 24",
        f". $HOME/.nvm/nvm.sh && npm install -g @google/gemini-cli@{GEMINI_CLI_VERSION}",
        "sudo apt-get install -y ripgrep",
    ],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "gemini_cli",
}

ANTIGRAVITY_CLI_GENERATOR_CONFIG = {
    "launch_command": (
        "HOME=/tmp/leanlean-antigravity-home "
        "GEMINI_API_KEY={api_key} "
        "GOOGLE_GEMINI_BASE_URL={base_url} "
        "/usr/local/bin/agy --dangerously-skip-permissions "
        "--disable-slash-commands --output-format stream-json "
        "--model {model} --effort {reasoning_effort} "
        "--print-timeout 12h -p {prompt}"
    ),
    "install_commands": [],
    "post_install_commands": [],
    "post_exec_commands": [],
    "cli_name": "antigravity_cli",
    "requires_model_proxy": True,
    "host_tool_bundle": "antigravity_standalone",
    "host_tool_version": ANTIGRAVITY_CLI_VERSION,
}


def add_generator_class(config: dict, generator_class: str) -> dict:
    if "generator_class" not in config:
        config["generator_class"] = generator_class
    return config


ALL_GENERATOR_CONFIGS = {
    # Muse Code at its shipped defaults apart from the benchmark's reasoning effort and the
    # stream timeouts. Muse streams nothing while reasoning and a 128k-token response takes
    # ~10 min, so the native 180 s first-event/idle limits would abort and retry it. Other
    # flags are operational: headless JSON events, the /testbed workspace, and --yolo
    # (approval prompts auto-cancel headless). Like every harness: no skills, MCP servers,
    # plugins, subagents or web search (Codex runs with web_search=disabled). No preset,
    # compaction or tool-output overrides.
    "muse_code_native": {
        "generator_class": "leanlean.generators.muse_code.MuseCodeNativeAgent",
        "launch_command": (
            "XDG_CONFIG_HOME=/tmp/leanlean-muse/config "
            "XDG_DATA_HOME=/tmp/leanlean-muse/data "
            "XDG_CACHE_HOME=/tmp/leanlean-muse/cache "
            "TBH_STREAM_IDLE_TIMEOUT_SECS=1200 TBH_STREAM_FIRST_EVENT_TIMEOUT_SECS=1200 "
            "MUSE_NO_AUTO_UPDATE=1 MUSE_EXPERIMENTAL_PLUGINS=off "
            "META_API_KEY={api_key} /usr/local/bin/muse exec "
            "--json --model {model} --reasoning-effort {reasoning_effort} --workspace /testbed --yolo "
            "--no-foreign-personal-context --disable-web-tools {prompt}"
        ),
        "install_commands": [], "post_install_commands": [], "post_exec_commands": [],
        "cli_name": "muse_code", "enable_subagents": False,
        "requires_model_proxy": True, "preserve_shared_proxy_traces": True,
        "host_tool_bundle": "muse_standalone", "host_tool_version": MUSE_CODE_CLI_VERSION,
    },
    # muse_code_native with the tool surface cut to the shell/file core (bash, bash_input,
    # read_file, write_file, edit_file), like Claude Code's Bash/Edit/Read/Write and Codex's
    # exec_command/write_stdin/apply_patch/view_image. Same launch command and settings otherwise.
    "muse_code_core": {
        "generator_class": "leanlean.generators.muse_code.MuseCodeCoreAgent",
        "launch_command": (
            "XDG_CONFIG_HOME=/tmp/leanlean-muse/config "
            "XDG_DATA_HOME=/tmp/leanlean-muse/data "
            "XDG_CACHE_HOME=/tmp/leanlean-muse/cache "
            "TBH_STREAM_IDLE_TIMEOUT_SECS=1200 TBH_STREAM_FIRST_EVENT_TIMEOUT_SECS=1200 "
            "MUSE_NO_AUTO_UPDATE=1 MUSE_EXPERIMENTAL_PLUGINS=off "
            "META_API_KEY={api_key} /usr/local/bin/muse exec "
            "--json --model {model} --reasoning-effort {reasoning_effort} --workspace /testbed --yolo "
            "--no-foreign-personal-context --disable-web-tools {prompt}"
        ),
        "install_commands": [], "post_install_commands": [], "post_exec_commands": [],
        "cli_name": "muse_code", "enable_subagents": False,
        "requires_model_proxy": True, "preserve_shared_proxy_traces": True,
        "host_tool_bundle": "muse_standalone", "host_tool_version": MUSE_CODE_CLI_VERSION,
    },
    "qwen_code": add_generator_class(QWEN_GENERATOR_CONFIG, "cli_agent"),
    "codex": add_generator_class(CODEX_GENERATOR_CONFIG, "cli_agent"),
    "codex_sub": add_generator_class(CODEX_SUB_GENERATOR_CONFIG, "cli_agent"),
    "codex_app_server_sub": add_generator_class(
        CODEX_APP_SERVER_SUB_GENERATOR_CONFIG,
        "leanlean.generators.codex_app_server_agent.CodexAppServerAgent",
    ),
    "claude_code": add_generator_class(CLAUDE_CODE_GENERATOR_CONFIG, "cli_agent"),
    "claude_code_sub": add_generator_class(CLAUDE_CODE_SUB_GENERATOR_CONFIG, "cli_agent"),
    "claude_code_glm": add_generator_class(CLAUDE_CODE_GLM_GENERATOR_CONFIG, "cli_agent"),
    "mistral_vibe_lean": add_generator_class(
        MISTRAL_VIBE_LEAN_GENERATOR_CONFIG,
        "cli_agent",
    ),
    "gemini_cli": add_generator_class(GEMINI_CLI_GENERATOR_CONFIG, "cli_agent"),
    "antigravity_cli": add_generator_class(
        ANTIGRAVITY_CLI_GENERATOR_CONFIG, "cli_agent"
    ),
}


# --- Augmentation matrix: lean-lsp-mcp (`mcp`) x lean4-skills (`skills`) -------
# Each base agent gets three new named variants (the unaugmented `none` cell is
# the base entry above). The builder is provider-agnostic (keyed on cli_name);
# enabling Qwen later is just uncommenting its line below — agent_augmentation
# already implements the qwen_code branch.
_AUGMENT_BASES = {
    "claude_code": CLAUDE_CODE_GENERATOR_CONFIG,
    "claude_code_sub": CLAUDE_CODE_SUB_GENERATOR_CONFIG,
    "codex": CODEX_GENERATOR_CONFIG,
    # "qwen_code": QWEN_GENERATOR_CONFIG,  # enable later — helper already supports it
}
for _name, _base in _AUGMENT_BASES.items():
    ALL_GENERATOR_CONFIGS[f"{_name}_mcp"] = add_generator_class(
        make_variant(_base, mcp=True, skills=False), "cli_agent"
    )
    ALL_GENERATOR_CONFIGS[f"{_name}_skills"] = add_generator_class(
        make_variant(_base, mcp=False, skills=True), "cli_agent"
    )
    ALL_GENERATOR_CONFIGS[f"{_name}_mcp_skills"] = add_generator_class(
        make_variant(_base, mcp=True, skills=True), "cli_agent"
    )

# The native Codex subscription path currently needs only the MCP cell. Keep it
# out of the full matrix until subscription-backed skills runs are requested.
ALL_GENERATOR_CONFIGS["codex_sub_mcp"] = add_generator_class(
    make_variant(CODEX_SUB_GENERATOR_CONFIG, mcp=True, skills=False), "cli_agent"
)


