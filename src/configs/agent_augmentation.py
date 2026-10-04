"""Provider-agnostic augmentation of CLI-agent generator configs.

Adds two orthogonal capabilities to a base generator config:

  * ``mcp``    — wire up the `lean-lsp-mcp` MCP server (live Lean goal state,
                 hover, loogle/mathlib search) as a local stdio subprocess the
                 CLI agent spawns. The server itself (`uvx lean-lsp-mcp`) is
                 identical across agents; only *where the MCP config file lives*
                 differs per CLI, so that is the only provider-specific branch.
  * ``skills`` — install the host-agnostic `lean4-skills` workflow pack. For now
                 we only *install* it (shallow clone into $HOME, outside the
                 testbed so it never lands in the task diff). Invocation is
                 deferred.

The commands are appended to ``post_install_commands`` (run before the agent
launches) and mirrored in ``post_exec_commands`` for cleanup — the harness
reuses one container across runs, so per-run artifacts must be torn down to
avoid bleed (same reason the Qwen base does ``rm -rf .qwen``).

Everything is keyed on the base config's ``cli_name`` so enabling a new agent is
a one-line change in :mod:`configs.generator_constants`.
"""

from __future__ import annotations

import copy

LEAN_SKILLS_REPO = "https://github.com/cameronfreer/lean4-skills"
SKILLS_DIR = "$HOME/lean4-skills"  # outside /testbed -> never shows in the diff
MCP_CONFIG_PATH = "$HOME/.lean-mcp.json"  # Claude Code --mcp-config target
CODEX_SUB_HOME = "/tmp/leanlean-codex-home"

# --- lean4-skills layout (host-agnostic pack) ------------------------------
# The pack is host-agnostic: every host uses the SAME core skill under
# plugins/lean4/skills/lean4/, plus deterministic `lean4-skills-*` wrappers in
# plugins/lean4/bin/. Per INSTALLATION.md, only the *invocation surface* differs
# per host; the substrate is identical:
#   * env bootstrap (LEAN4_* + PATH) so the model's shell resolves the wrappers,
#   * the SKILL.md dropped into the host's skill-discovery dir.
# We keep everything under $HOME (never /testbed) so it never lands in the diff
# or the compression word count.
_SKILLS_PLUGIN_ROOT = f"{SKILLS_DIR}/plugins/lean4"
_SKILLS_SKILL_DIR = f"{_SKILLS_PLUGIN_ROOT}/skills/lean4"  # holds SKILL.md
# Env exports prepended to the launch_command (env.execute is stateless per
# call, so — like MCP's --mcp-config flag — this must live on the launch line,
# not in post_install). The model's Bash/shell tool inherits the agent
# process's environment, so `lean4-skills-*` wrappers resolve as bare commands.
_SKILLS_ENV_EXPORTS = (
    f'export LEAN4_PLUGIN_ROOT="{_SKILLS_PLUGIN_ROOT}"; '
    f'export LEAN4_SCRIPTS="$LEAN4_PLUGIN_ROOT/lib/scripts"; '
    f'export LEAN4_REFS="$LEAN4_PLUGIN_ROOT/skills/lean4/references"; '
    f'export PATH="$LEAN4_PLUGIN_ROOT/bin:$PATH"; '
)
# Per-host skill-discovery dirs. Both hosts rely purely on native discovery of
# a SKILL.md dropped here (symmetric, no per-host AGENTS.md pointer): Claude
# auto-discovers ~/.claude/skills/*, Codex auto-discovers ~/.agents/skills/*.
# A SKILL.md here activates headlessly, no interactive `/plugin` command needed.
_CLAUDE_SKILLS_DIR = "$HOME/.claude/skills"
_CODEX_SKILLS_DIR = "$HOME/.agents/skills"

# The MCP server is always `uvx lean-lsp-mcp` over stdio. We reference uvx by its
# absolute install path so the agent's MCP spawner finds it regardless of PATH.
UVX = "$HOME/.local/bin/uvx"

# Legacy online uv/uvx bootstrap. Safe model-driven runners reject variants
# containing this command until uvx and lean-lsp-mcp have an approved offline
# bundle; task containers never receive package-download egress.
UV_INSTALL = f'[ -x "{UVX}" ] || curl -LsSf https://astral.sh/uv/install.sh | sh'

# Claude Code cli_names that share the same MCP wiring (--mcp-config flag).
_CLAUDE_CLI_NAMES = {"claude_code", "claude_code_sub", "claude_code_glm"}


def _mcp_config_write_cmds(cli_name: str) -> list[str]:
    """Commands that write the per-CLI MCP config file. $HOME expands in-shell."""
    if cli_name in _CLAUDE_CLI_NAMES:
        return [
            f'cat > "{MCP_CONFIG_PATH}" <<EOF\n'
            f'{{"mcpServers":{{"lean-lsp":{{"command":"{UVX}","args":["lean-lsp-mcp"]}}}}}}\n'
            f"EOF",
        ]
    if cli_name in {"codex", "codex_sub"}:
        config_dir = (
            '"$HOME/.codex"' if cli_name == "codex" else CODEX_SUB_HOME
        )
        return [
            f"mkdir -p {config_dir}",
            f"cat > {config_dir}/config.toml <<EOF\n"
            "[mcp_servers.lean-lsp]\n"
            f'command = "{UVX}"\n'
            'args = ["lean-lsp-mcp"]\n'
            "EOF",
        ]
    if cli_name == "qwen_code":
        # Qwen reads .qwen/settings.json. The base config already wrote that file
        # in its own post_install; we rewrite it here with the same fields *plus*
        # an mcpServers block (no jq/python guaranteed in-image, so a full
        # rewrite is simpler than an in-place merge). Base `rm -rf .qwen` cleans
        # it up. Not registered yet — provided so enabling Qwen is one line.
        return [
            "mkdir -p .qwen",
            "cat > .qwen/settings.json <<EOF\n"
            "{\n"
            '  "sessionTokenLimit": 262144,\n'
            '  "contextFileName": "AGENTS.md",\n'
            '  "chatCompression": {\n'
            '    "contextPercentageThreshold": 0.6\n'
            "  },\n"
            '  "summarizeToolOutput": {\n'
            '    "run_shell_command": {\n'
            '      "tokenBudget": 2000\n'
            "    }\n"
            "  },\n"
            '  "mcpServers": {\n'
            '    "lean-lsp": {\n'
            f'      "command": "{UVX}",\n'
            '      "args": ["lean-lsp-mcp"]\n'
            "    }\n"
            "  }\n"
            "}\n"
            "EOF",
            "cat .qwen/settings.json",
        ]
    raise ValueError(f"MCP wiring not implemented for cli_name={cli_name!r}")


def mcp_install_cmds(cli_name: str) -> list[str]:
    """Install commands for the `lean-lsp-mcp` server (uv + per-CLI config)."""
    return [UV_INSTALL, *_mcp_config_write_cmds(cli_name)]


def mcp_cleanup_cmds(cli_name: str) -> list[str]:
    """Remove the per-CLI MCP config file written by :func:`mcp_install_cmds`."""
    if cli_name in _CLAUDE_CLI_NAMES:
        return [f'rm -f "{MCP_CONFIG_PATH}"']
    if cli_name == "codex":
        return ['rm -f "$HOME/.codex/config.toml"']
    if cli_name == "codex_sub":
        # cli_agent removes the entire isolated CODEX_HOME before calling the
        # variant cleanup hooks.
        return []
    if cli_name == "qwen_code":
        return []  # base config already does `rm -rf .qwen`
    raise ValueError(f"MCP cleanup not implemented for cli_name={cli_name!r}")


def skills_install_cmds(cli_name: str) -> list[str]:
    """Install the host-agnostic lean4-skills pack for `cli_name`.

    Common to all hosts: shallow-clone the pack into $HOME. Then wire the
    host-specific invocation surface (env bootstrap goes on the launch line via
    :func:`_augment_launch_for_skills`, not here):

      * Claude Code — drop the core SKILL.md into `~/.claude/skills/lean4`.
      * Codex — drop the same SKILL.md into `~/.agents/skills/lean4`.

    Both hosts then auto-discover and activate the skill natively (symmetric —
    no host-specific AGENTS.md pointer). All artifacts live under $HOME (never
    /testbed) so nothing lands in the diff.
    """
    clone = (
        f'rm -rf "{SKILLS_DIR}" && '
        f'git clone --depth 1 {LEAN_SKILLS_REPO} "{SKILLS_DIR}"'
    )
    skills_dir = {
        **{n: _CLAUDE_SKILLS_DIR for n in _CLAUDE_CLI_NAMES},
        "codex": _CODEX_SKILLS_DIR,
    }.get(cli_name)
    if skills_dir is None:
        raise ValueError(f"skills wiring not implemented for cli_name={cli_name!r}")
    return [
        clone,
        f'mkdir -p "{skills_dir}" && ln -sfn "{_SKILLS_SKILL_DIR}" "{skills_dir}/lean4"',
    ]


def skills_cleanup_cmds(cli_name: str) -> list[str]:
    """Tear down everything :func:`skills_install_cmds` created (container reuse)."""
    skills_dir = {
        **{n: _CLAUDE_SKILLS_DIR for n in _CLAUDE_CLI_NAMES},
        "codex": _CODEX_SKILLS_DIR,
    }.get(cli_name)
    if skills_dir is None:
        raise ValueError(f"skills cleanup not implemented for cli_name={cli_name!r}")
    return [f'rm -rf "{skills_dir}/lean4" "{SKILLS_DIR}"']


def _augment_launch_for_skills(cfg: dict) -> str:
    """Prepend the LEAN4_* / PATH env bootstrap to the launch command.

    env.execute runs each command in a fresh shell, so exports from
    post_install don't survive to launch — the bootstrap must ride on the
    launch line itself (same reason MCP injects --mcp-config there). The agent
    process inherits these, and its shell/Bash tool inherits them in turn, so
    the model can call `lean4-skills-*` wrappers as bare commands.
    """
    return _SKILLS_ENV_EXPORTS + cfg["launch_command"]


def _augment_launch_for_mcp(cfg: dict) -> str:
    """For Claude Code, point the CLI at our MCP config via --mcp-config.

    Inserts the flags just before the trailing ``-p {prompt}`` so they parse as
    options (works for the api, sub, and glm launch templates, which all end in
    ``-p {prompt}``). ``--strict-mcp-config`` ignores any user-scoped servers.
    Other CLIs auto-load their config file and need no launch change.
    """
    launch = cfg["launch_command"]
    if cfg["cli_name"] == "codex_sub":
        # The subscription runner points CODEX_HOME at a fresh, fixed /tmp
        # directory containing only the streamed auth cache and the MCP config
        # written above. Allow Codex to read that isolated config; no host/user
        # config is reachable from this CODEX_HOME.
        marker = "--ignore-user-config "
        if marker not in launch:
            raise ValueError(
                "cannot enable MCP for codex_sub: launch_command has no "
                f"{marker!r} marker"
            )
        return launch.replace(marker, "", 1)
    if cfg["cli_name"] not in _CLAUDE_CLI_NAMES:
        return launch
    flags = f'--mcp-config "{MCP_CONFIG_PATH}" --strict-mcp-config'
    marker = " -p {prompt}"
    if marker not in launch:
        raise ValueError(
            f"cannot inject --mcp-config: launch_command for {cfg['cli_name']!r} "
            f"has no {marker!r} suffix"
        )
    return launch.replace(marker, f" {flags}{marker}")


def make_variant(base: dict, *, mcp: bool, skills: bool) -> dict:
    """Compose mcp/skills augmentation onto a copy of a base generator config."""
    cfg = copy.deepcopy(base)  # never share mutable lists with the baseline
    if mcp:
        cfg["post_install_commands"] = (
            cfg["post_install_commands"] + mcp_install_cmds(cfg["cli_name"])
        )
        cfg["post_exec_commands"] = (
            mcp_cleanup_cmds(cfg["cli_name"]) + cfg["post_exec_commands"]
        )
        cfg["launch_command"] = _augment_launch_for_mcp(cfg)
    if skills:
        cfg["post_install_commands"] = (
            cfg["post_install_commands"] + skills_install_cmds(cfg["cli_name"])
        )
        cfg["post_exec_commands"] = (
            skills_cleanup_cmds(cfg["cli_name"]) + cfg["post_exec_commands"]
        )
        # Env bootstrap rides on the launch line (composes with the mcp flags
        # already injected above, if any).
        cfg["launch_command"] = _augment_launch_for_skills(cfg)
    return cfg
