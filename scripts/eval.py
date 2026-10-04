#!/usr/bin/env python3
"""Launch one artifact-first LeanLean evaluation."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from configs.generator_constants import ALL_GENERATOR_CONFIGS  # noqa: E402
from leanlean.generators.cli_agent import (  # noqa: E402
    resolve_claude_standalone_tool,
    resolve_codex_standalone_tools,
    resolve_mistral_vibe_standalone_bundle,
    resolve_antigravity_standalone_tool,
)
from leanlean.pipeline.evaluation import (  # noqa: E402
    PreparedEvaluation,
    ResolvedEvaluation,
    load_run_manifest,
    resolve_named_config,
    validate_named_run_identity,
    update_run_status,
    write_run_definition,
)
from leanlean.subscription_queue import (  # noqa: E402
    CapacityDecision,
    QueuePolicy,
    capacity_decision,
    timestamp,
    wait_for_capacity,
)
from leanlean.subscription_auth import (  # noqa: E402
    activate_subscription_credential,
    subscription_auth_record,
)
from leanlean.subscription_status import QUOTA_EXIT_CODE  # noqa: E402
from scripts import run_codex_sub_yaml as engine  # noqa: E402


console = Console()


def _preflight_harness(engine_manifest: dict) -> None:
    harness = str(engine_manifest["generator"])
    model = str(engine_manifest["model"])
    if model not in engine.runner.ALL_MODEL_CONFIGS:
        raise RuntimeError(f"unknown model {model!r}")
    config = ALL_GENERATOR_CONFIGS[harness]
    if config.get("subscription_transport") == "chatgpt":
        subscription_auth_record()
    if harness == "claude_code_sub" and not os.environ.get(
        "CLAUDE_CODE_OAUTH_TOKEN"
    ):
        raise RuntimeError(
            "claude_code_sub requires CLAUDE_CODE_OAUTH_TOKEN; run "
            "`claude setup-token` and place it in secret.sh"
        )
    if harness == "mistral_vibe_lean" and not os.environ.get(
        "MISTRAL_API_KEY"
    ):
        raise RuntimeError(
            "mistral_vibe_lean requires MISTRAL_API_KEY in secret.sh"
        )
    if harness == "claude_code_glm":
        required_key = (
            "PROXY_API_KEY"
            if model == "glm-5.3-flash"
            else "Z_AI_API_KEY"
        )
        if not os.environ.get(required_key):
            raise RuntimeError(
                f"{harness} requires {required_key} in secret.sh"
            )
    if harness == "antigravity_cli":
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not gemini_key:
            raise RuntimeError(
                "antigravity_cli requires GEMINI_API_KEY or GOOGLE_API_KEY "
                "in secret.sh"
            )
        os.environ.setdefault("GEMINI_API_KEY", gemini_key)
    if model.startswith("muse-spark-"):
        if not os.environ.get("META_API_KEY"):
            raise RuntimeError(f"{model} requires META_API_KEY in secret.sh")
    version = str(engine_manifest.get("harness_version") or config.get("host_tool_version") or "")
    bundle = config.get("host_tool_bundle")
    if not version:
        raise RuntimeError(f"{harness} has no pinned host tool version")
    if bundle == "codex_standalone":
        resolve_codex_standalone_tools(version)
    elif bundle == "claude_standalone":
        resolve_claude_standalone_tool(version)
    elif bundle == "mistral_vibe_standalone":
        resolve_mistral_vibe_standalone_bundle(version)
    elif bundle == "antigravity_standalone":
        resolve_antigravity_standalone_tool(version)
    elif bundle == "muse_standalone":
        from leanlean.generators.muse_code import resolve_muse_standalone_tool
        resolve_muse_standalone_tool(version)
    else:
        raise RuntimeError(
            f"{harness} has no approved offline CLI bundle"
        )


def _validate_engine_path(path: Path) -> dict:
    return engine.load_manifest(path)


def _activate_manifest_subscription_credential(manifest: dict) -> str | None:
    """Select the manifest's non-secret credential name after secret.sh loads."""
    access = manifest.get("access")
    if not isinstance(access, dict):
        return None
    key_env = access.get("credential_env")
    if key_env is None:
        return None
    activate_subscription_credential(key_env)
    return key_env


def _validate_resolved(resolved: ResolvedEvaluation) -> dict:
    _activate_manifest_subscription_credential(resolved.manifest)
    content = yaml.safe_dump(
        resolved.engine_manifest, sort_keys=False, width=100
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir="/tmp", delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        spec = _validate_engine_path(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _preflight_harness(spec)
    return spec


def _launch_tmux(manifest: Path, session: str) -> None:
    attach = f"tmux attach -t {shlex.quote(session)}"
    existing = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    if existing.returncode == 0:
        console.print(f"[yellow]already running[/yellow]: {session}")
        console.print(attach)
        return
    launch = [
        "bash",
        str(REPO_ROOT / "eval.sh"),
        "--run-manifest",
        str(manifest.resolve()),
        "--inside-tmux",
    ]
    claude_token_env = os.environ.get("LEANLEAN_CLAUDE_TOKEN_ENV")
    if claude_token_env:
        # tmux servers can predate the caller and therefore need not inherit a
        # custom selector. Carry only its non-secret variable name explicitly;
        # eval.sh resolves the value after sourcing secret.sh in the new pane.
        launch = [
            "env",
            f"LEANLEAN_CLAUDE_TOKEN_ENV={claude_token_env}",
            *launch,
        ]
    command = shlex.join(launch)
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(REPO_ROOT),
            command,
        ],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "set-option",
            "-t",
            session,
            "remain-on-exit",
            "off",
        ],
        check=False,
    )
    console.print(f"[green]launched[/green] in tmux session {session}")
    console.print(attach)


def _run_artifact(path: Path) -> dict:
    try:
        artifact = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return artifact if isinstance(artifact, dict) else {}


def _artifact_status(path: Path) -> str:
    return str(_run_artifact(path).get("status", ""))


def _attempt_count(path: Path) -> int:
    lifecycle = _run_artifact(path).get("lifecycle")
    if not isinstance(lifecycle, dict):
        return 0
    try:
        return int(lifecycle.get("attempt_count", 0))
    except (TypeError, ValueError):
        return 0


def _completed_named_run(resolved: ResolvedEvaluation) -> bool:
    """Return a verified no-op for an already completed config pair."""

    if validate_named_run_identity(resolved, repo_root=REPO_ROOT) != "complete":
        return False
    console.print(
        f"[cyan]run exists and is complete[/cyan]: {resolved.run_artifact_path}"
    )
    return True


def _queue_policy(prepared: PreparedEvaluation) -> QueuePolicy:
    row = prepared.manifest.get("subscription")
    if not isinstance(row, dict):
        return QueuePolicy(False, 300, 18000, "fixed_cooldown")
    policy = QueuePolicy(
        enabled=bool(row["quota_queue"]),
        retry_backoff_seconds=int(row["retry_backoff_seconds"]),
        fallback_cooldown_seconds=int(row["fallback_cooldown_seconds"]),
        capacity_probe=str(row["capacity_probe"]),
        max_utilization_percent=float(row["max_utilization_percent"]),
    )
    if os.environ.get("LEANLEAN_QUOTA_HANDOFF") == "1":
        # A multi-account router owns the retry decision. Returning the quota
        # exit to it prevents this child from sleeping on one exhausted account.
        return QueuePolicy(
            enabled=False,
            retry_backoff_seconds=policy.retry_backoff_seconds,
            fallback_cooldown_seconds=policy.fallback_cooldown_seconds,
            capacity_probe=policy.capacity_probe,
            max_utilization_percent=policy.max_utilization_percent,
        )
    return policy


def _wait_metadata(
    policy: QueuePolicy, decision: CapacityDecision
) -> dict:
    assert decision.wait_until is not None
    return {
        "capacity_probe": policy.capacity_probe,
        "reason": decision.reason,
        "next_check_at": timestamp(decision.wait_until),
        "windows": [
            {
                "name": window.name,
                "utilization": window.utilization,
                "resets_at": window.resets_at,
            }
            for window in decision.usage.windows
        ],
    }


def _wait_on_decision(
    prepared: PreparedEvaluation,
    policy: QueuePolicy,
    decision: CapacityDecision,
    *,
    attempt: int,
    recorded_attempt: int,
    exit_code: int | None = None,
) -> None:
    if not decision.should_wait:
        return
    update_run_status(
        prepared.run_artifact_path,
        "waiting_for_quota",
        exit_code=exit_code,
        attempt=recorded_attempt,
        quota_wait=_wait_metadata(policy, decision),
    )
    reason = decision.reason.replace("_", " ")
    console.print(
        f"[yellow]subscription gate: {reason}; next usage check at "
        f"{timestamp(decision.wait_until)}[/yellow]"
    )
    wait_for_capacity(
        decision,
        run_id=str(prepared.manifest["run_id"]),
        harness=str(prepared.manifest["harness"]),
        attempt=attempt,
    )


def _gate_before_attempt(
    prepared: PreparedEvaluation,
    policy: QueuePolicy,
    *,
    attempt: int,
    recorded_attempt: int,
) -> None:
    if not policy.enabled:
        return
    while True:
        decision = capacity_decision(policy)
        if not decision.should_wait:
            return
        _wait_on_decision(
            prepared,
            policy,
            decision,
            attempt=attempt,
            recorded_attempt=recorded_attempt,
        )


def _execute(prepared: PreparedEvaluation) -> int:
    """Serialize callers of the same pinned run, including pipeline joins."""
    import fcntl
    with prepared.run_artifact_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _execute_locked(prepared)


def _execute_locked(prepared: PreparedEvaluation) -> int:
    if _artifact_status(prepared.run_artifact_path) == "complete":
        console.print(
            f"[cyan]already complete[/cyan]: "
            f"{prepared.run_artifact_path}"
        )
        return 0
    _activate_manifest_subscription_credential(dict(prepared.manifest))
    spec = _validate_engine_path(prepared.engine_manifest_path)
    _preflight_harness(spec)
    policy = _queue_policy(prepared)
    attempt = _attempt_count(prepared.run_artifact_path)

    while True:
        _gate_before_attempt(
            prepared,
            policy,
            attempt=attempt + 1,
            recorded_attempt=attempt,
        )
        attempt += 1
        update_run_status(
            prepared.run_artifact_path,
            "running",
            attempt=attempt,
        )
        try:
            subprocess.run(
                ["bash", "scripts/build_model_relay_image.sh"],
                cwd=REPO_ROOT,
                check=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_codex_sub_yaml.py",
                    str(prepared.engine_manifest_path),
                ],
                cwd=REPO_ROOT,
            )
        except Exception as error:
            update_run_status(
                prepared.run_artifact_path,
                "failed",
                exit_code=1,
                error=str(error),
                attempt=attempt,
            )
            raise

        if result.returncode == 0:
            update_run_status(
                prepared.run_artifact_path,
                "complete",
                exit_code=0,
                attempt=attempt,
            )
            console.print(
                f"[green]complete[/]: {prepared.run_artifact_path}"
            )
            return 0

        if result.returncode == QUOTA_EXIT_CODE and policy.enabled:
            decision = capacity_decision(
                policy,
                after_quota_failure=True,
            )
            _wait_on_decision(
                prepared,
                policy,
                decision,
                attempt=attempt,
                recorded_attempt=attempt,
                exit_code=result.returncode,
            )
            continue

        status = (
            "quota_exhausted"
            if result.returncode == QUOTA_EXIT_CODE
            else "failed"
        )
        update_run_status(
            prepared.run_artifact_path,
            status,
            exit_code=result.returncode,
            attempt=attempt,
        )
        console.print(
            f"[red]{status}[/]: {prepared.run_artifact_path}"
        )
        return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        metavar="DATASET",
        help="extensionless dataset name under configs/dataset/",
    )
    parser.add_argument(
        "--model", metavar="PROVIDER/MODEL",
        help="extensionless <provider>/<model> name under configs/models/"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--run-manifest", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inside-tmux", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.run_manifest is not None:
        if args.config is not None or args.model is not None or not args.inside_tmux:
            parser.error(
                "--run-manifest is reserved for the tmux launcher"
            )
        if not os.environ.get("TMUX"):
            parser.error("--inside-tmux requires a tmux session")
        prepared = load_run_manifest(
            args.run_manifest, repo_root=REPO_ROOT
        )
        return _execute(prepared)
    if args.config is None:
        parser.error(
            "DATASET is required (for example leanlean-20260905 "
            "--model openai/gpt-5.6-luna-xhigh)"
        )

    if args.model is None:
        parser.error("--model is required (for example openai/gpt-5.6-luna-xhigh)")
    try:
        resolved = resolve_named_config(
            args.config, args.model, repo_root=REPO_ROOT
        )
    except ValueError as error:
        parser.error(str(error))
    if not args.validate_only and _completed_named_run(resolved):
        return 0
    spec = _validate_resolved(resolved)
    console.print(
        f"dataset: {resolved.manifest['dataset']['id']} "
        f"{resolved.manifest['dataset']['version']} / "
        f"{resolved.manifest['dataset']['variant']} "
        f"({len(resolved.manifest['repositories'])} repositories)"
    )
    console.print(
        f"agent: {spec['model']} / {spec['reasoning_effort']} / "
        f"{spec['generator']}"
    )
    console.print(
        f"parallelism: {spec['workers']} containers; "
        f"{spec['container_cpus']} CPUs and "
        f"{spec['container_memory']} each; ceiling "
        f"{resolved.manifest['container']['max_total_memory']}"
    )
    monitoring = (
        f"every {spec['playback']['every_n_edits']} edit(s), "
        f"reconcile every "
        f"{spec['playback']['reconciliation_interval_seconds']}s"
        if spec["playback"]["enabled"]
        else "disabled"
    )
    console.print(f"monitoring: {monitoring}")
    subscription = resolved.manifest["subscription"]
    access = resolved.manifest.get("access")
    if isinstance(access, dict):
        if access["mode"] == "api":
            console.print("access: API; subscription monitoring disabled")
        else:
            limits = access["limits"]
            names = [
                label
                for key, label in (
                    ("five_hour", "5h"),
                    ("weekly", "weekly"),
                )
                if limits[key]
            ]
            strategy = access["monitoring"]["strategy"]
            console.print(
                f"access: subscription; limits {', '.join(names)}; "
                f"monitoring {strategy}"
            )
    else:
        queue = (
            "enabled; check once before launch; "
            f"{subscription['capacity_probe']}; "
            f"backoff {subscription['retry_backoff_seconds']}s; "
            f"fallback {subscription['fallback_cooldown_seconds']}s"
            if subscription["quota_queue"]
            else "disabled"
        )
        console.print(f"subscription queue: {queue}")

    if args.validate_only:
        console.print(f"[green]validated[/green] {resolved.config_path}")
        return 0

    # The caller owns the terminal/session. In particular, running this command
    # inside tmux keeps validation and generation in that same visible pane.
    write_run_definition(resolved)
    prepared = load_run_manifest(
        resolved.manifest_path, repo_root=REPO_ROOT
    )
    return _execute(prepared)


if __name__ == "__main__":
    raise SystemExit(main())
