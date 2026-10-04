# Forked from: https://github.com/SWE-agent/mini-swe-generator

import dataclasses
from leanlean.utils import json_utils as json
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, Optional
import textwrap

from leanlean import Generator
from leanlean.utils.log import logger

def _get_class_name_with_module(obj: Any) -> str:
    """Get the full class name with module path."""
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

def _asdict(obj: Any) -> dict:
    """Convert config objects to dicts."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)  # type: ignore[arg-type]
    return obj  # let's try our luck

def _remove_api_key(d: dict, field: str = "api_key") -> dict:
    """Recursively remove any keys that look like API keys."""
    if isinstance(d, dict):
        return {k: _remove_api_key(v, field) for k, v in d.items() if field not in k.lower()}
    elif isinstance(d, list):
        return [_remove_api_key(item, field) for item in d]
    else:
        return d

def save_traj(
    generator: Generator | None,
    path: Path,
    *,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    messages: list[dict[str, Any]] | None = None,
    responses: list[dict[str, Any]] | None = None,
    trajectory_format: str | None = None,
    **kwargs,
):
    """Save the trajectory of the generator to a file.

    Args:
        generator: The generator to save the trajectory of.
        path: The path to save the trajectory to.
        print_path: Whether to print confirmation of path to the terminal.
        exit_status: The exit status of the generator.
        result: The result/submission of the generator.
        extra_info: Extra information to save (will be merged into the info dict).
        messages: Optional explicit messages to persist instead of generator messages.
        responses: Optional explicit responses to persist instead of generator responses.
        trajectory_format: Optional explicit trajectory format name.
        **kwargs: Additional information to save (will be merged into top level)

    """
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
            },
        },
        "messages": [],
        "trajectory_format": "mini-swe-generator-1",
        "responses": [],
    } | kwargs
    if generator is not None:
        data["info"]["model_stats"]["instance_cost"] = generator.model.cost
        data["info"]["model_stats"]["api_calls"] = generator.model.n_calls
        data["messages"] = generator.messages
        data["info"]["config"] = {
            "generator": _asdict(generator.config),
            "model": _asdict(generator.model.config),
            "environment": _asdict(generator.env.config),
            "generator_type": _get_class_name_with_module(generator),
            "model_type": _get_class_name_with_module(generator.model),
            "environment_type": _get_class_name_with_module(generator.env),
        }
        data["responses"] = generator.model.responses
    if messages is not None:
        data["messages"] = messages
    if responses is not None:
        data["responses"] = responses
    if trajectory_format is not None:
        data["trajectory_format"] = trajectory_format
    if extra_info:
        data["info"].update(extra_info)

    #data = _remove_api_key(data)

    logger.debug(f"Saving {path} with data: {data}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def parse_traces_to_md(
    messages: list[dict[str, Any]],
    include_system: bool = True,
    markdown: bool = True,
) -> str:
    """
    Convert a list of OpenAI-style messages into a compact, LLM-friendly string.
    - Keeps only the content shown in the HTML viewer:
        * system/user/assistant text (including 'content' lists of {'type':'text', ...})
        * assistant tool_calls: function name + pretty-printed arguments
        * matching tool results by id (tool.tool_call_id)
        * unpaired tool messages are included as standalone tool_result blocks
    - Bundles each tool call together with its tool results.

    Args:
        messages: list of message dicts (system/user/assistant/tool)
        include_system: whether to include system messages at the top
        markdown: if True, produce Markdown with headings and code fences

    Returns:
        A single string ready to send to an LLM.
    """

    def maybe_parse_json(s: str) -> Tuple[str, Optional[Any]]:
        try:
            obj = json.loads(s)
            return json.dumps(obj, indent=2, ensure_ascii=False), obj
        except Exception:
            return s, None

    def content_to_text(content: Union[str, List[Dict[str, Any]], Dict[str, Any], None]) -> str:
        """Mirror the viewer logic: keep text, show image URLs, pretty-print JSON-ish strings."""
        if content is None:
            return ""
        if isinstance(content, str):
            pretty, _ = maybe_parse_json(content)
            return pretty
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type")
                    if t == "text":
                        parts.append(str(item.get("text", "")))
                    elif t == "image_url":
                        url = item.get("image_url") if isinstance(item.get("image_url"), str) else item.get("image_url", {}).get("url")
                        parts.append(f"[image] {url}")
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if isinstance(content, dict):
            return json.dumps(content, indent=2, ensure_ascii=False)
        return str(content)

    # Index tool results by tool_call_id
    tool_results_by_id: Dict[str, List] = {}
    for m in messages:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid:
                tool_results_by_id.setdefault(tcid, []).append(m)

    lines: List[str] = []

    def add_header(text: str, level: int = 3):
        if markdown:
            lines.append("#" * level + " " + text)
        else:
            lines.append(text.upper())

    def add_block(text: str, code_lang: str = ""):
        text = text if text is not None else ""
        if markdown:
            fence = "```" + (code_lang or "")
            lines.append(fence)
            lines.append(text)
            lines.append("```")
        else:
            lines.append(textwrap.indent(text, prefix="    "))

    for m in messages:
        role = m.get("role", "")
        # Skip system if not requested
        if role == "system" and not include_system:
            continue

        # For tool messages that are paired, skip here; they’ll appear under the assistant tool_call bundle.
        if role == "tool" and m.get("tool_call_id") in tool_results_by_id:
            # We will render them within the assistant section for the matching call.
            continue

        if role == "system":
            add_header("SYSTEM", 3)
            add_block(content_to_text(m.get("content")), code_lang="text")

        elif role == "user":
            add_header("USER", 3)
            user_text = content_to_text(m.get("content"))
            lines.append(user_text if user_text else "(no text)")

        elif role == "assistant":
            add_header("ASSISTANT", 3)
            asst_text = content_to_text(m.get("content"))
            lines.append(asst_text if asst_text else "(no text)")

            # Inline tool calls + results
            for call in m.get("tool_calls", []) or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                fname = fn.get("name", "unknown")
                args_raw = fn.get("arguments", "")
                args_pretty, _ = maybe_parse_json(args_raw if isinstance(args_raw, str) else json.dumps(args_raw))

                add_header(f"TOOL CALL • {fname}", 4)
                # Minimal metadata, as shown in the app
                meta = {
                    "type": call.get("type", "function"),
                    "function": fname,
                }
                add_block(json.dumps(meta, indent=2, ensure_ascii=False), code_lang="json")

                lines.append("**arguments**:")
                add_block(args_pretty, code_lang="json" if args_pretty.strip().startswith("{") else "text")

                # Results
                results = tool_results_by_id.get(call.get("id", ""), [])
                if results:
                    for i, tr in enumerate(results, 1):
                        lines.append(f"**tool_result {i}**:")
                        tr_text = content_to_text(tr.get("content"))
                        tr_pretty, _ = maybe_parse_json(tr_text)
                        add_block(tr_pretty, code_lang="json" if tr_pretty.strip().startswith("{") else "text")
                else:
                    lines.append("**tool_result**: (none)")

        elif role == "tool":
            # Unpaired tool message: show as a standalone tool_result
            add_header("TOOL RESULT", 4)
            tr_text = content_to_text(m.get("content"))
            tr_pretty, _ = maybe_parse_json(tr_text)
            add_block(tr_pretty, code_lang="json" if tr_pretty.strip().startswith("{") else "text")

        else:
            # Unknown roles: include minimally for completeness
            add_header(role or "UNKNOWN", 3)
            add_block(json.dumps(m, indent=2, ensure_ascii=False), code_lang="json")

        # Spacer between messages
        lines.append("")

    return "\n".join(lines).strip()


def _compute_run_directory(
    *,
    output_dir: Path,
    dataset_name: str,
    plan_type: str,
    generator: str,
    exec_model: str,
    run_id: str | int,
    planner_config: dict[str, Any],
    kind: str,
    train_plan: bool = False,
    continuous_training: bool = False,
    task_variant: str | None = None,
) -> Path:
    dataset_segment = dataset_name.replace("/", "_")

    if train_plan:
        assert kind == "leanlean", "Invalid kind for training plan"
        kind = "leanlean_training"

    # task_variant replaces plan_type as the varying dimension when set
    plan_segment = task_variant if task_variant is not None else plan_type
    run_dir = output_dir / kind / dataset_segment / plan_segment

    if train_plan:
        return run_dir / generator / exec_model

    return run_dir / generator / exec_model / f"run_{run_id}"
