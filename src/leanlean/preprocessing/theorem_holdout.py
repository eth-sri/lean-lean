"""Self-contained theorem holdout source artifacts."""

from __future__ import annotations


def theorem_statement_prefix(command: str) -> str:
    """Return a theorem command through its top-level type, before ``:=``."""

    depth = 0
    block_comment = 0
    in_string = False
    escaped = False
    pending_local_values = 0
    index = 0
    while index + 1 < len(command):
        pair = command[index:index + 2]
        char = command[index]
        if block_comment:
            if pair == "/-":
                block_comment += 1
                index += 2
                continue
            if pair == "-/":
                block_comment -= 1
                index += 2
                continue
            index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if pair == "/-":
            block_comment = 1
            index += 2
            continue
        if pair == "--":
            newline = command.find("\n", index + 2)
            index = len(command) if newline < 0 else newline + 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if depth == 0 and (char.isalpha() or char == "_"):
            end = index + 1
            while end < len(command) and (command[end].isalnum() or command[end] in "_'."):
                end += 1
            if command[index:end] in {"let", "letI", "have"}:
                pending_local_values += 1
            index = end
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif pair == ":=" and depth == 0:
            if pending_local_values:
                pending_local_values -= 1
                index += 2
                continue
            return command[:index].rstrip()
        index += 1
    raise ValueError("held-out source command has no top-level := proof boundary")


def theorem_command_with_sorry(source_command: str) -> str:
    """Serialize the exact theorem statement with only its proof erased."""

    return theorem_statement_prefix(source_command) + " := by\n  sorry\n"


__all__ = ["theorem_command_with_sorry", "theorem_statement_prefix"]
