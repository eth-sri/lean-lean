from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Iterable


@dataclass
class DiffEntry:
    path: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _safe_split(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def _strip_diff_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _normalize_path(path: str | None) -> str | None:
    if not path:
        return None
    cleaned = path.strip()
    if cleaned == "/dev/null":
        return None
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) > 1:
        cleaned = cleaned[1:-1]
    return _strip_diff_prefix(cleaned)


def _is_python_path(path: str | None) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    return normalized.lower().endswith(".py")


def _has_hidden_component(path: str | None) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    for part in normalized.split("/"):
        if part.startswith(".") and part not in {".", ".."}:
            return True
    return False


_DISALLOWED_DIR_NAMES = {
    ".env",
    ".nox",
    ".tox",
    ".venv",
    "__pypackages__",
    "anaconda",
    "anaconda3",
    "conda",
    "conda-env",
    "condaenv",
    "dist-packages",
    "env",
    "mambaforge",
    "micromamba",
    "miniconda",
    "miniconda3",
    "miniforge",
    "miniforge3",
    "pipenv",
    "poetry",
    "pyenv",
    "site-packages",
    "venv",
    "virtualenv",
    "examples",
    "docs",
    "sites"
}


def _has_disallowed_dir(path: str | None) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    for part in normalized.split("/"):
        if part.lower() in _DISALLOWED_DIR_NAMES:
            return True
    return False


def _is_root_test_file(path: str | None) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return False
    if "/" in normalized:
        return False
    return "test" in normalized.lower()


def _is_allowed_path(path: str | None) -> bool:
    return (
        _is_python_path(path)
        and not _has_hidden_component(path)
        and not _has_disallowed_dir(path)
        and not _is_root_test_file(path)
    )


def _parse_diff_paths(line: str) -> tuple[str | None, str | None]:
    parts = _safe_split(line)
    if len(parts) >= 4:
        return parts[2], parts[3]
    return None, None


def _parse_file_header_path(line: str) -> str | None:
    if not (line.startswith("+++ ") or line.startswith("--- ")):
        return None
    payload = line[4:]
    if "\t" in payload:
        payload = payload.split("\t", 1)[0]
    payload = payload.strip()
    return payload or None


def _select_path(path_a: str | None, path_b: str | None, header_path: str | None = None) -> str | None:
    for candidate in (path_b, path_a, header_path):
        normalized = _normalize_path(candidate)
        if normalized:
            return normalized
    return None


class DiffFile:
    """Load and filter a git diff file for Python paths excluding hidden/env dirs."""

    def __init__(self, diff_path: str | Path, clean_diff: bool = True) -> None:
        self._path = Path(diff_path)
        self._raw_text = self._path.read_text()
        self._preamble_lines: list[str] = []
        self._entries: list[DiffEntry] = []
        self._entries_by_path: dict[str, list[DiffEntry]] = {}
        self._file_order: list[str] = []
        self._clean_diff = clean_diff
        self._parse(self._raw_text)

    @classmethod
    def from_text(cls, diff_text: str, clean_diff: bool = True) -> "DiffFile":
        instance = cls.__new__(cls)
        instance._path = None
        instance._raw_text = diff_text
        instance._preamble_lines = []
        instance._entries = []
        instance._entries_by_path = {}
        instance._file_order = []
        instance._clean_diff = clean_diff
        instance._parse(diff_text)
        return instance

    def _add_entry(self, path: str, lines: list[str]) -> None:
        entry = DiffEntry(path=path, lines=lines)
        self._entries.append(entry)
        if path not in self._entries_by_path:
            self._entries_by_path[path] = []
            self._file_order.append(path)
        self._entries_by_path[path].append(entry)

    def _parse_single_section(self, lines: list[str]) -> None:
        path_a = None
        path_b = None
        for line in lines:
            if line.startswith("--- "):
                path_a = _parse_file_header_path(line)
            elif line.startswith("+++ "):
                path_b = _parse_file_header_path(line)
            if path_a and path_b:
                break

        path = _select_path(path_a, path_b)
        if path and _is_allowed_path(path):
            self._add_entry(path, lines)

    def _parse(self, diff_text: str) -> None:
        if not diff_text:
            return

        lines = diff_text.splitlines()
        if not any(line.startswith("diff --git ") for line in lines):
            self._parse_single_section(lines)
            return

        current_lines: list[str] = []
        current_allowed = False
        current_path: str | None = None
        preamble: list[str] = []
        started = False

        def flush_section() -> None:
            nonlocal current_lines, current_allowed, current_path
            if current_lines and current_allowed and current_path:
                self._add_entry(current_path, current_lines)
            current_lines = []
            current_allowed = False
            current_path = None

        for line in lines:
            if line.startswith("diff --git "):
                if current_lines:
                    flush_section()
                else:
                    if not started:
                        self._preamble_lines = preamble
                started = True
                current_lines = [line]
                path_a, path_b = _parse_diff_paths(line)
                current_path = _select_path(path_a, path_b)
                current_allowed = _is_allowed_path(current_path) or (not self._clean_diff)
                continue

            if not started:
                preamble.append(line)
                continue

            if line.startswith(("--- ", "+++ ")):
                header_path = _parse_file_header_path(line)
                if header_path:
                    if line.startswith("+++ "):
                        header_candidate = _normalize_path(header_path)
                        if header_candidate:
                            current_path = header_candidate
                            current_allowed = _is_allowed_path(current_path) or (not self._clean_diff)
                    if not current_path:
                        current_path = _select_path(None, None, header_path)
                        current_allowed = _is_allowed_path(current_path) or (not self._clean_diff)
                current_lines.append(line)
                continue

            current_lines.append(line)

        if current_lines:
            flush_section()

    def get_file_names(self) -> list[str]:
        return list(self._file_order)

    def get_file_diff(self, file_name: str) -> str:
        normalized = _normalize_path(file_name) or file_name
        entries = self._entries_by_path.get(normalized)
        if not entries:
            raise KeyError(f"File not found in diff: {file_name}")
        return "\n".join(entry.text for entry in entries)

    def iter_file_diffs(self) -> Iterable[tuple[str, str]]:
        for path in self._file_order:
            yield path, self.get_file_diff(path)

    def get_cleaned_diff(self) -> str:
        if not self._entries:
            return ""
        lines: list[str] = []
        if self._preamble_lines:
            lines.extend(self._preamble_lines)
        for entry in self._entries:
            lines.extend(entry.lines)
        return "\n".join(lines)
    
    def get_number_of_lines(self) -> int:
        count = 0
        for entry in self._entries:
            count += len(entry.lines)
        return count

    def extract_created_file_content(self, file_name: str) -> str:
        """
        Extract the content of a newly created file from the diff.

        Parameters:
          file_name: target file name, e.g. "foo.py" or "path/to/foo.py"

        Returns:
          The file content as a string.

        Raises:
          ValueError: if file_name is not found as a created file in the diff.
        """
        normalized = _normalize_path(file_name) or file_name
        entries = self._entries_by_path.get(normalized)

        if not entries:
            raise ValueError(f"File {file_name!r} not found as a created file in diff.")

        content_lines: list[str] = []
        for entry in entries:
            for line in entry.lines:
                if line.startswith("diff --git "):
                    continue
                if line.startswith("@@") or line.startswith("--- "):
                    continue

                if line.startswith("+") and not line.startswith("+++ "):
                    content_lines.append(line[1:])
                elif line.startswith(" "):
                    content_lines.append(line[1:])

        return "\n".join(content_lines)

    def __len__(self) -> int:
        return len(self._file_order)
