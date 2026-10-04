"""Strict Edit in the agent-action classes (scripts/analysis/plot_action_classes.py)."""

from __future__ import annotations

import shlex

import pytest

from scripts.analysis import plot_action_classes as pac

REPO = frozenset({"Solution.lean", "Foo/Bar.lean", "Foo/Baz.lean", "lakefile.toml",
                  "lean-toolchain", "lake-manifest.json", ".gitignore"})
SCRATCH_CHECK = ("cat > PalomarProof.lean <<'EOF'\nimport PalomarCommon\ntheorem x : True := by\n  simp\nEOF\n"
                 "lake env lean PalomarProof.lean")


@pytest.mark.parametrize("name, args, expected", [
    # A scratch file at the repository root, then a check of it.
    ("exec_command", {"command": SCRATCH_CHECK}, "Lake"),
    ("Bash", {"command": "/bin/bash -lc " + shlex.quote(SCRATCH_CHECK)}, "Lake"),
    # Edits win over the check that follows in the same call.
    ("Bash", {"command": "cat > Foo/Bar.lean <<'EOF'\nx\nEOF\nlake build Solution"}, "Edit"),
    ("Bash", {"command": "cat > Foo/Bar.lean <<'EOF' && lake build\nx\nEOF"}, "Edit"),
    ("Bash", {"command": "sed -i 's/a/b/' Foo/Bar.lean && python3 proof_length.py"}, "Edit"),
    ("Bash", {"command": "perl -0pi -e 's/a/b/' Solution.lean; lean_verify"}, "Edit"),
    ("Bash", {"command": "cp /tmp/defs.bak Foo/Bar.lean && lean_verify"}, "Edit"),
    ("Bash", {"command": "python3 - <<'EOF'\np='Foo/Bar.lean'\ns=open(p).read()\nopen(p,'w').write(s)\nEOF\n"
                         "lean_verify"}, "Edit"),
    ("Bash", {"command": "python3 - <<'EOF'\nfrom pathlib import Path\nfor p in Path('/testbed').rglob('*.lean'):\n"
                         "    p.write_text(p.read_text())\nEOF"}, "Edit"),
    ("Bash", {"command": "rm Foo/Baz.lean && lake build"}, "Edit"),
    ("Bash", {"command": "cd Foo && sed -i 's/a/b/' Bar.lean"}, "Edit"),
    ("Bash", {"command": "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: /testbed/Foo/Bar.lean\n@@\n-a\n+b\n"
                         "*** End Patch\nEOF"}, "Edit"),
    # A new module beside existing sources is an edit; root-level and helper files are scratch.
    ("Bash", {"command": "python3 - <<'EOF'\np='Foo/Notation.lean'\nopen(p,'w').write('x')\nEOF"}, "Edit"),
    ("Edit", {"file_path": "/testbed/Foo/New.lean"}, "Edit"),
    ("Edit", {"file_path": "/testbed/Scratch.lean"}, "Other"),
    ("Write", {"file_path": "/testbed/tools/dump.lean"}, "Other"),
    # Writes outside the repository never count.
    ("Bash", {"command": "cat > /tmp/w/T.lean <<'EOF'\nx\nEOF\nlake env lean /tmp/w/T.lean"}, "Lake"),
    ("Bash", {"command": "cd /tmp/w && cat > X.lean <<'EOF' && lake build\nx\nEOF"}, "Build"),
    ("Bash", {"command": "cd /tmp/w && python3 - <<'EOF'\ns=open('/testbed/Foo/Bar.lean').read()\n"
                         "open('out.lean','w').write(s)\nEOF"}, "Read"),
    ("Bash", {"command": "rm -rf __pycache__; lake build PalomarSolution > /tmp/w/b.log 2>&1"}, "Build"),
    ("Bash", {"command": "cp Foo/Bar.lean /tmp/defs.bak && lean_verify"}, "Verify"),
    ("Bash", {"command": "git show HEAD:Foo/Bar.lean > /tmp/orig.lean && grep -n x /tmp/orig.lean"}, "Git"),
    ("Bash", {"command": "grep -rn foo . 2>/dev/null | head"}, "Search"),
    ("Bash", {"command": "sed -n '1,20p' Foo/Bar.lean"}, "Read"),
    ("Bash", {"command": "mkdir -p /tmp/w && lake build"}, "Build"),
    ("FileChange", {"changes": [{"kind": "add", "path": "/testbed/Scratch.lean"}]}, "Other"),
    ("FileChange", {"changes": [{"kind": "update", "path": "/testbed/Foo/Bar.lean"}]}, "Edit"),
    ("FileChange", {"changes": "[{'kind': 'update', 'path': '/testbed/Solution.lean'}]"}, "Edit"),
    ("Write", {"file_path": "/tmp/w/gen.py", "content": "x"}, "Other"),
    ("write_file", {"path": "/tmp/encode_replay.py"}, "Other"),
    ("replace_file_content", {"TargetFile": "/testbed/Foo/Bar.lean"}, "Edit"),
])
def test_strict_edit_classes(name, args, expected):
    assert pac.classify(name, args, REPO)[0] == expected


def test_heredoc_inside_nested_shell_keeps_its_body():
    command = "/bin/bash -lc \"python3 - <<'PY'\nfrom pathlib import Path\nPath('Foo/Bar.lean').write_text('x')\nPY\""
    assert pac.classify("Bash", {"command": command}, REPO)[0] == "Edit"


def test_double_quoted_wrapper_unescapes_the_heredoc():
    # Codex double-quotes scripts containing single quotes, escaping the payload's quotes.
    inner = ("python3 - <<'PY'\np='Foo/Bar.lean'\ns=open(p).read()\ns=s.replace(\"a\", \"b\")\n"
             "open(p,'w').write(s)\nPY\nlake build")
    command = '/bin/bash -lc "' + inner.replace("\\", "\\\\").replace('"', '\\"') + '"'
    assert pac.classify("Bash", {"command": command}, REPO)[0] == "Edit"


def test_repository_files_cover_sources_assets_and_lake_metadata():
    report = {
        "protected_provenance": {"solution_source": "Solution.lean", "import_closure": {
            "retained_local_sources": ["Solution.lean", "Foo/Bar.lean", "Foo/Gone.lean"],
            "retained_local_assets": ["scripts/gen.py"], "lake_metadata": {"path": "lakefile.toml"}}},
        "module_pruning": {"deleted_files": ["Foo/Gone.lean"]},
    }
    assert pac.repository_files(report) == {"Solution.lean", "Foo/Bar.lean", "scripts/gen.py", "lakefile.toml",
                                            "lean-toolchain", "lake-manifest.json", ".gitignore"}


def test_lean_strip_datasets_fall_back_to_the_stripped_tree(tmp_path):
    # lean-strip reports carry no import closure; the published tree is the repository.
    directory = tmp_path / "repos/r"
    for path in ("stripped/Foo/Bar.lean", "stripped/lakefile.toml", "stripped/.lake/build/x.olean"):
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_text("")
    (directory / "prod-strip-report.json").write_text('{"protected_provenance": {}}')
    assert pac.starting_repository(tmp_path, "r") == {"Foo/Bar.lean", "lakefile.toml"}
