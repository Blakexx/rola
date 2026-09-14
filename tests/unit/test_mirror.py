"""The public mirror's export (.github/mirror/mirror.py): every tracked path is declared, the longest declaration
wins, and the tripwire refuses agent files and private-shaped strings, including in the tool's own text. No GPU."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("mirror", ROOT / ".github" / "mirror" / "mirror.py")
mirror = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mirror)
DECL = mirror.declarations()
#: a home-directory path, joined so this file carries none (§17); the fixtures below split their other
#: private shapes the same way, so the file passes the tripwire it tests
SOMEONES_HOME = "/".join(("", "home", "someone"))


def test_every_tracked_path_is_declared():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    assert mirror.undeclared([p for p in tracked.decode().split("\0") if p], DECL) == []


def test_the_longest_declaration_wins():
    assert mirror.declaration(".github/workflows/lint.yml", DECL) == "ships"
    assert mirror.declaration(".github/workflows/mirror.yml", DECL) == "private"
    assert mirror.declaration("CLAUDE.md", DECL) == "private"
    assert mirror.declaration("rola/layer.py", DECL) == "ships"
    assert mirror.declaration("rolax/layer.py", DECL) is None


def test_a_clean_export_passes_and_the_tool_does_not_trip_on_itself(tmp_path):
    (tmp_path / ".github" / "mirror").mkdir(parents=True)
    for name in ("mirror.py", "README.md", "declarations.json"):
        shutil.copy(ROOT / ".github" / "mirror" / name, tmp_path / ".github" / "mirror" / name)
    (tmp_path / "README.md").write_text("RoLA\n")
    assert mirror.tripwire(tmp_path, {}) == []


@pytest.mark.parametrize(("name", "text", "finding"), [
    ("csrc/CLAUDE.md", "notes", "an agent file"),
    ("docs/a/.claude/settings.json", "{}", "an agent file"),
    ("docs/x.md", f"built at {SOMEONES_HOME}/rola", "a home directory"),
    ("docs/x.md", "contact someone" + "@gmail.com", "a personal email address"),
    ("tools/x.py", "token = 'ghp_" + "a" * 36 + "'", "a credential"),
])
def test_the_tripwire_refuses_private_shaped_files(tmp_path, name, text, finding):
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text(text)
    assert mirror.tripwire(tmp_path, {}) == [f"{name}: {finding}"]


def test_an_allowed_shape_passes_but_a_credential_can_never_be_allowed(tmp_path):
    (tmp_path / "r.json").write_text(f'{{"rola_file": "{SOMEONES_HOME}/rola/__init__.py"}}')
    assert mirror.tripwire(tmp_path, {"a home directory": "records name their checkout"}) == []
    declarations = tmp_path / "declarations.json"
    declarations.write_text('{"public": "o/r", "ships": [], "private": [], "allow": {"a credential": "no"}}')
    with pytest.raises(SystemExit, match="no allowable shape"):
        mirror.declarations(declarations)
