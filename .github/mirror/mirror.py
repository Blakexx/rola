"""THE PUBLIC MIRROR'S EXPORT: the committed files a repository publishes, written to a directory, refused when a
tracked path has no declaration or anything private-shaped would leave.

    python .github/mirror/mirror.py EXPORT_DIR   write HEAD's export; exit 1 on any refusal
    python .github/mirror/mirror.py --check      every tracked path is declared; exit 1 naming the ones that are not
    python .github/mirror/mirror.py --public     print the public repository (owner/name) the export is pushed to

`declarations.json` beside this file is the repository's own part; this file is the same in every repository that
mirrors. See README.md beside it.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: agent and scratch files, refused at any depth
AGENT_FILES = re.compile(r"(^|/)(\.claude/|CLAUDE\.md$|SCRATCHPAD\.md$)")
#: private-shaped strings. Each is spelled so that this file's own text does not match it: the tool is scanned with
#: everything else wherever it ships.
PRIVATE_SHAPES = {
    "a home directory": re.compile(r"/home/[a-z]|/Users/[A-Za-z]"),
    "the private scratch repository": re.compile(r"rola-scr[a]tch"),
    "a personal email address": re.compile(r"[\w.]+@gmail[.]com"),
    "a cloud project or bucket": re.compile(r"project-[0-9a-f]{8}-[0-9a-f]{4}|cla-zoology-sw[e]ep"),
    "a credential": re.compile(r"ghp_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}|hf_[A-Za-z0-9]{30}|sk-ant-[a-z0-9]"
                               r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"),
}
#: shapes no declaration may allow
NEVER_ALLOWED = ("a credential",)


def declarations(path: Path = HERE / "declarations.json") -> dict:
    blob = json.loads(path.read_text())
    if set(blob) != {"public", "ships", "private", "allow"}:
        raise SystemExit(f"{path}: fields must be exactly public, ships, private, allow; got {sorted(blob)}")
    unknown = set(blob["allow"]) - set(PRIVATE_SHAPES)
    forbidden = set(blob["allow"]) & set(NEVER_ALLOWED)
    if unknown or forbidden:
        raise SystemExit(f"{path}: `allow` names {sorted(unknown | forbidden)}, which is no allowable shape")
    return blob


def declaration(path: str, decl: dict) -> str | None:
    """"ships" or "private" by the longest entry `path` falls under; None when it falls under none."""
    matches = [(len(entry), kind) for kind in ("ships", "private") for entry in decl[kind]
               if path == entry or path.startswith(entry.rstrip("/") + "/")]
    return max(matches)[1] if matches else None


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def undeclared(paths: list[str], decl: dict) -> list[str]:
    return [p for p in paths if declaration(p, decl) is None]


def tripwire(export: Path, allow: dict) -> list[str]:
    findings = []
    for path in sorted(p for p in export.rglob("*") if p.is_file()):
        rel = path.relative_to(export).as_posix()
        if AGENT_FILES.search(rel):
            findings.append(f"{rel}: an agent file")
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        findings += [f"{rel}: {name}" for name, shape in PRIVATE_SHAPES.items()
                     if name not in allow and shape.search(text)]
    return findings


def export(target: Path, decl: dict) -> list[str]:
    """Write HEAD's shipping files under `target`; the refusals, empty when the export is clean."""
    paths = [p for p in _git("ls-tree", "-r", "-z", "--name-only", "HEAD").decode().split("\0") if p]
    missing = undeclared(paths, decl)
    if missing:
        return [f"{p}: tracked but declared neither ships nor private" for p in missing]
    ships = [p for p in paths if declaration(p, decl) == "ships"]
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(_git("archive", "--format=tar", "HEAD", "--", *ships))) as tar:
        tar.extractall(target, filter="data")
    return tripwire(target, decl["allow"])


def main(argv: list[str]) -> int:
    decl = declarations()
    if argv == ["--public"]:
        print(decl["public"])
        return 0
    if argv == ["--check"]:
        refusals = [f"{p}: tracked but declared neither ships nor private"
                    for p in undeclared([p for p in _git("ls-files", "-z").decode().split("\0") if p], decl)]
    elif len(argv) == 1 and not argv[0].startswith("-"):
        refusals = export(Path(argv[0]), decl)
    else:
        raise SystemExit(__doc__)
    for refusal in refusals:
        print(f"MIRROR REFUSED: {refusal} (.github/mirror/declarations.json)")
    return 1 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
