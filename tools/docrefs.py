#!/usr/bin/env python3
"""THE DOC REFERENCES: every place a doc cites the tree is pinned to the content it was last read against.

    python3 tools/docrefs.py                   # the gate: every reference pinned, the lock matching the docs
    python3 tools/docrefs.py pending           # the references whose target changed or vanished, with the change
    python3 tools/docrefs.py accept DOC...     # pin DOC's new and changed references, read against their targets
    python3 tools/docrefs.py accept --whole DOC  # the docs pass read all of DOC: every reference in it is read
    python3 tools/docrefs.py prune             # drop the pins of references no doc makes any more
    python3 tools/docrefs.py --closing         # the docs pass is done: nothing pending, unread or broken
    python3 tools/docrefs.py init              # pin today's references, unread; refused when a lock exists

A REFERENCE is a doc's citation of a primitive: `` `tools/ratify.py`'s `main` ``, a dotted name in code font whose
module is in the tree (`` `rola.ops.carry.carry_forward` ``), a path in code font, or a link (`[x](build.md#wheels)`).
Its TARGET is a function, method, class or constant (Python, found by `ast`); a function or a `//: @ref NAME` ...
`//: @end` region (C++/CUDA); a markdown section, from its heading to the next heading at its level or above; or a
whole file or directory, which is checked to exist and never hashed. A target's content is normalized before it is
hashed -- Python through `ast.unparse` with docstrings dropped, everything else with comments dropped and whitespace
collapsed -- so a reformat or a comment edit moves no pin.

`docs/references.lock.json` holds one pin per (doc, target) -- `.claude/references.lock.json` for the docs the
public mirror does not ship: the digest, the commit it was pinned at, and whether a docs pass has read it (`init` pins today's references UNREAD). A pinned reference whose target's digest moved is
PENDING, and so is one whose target vanished; neither fails the gate, because docs are rewritten in one pass with the
whole change in view, and `pending` shows each target's change since its pin. The gate fails on a reference the lock
does not hold (a new citation: read it, then `accept` its doc), a new reference that resolves to nothing, and a pin
whose reference no doc makes any more (`prune`). The references that were already broken at `init` are the lock's
backlog: it only shrinks. `--closing` fails on anything pending, unread or broken.
Docs: docs/internals/tools/docrefs.md.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import functools
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The pins live with the docs they pin: a doc the public mirror does not ship (`.github/mirror/declarations.json`'s
#: `private`) keeps its pins in the private lock, so the public lock never names it.
LOCKS = {"public": "docs/references.lock.json", "private": ".claude/references.lock.json"}
MIRROR_DECLARATIONS = ".github/mirror/declarations.json"
#: A dotted name in code font is a reference when its first component is one of these; the value is the directory
#: the package imports from.
PACKAGES = {"rola": "", "tests": "", "tools": "", "benchmarks": "", "bench": "benchmarks", "rola_cu13": ""}
#: A path in code font is a reference when it ends in one of these, or in `/`.
PATH_SUFFIXES = (".py", ".md", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".json", ".toml", ".yaml", ".yml", ".sh", ".in",
                 ".cfg", ".txt", ".lock")
CPP_SUFFIXES = (".cu", ".cuh", ".cpp", ".h", ".hpp")

POSSESSIVE = re.compile(r"`([^`\s]+)`'s `([A-Za-z_][\w.]*)(?:\(\))?`")
CODE = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\]\(([^)\s]+)\)")
DOTTED = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\(\))?$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
EXPLICIT_ANCHOR = re.compile(r'<a id="([A-Za-z0-9_-]+)"')


@dataclass(frozen=True, order=True)
class Target:
    path: str  #: repo-relative
    name: str = ""  #: a primitive's qualified name, a markdown anchor, or "" for the whole file

    def __str__(self) -> str:
        return f"{self.path}#{self.name}" if self.path.endswith(".md") and self.name else (
            f"{self.path}::{self.name}" if self.name else self.path)


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return set(out.splitlines())


#: A ledger row describes the tree AT ITS COMMIT, so its citations are never re-pinned to a later tree.
HISTORY = ("docs/internals/DELETIONS.md",)


def docs(files: set[str]) -> list[str]:
    return sorted(f for f in files if f.endswith(".md") and "third_party" not in f and f not in HISTORY)


# ------------------------------------------------------------------------------------------------ finding references

def _prose_lines(text: str):
    """``(lineno, line)`` outside fenced code blocks."""
    fenced = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            yield lineno, line


def _module_file(parts: list[str], files: set[str]) -> str | None:
    stem = "/".join([*filter(None, [PACKAGES[parts[0]]]), *parts])
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if candidate in files:
            return candidate
    return None


def _dotted(span: str, files: set[str]) -> Target | None:
    match = DOTTED.match(span)
    if not match or match.group(1).split(".")[0] not in PACKAGES:
        return None
    parts = match.group(1).split(".")
    for k in range(len(parts), 0, -1):
        module = _module_file(parts[:k], files)
        if module:
            return Target(module, ".".join(parts[k:]))
    return Target("/".join(parts) + ".py", "")  #: a name under our packages that is no module: broken


def _path(span: str, doc: str, files: set[str]) -> Target | None:
    span = span.split("::")[0]
    if any(c in span for c in " <>*{}$~|,;=()[]…") or span.startswith(("/", "http", "-", "./", "../")) or "/" not in span:
        return None
    path, _, anchor = span.partition("#")
    path = path.split(":")[0]
    if not (path.endswith(PATH_SUFFIXES) or path.endswith("/")):
        return None
    if path.split("/")[0] not in {f.split("/")[0] for f in files}:
        return None
    return Target(path.rstrip("/") if path.endswith("/") else path, anchor if path.endswith(".md") else "")


def _resolve_path(name: str, files: set[str]) -> str | None:
    """A path as a doc writes it: repo-relative, or the unique tracked file it is a trailing part of."""
    if name in files or any(f.startswith(name.rstrip("/") + "/") for f in files):
        return name
    hits = [f for f in files if f.endswith("/" + name)]
    return hits[0] if len(hits) == 1 else None


def _link_target(doc: str, link: str) -> Target:
    """A markdown link's file, relative to the doc, and its section when it names one."""
    path, _, anchor = link.partition("#")
    parts: list[str] = []
    for part in (doc if not path else f"{Path(doc).parent.as_posix()}/{path}").split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    resolved = "/".join(parts)
    return Target(resolved, anchor if resolved.endswith(".md") else "")


def mirrored(doc: str, files: set[str]) -> list[str]:
    """The sources a mirror page describes: `docs/internals/<rel>.md` mirrors `csrc/rola/src/<rel>` and
    `docs/internals/tools/<name>.md` mirrors `tools/<name>`, in each source suffix the tree carries, plus any path in
    code font on the page's title line or in its opening `Mirrors ...` / `The mirror of ...` sentence."""
    if not doc.startswith("docs/internals/"):
        return []
    rel = doc[len("docs/internals/"):-len(".md")]
    stems = [f"csrc/rola/src/{rel}", f"tools/{rel[len('tools/'):]}" if rel.startswith("tools/") else ""]
    sources = [f"{stem}{suffix}" for stem in filter(None, stems) for suffix in (*CPP_SUFFIXES, ".py", ".sh")
               if f"{stem}{suffix}" in files]
    head = (ROOT / doc).read_text().splitlines()[:8]
    for line in head:
        if line.startswith("# ") or re.match(r"^(Mirrors|The mirror of) `", line):
            sources += [p for p in CODE.findall(line) if p in files and p not in sources]
    return sources


def references(doc: str, files: set[str], pinned: frozenset[str] = frozenset()) -> dict[Target, int]:
    """Every target ``doc`` cites, with the first line citing it. On a mirror page a bare name in code font cites the
    mirrored source's primitive of that name when it resolves there, or when ``pinned`` (the doc's pinned targets)
    already holds it -- a pinned name whose primitive left is a vanished target, not a citation that went away."""
    found: dict[Target, int] = {}
    sources = mirrored(doc, files)
    version = Version(frozenset(files))
    for lineno, line in _prose_lines((ROOT / doc).read_text()):
        rest = line
        for path, name in POSSESSIVE.findall(line):
            if not path.endswith(PATH_SUFFIXES):
                continue
            found.setdefault(Target(_resolve_path(path, files) or path, name), lineno)
            rest = rest.replace(f"`{path}`'s `{name}`", "")
        for span in CODE.findall(rest):
            target = _dotted(span, files) or _path(span, doc, files)
            if target:
                found.setdefault(target, lineno)
                continue
            bare = re.fullmatch(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)(?:\(\))?", span)
            for source in sources if bare else ():
                candidate = Target(source, bare.group(1))
                if str(candidate) in pinned or locate(candidate, version) not in (None, ("", "")):
                    found.setdefault(candidate, lineno)
                    break
        for link in LINK.findall(rest):
            if not link.startswith(("http:", "https:", "mailto:")):
                found.setdefault(_link_target(doc, link), lineno)
    return found


# ------------------------------------------------------------------------------------------------ reading a target

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _strip_docstrings(node: ast.AST) -> ast.AST:
    node = copy.deepcopy(node)
    for inner in ast.walk(node):
        body = getattr(inner, "body", None)
        if (isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            inner.body = body[1:] or [ast.Pass()]
    return node


@dataclass(frozen=True)
class Version:
    """The tree a target is read in: the working tree, or a commit (`pending` reads a pin's)."""

    files: frozenset[str]
    commit: str | None = None

    def text(self, path: str) -> str | None:
        return _text(ROOT, self.commit, path)


@functools.cache
def _text(root: Path, commit: str | None, path: str) -> str | None:
    if commit is None:
        return (root / path).read_text(errors="replace") if (root / path).is_file() else None
    shown = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True, text=True)
    return shown.stdout if shown.returncode == 0 else None


@functools.cache
def _parsed(text: str) -> ast.Module:
    return ast.parse(text)


def _binds(stmt: ast.stmt, name: str) -> bool:
    """Whether ``stmt`` defines ``name``: a function, a class, or an assignment to it."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name == name
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
    return any(isinstance(t, ast.Name) and t.id == name for t in targets)


def _python(version: Version, path: str, name: str) -> tuple[str, str] | None:
    """``(source, normalized)`` of ``name`` in ``path``, following up to four `from x import name` re-exports."""
    for _ in range(5):
        text = version.text(path)
        scope: list[ast.stmt] = _parsed(text).body
        parts, node, moved = name.split("."), None, False
        for i, part in enumerate(parts):
            node = next((stmt for stmt in scope if _binds(stmt, part)), None)
            if node is None and i == 0:
                for stmt in scope:
                    if not (isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0
                            and stmt.module.split(".")[0] in PACKAGES):
                        continue
                    module = _module_file(stmt.module.split("."), version.files)
                    alias = next((a for a in stmt.names if (a.asname or a.name) == part), None)
                    if alias and module:
                        path, name, moved = module, ".".join([alias.name, *parts[1:]]), True
                        break
            if node is None:
                break
            scope = getattr(node, "body", [])
        if moved:
            continue
        if node is None:
            return None
        return ast.get_source_segment(text, node, padded=True) or "", ast.unparse(_strip_docstrings(node))
    return None


def _without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return " ".join(re.sub(r"//.*", " ", text).split())


def _cpp(text: str, name: str) -> str | None:
    """A ``//: @ref name`` region, else the first brace-matched definition of ``name``; "" when the file only names
    it."""
    region = re.search(rf"//:\s*@ref\s+{re.escape(name)}\s*\n(.*?)//:\s*@end\b", text, flags=re.S)
    if region:
        return region.group(1)
    for match in re.finditer(rf"\b{re.escape(name)}\s*(?:<[^;{{}}]*>)?\s*\(", text):
        line = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
        if not re.search(r"[\w*&>:]\s*$", line) or re.match(r"\s*(if|for|while|switch|return|else)\b", line):
            continue  #: a call, not a definition: a definition's name follows its return type
        head_end = text.find("{", match.end())
        if head_end < 0 or ";" in text[match.end():head_end]:
            continue
        depth, i = 0, head_end
        while i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                return text[text.rfind("\n", 0, match.start()) + 1:i + 1]
            i += 1
    return "" if re.search(rf"\b{re.escape(name)}\b", text) else None


def _slug(heading: str) -> str:
    heading = re.sub(r"[`*_]", "", heading).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading))


def _section(text: str, anchor: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        heading = HEADING.match(line)
        named = EXPLICIT_ANCHOR.search(line)
        if (heading and _slug(heading.group(2)) == anchor) or (named and named.group(1) == anchor):
            level = len(heading.group(1)) if heading else 7
            j = i + 1
            while j < len(lines) and not ((h := HEADING.match(lines[j])) and (len(h.group(1)) <= level or not heading)):
                j += 1
            return "\n".join(lines[i:j])
    return None


def _generated(path: str) -> bool:
    """A build's output the tree ignores (`rola_cu13/_build_config.py`): a doc may cite it, and no checkout has it."""
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


def locate(target: Target, version: Version) -> tuple[str, str] | None:
    """``(source, normalized)``: the target's text as written and as hashed; ``("", "")`` for a target checked only to
    exist; None when it resolves to nothing."""
    if not (target.path in version.files or any(f.startswith(target.path + "/") for f in version.files)):
        return ("", "") if version.commit is None and _generated(target.path) else None
    text = version.text(target.path)
    if not target.name or text is None:
        return ("", "") if not target.name else None
    if target.path.endswith(".py"):
        try:
            return _python(version, target.path, target.name)
        except SyntaxError:
            return None
    if target.path.endswith(".md"):
        section = _section(text, target.name)
        return None if section is None else (section, " ".join(section.split()))
    if target.path.endswith(CPP_SUFFIXES):
        body = _cpp(text, target.name)
        return None if body is None else ("", "") if body == "" else (body, _without_comments(body))
    return ("", "") if re.search(rf"\b{re.escape(target.name)}\b", text) else None


def read(target: Target, files: set[str]) -> str | None:
    """The target's digest in the working tree: "" for one checked only to exist, None when it resolves to nothing."""
    found = locate(target, Version(frozenset(files)))
    return None if found is None else ("" if found == ("", "") else _digest(found[1]))


def change(target: Target, pinned_at: str, files: set[str]) -> str:
    """The target's source as a unified diff from its pin to the working tree."""
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", pinned_at], cwd=ROOT, capture_output=True,
                            text=True).stdout.splitlines()
    before = locate(target, Version(frozenset(listed), pinned_at))
    after = locate(target, Version(frozenset(files)))
    old, new = ("" if x is None else x[0] for x in (before, after))
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"{target} @ {pinned_at}",
                                        f"{target} (working tree)"))


# ------------------------------------------------------------------------------------------------ the lock

def _visibility(doc: str) -> str:
    private = json.loads((ROOT / MIRROR_DECLARATIONS).read_text())["private"]
    return "private" if any(doc == p or doc.startswith(p.rstrip("/") + "/") for p in private) else "public"


def _load(blobs) -> dict[tuple[str, str], dict]:
    return {(e["doc"], e["target"]): e for blob in blobs for e in json.loads(blob)["references"]}


def _dump(entries: dict[tuple[str, str], dict]) -> None:
    for visibility, path in LOCKS.items():
        rows = [entries[k] for k in sorted(entries) if _visibility(k[0]) == visibility]
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / path).write_text("{\n \"references\": [\n" + ",\n".join("  " + json.dumps(r, sort_keys=True)
                                                                     for r in rows) + "\n ]\n}\n")


def _lock() -> dict[tuple[str, str], dict] | None:
    blobs = [(ROOT / path).read_text() for path in LOCKS.values() if (ROOT / path).is_file()]
    return _load(blobs) if blobs else None


def _head_lock() -> dict[tuple[str, str], dict] | None:
    blobs = []
    for path in LOCKS.values():
        shown = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, text=True)
        if shown.returncode == 0:
            blobs.append(shown.stdout)
    return _load(blobs) if blobs else None


def _head_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def survey(files: set[str], lock: dict | None) -> dict[tuple[str, str], tuple[Target, int, str | None]]:
    """``(doc, target) -> (target, line, digest)`` over every doc in the tree."""
    out = {}
    for doc in docs(files):
        pinned = frozenset(target for d, target in (lock or {}) if d == doc)
        for target, line in references(doc, files, pinned).items():
            out[(doc, str(target))] = (target, line, read(target, files))
    return out


def verdict(found, lock, head):
    """``(new, stale, grown, pending)``: references the lock lacks, pins no doc makes, broken pins HEAD's lock did not
    hold, and pinned references whose target's digest moved (a vanished target reads None)."""
    new = {k: v for k, v in found.items() if k not in lock}
    stale = sorted(k for k in lock if k not in found)
    grown = sorted(k for k, v in lock.items() if v["sha256"] is None and head is not None
                   and (k not in head or head[k]["sha256"] is not None))
    pending = {k: (found[k], lock[k]) for k in found.keys() & lock.keys()
               if lock[k]["sha256"] is not None and found[k][2] != lock[k]["sha256"]}
    return new, stale, grown, pending


def _pin(target: Target, digest: str | None, sha: str, doc: str, reviewed: bool) -> dict:
    return {"doc": doc, "target": str(target), "sha256": digest, "pinned_at": sha, "reviewed": reviewed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="check", choices=("check", "pending", "accept", "prune", "init"))
    ap.add_argument("docs", nargs="*", help="accept: the docs whose references were read")
    ap.add_argument("--closing", action="store_true", help="check: also fail on anything pending, unread or broken")
    ap.add_argument("--whole", action="store_true",
                    help="accept: every reference in the doc was read, not only the new and changed ones")
    a = ap.parse_intermixed_args()
    files = tracked()
    found = survey(files, _lock())

    if a.command == "init":
        if _lock() is not None:
            raise SystemExit(f"docrefs: {' and '.join(LOCKS.values())} exist; the locks are initialized once")
        sha = _head_sha()
        _dump({k: _pin(t, d, sha, k[0], False) for k, (t, _, d) in found.items()})
        broken = sum(1 for _, _, d in found.values() if d is None)
        print(f"docrefs: pinned {len(found)} reference(s) unread, {broken} of them broken")
        return 0
    lock = _lock()
    if lock is None:
        raise SystemExit("docrefs: no lock (python3 tools/docrefs.py init records one)")

    if a.command == "accept":
        if not a.docs:
            raise SystemExit("docrefs accept: name the docs whose references you read")
        sha, refused = _head_sha(), []
        for doc in a.docs:
            doc = str(Path(doc).resolve().relative_to(ROOT)) if Path(doc).exists() else doc
            for key, (target, line, digest) in sorted((k, v) for k, v in found.items() if k[0] == doc):
                pin = lock.get(key)
                if digest is None:
                    if pin is None or pin["sha256"] is not None:
                        refused.append(f"  {doc}:{line}: `{target}` resolves to nothing")
                    continue
                if pin is None or pin["sha256"] != digest or (a.whole and not pin["reviewed"]):
                    lock[key] = _pin(target, digest, sha, doc, True)
        _dump(lock)
        if refused:
            print("docrefs accept: these references stay unpinned until they name something that exists:")
            print("\n".join(refused))
            return 1
        print(f"docrefs: accepted {', '.join(a.docs)}")
        return 0

    if a.command == "prune":
        gone = [k for k in lock if k not in found]
        for key in gone:
            del lock[key]
        _dump(lock)
        print(f"docrefs: pruned {len(gone)} pin(s) no doc makes any more")
        return 0

    new, stale, grown, pending = verdict(found, lock, _head_lock())

    if a.command == "pending":
        by_target: dict[Target, list] = {}
        for (doc, _), ((target, line, digest), pin) in sorted(pending.items()):
            by_target.setdefault(target, []).append((doc, line, digest, pin["pinned_at"]))
        for target, cites in sorted(by_target.items()):
            print(f"{target}: {'VANISHED' if cites[0][2] is None else 'CHANGED'} since {cites[0][3]}")
            for doc, line, _, _ in cites:
                print(f"  cited by {doc}:{line}")
            print(change(target, cites[0][3], files))
        print(f"docrefs: {len(by_target)} target(s) pending across {len(pending)} reference(s)")
        return 0

    for (doc, target), (t, line, digest) in sorted(new.items()):
        why = "resolves to nothing" if digest is None else "is not pinned: read it against its target, then `accept`"
        print(f"{doc}:{line}: the reference to `{target}` {why}")
    for doc, target in stale:
        print(f"{doc}: the pin for `{target}` names a reference the doc no longer makes (`prune`)")
    for doc, target in grown:
        print(f"{doc}: `{target}` joined the broken backlog, which only shrinks")
    unread = sum(1 for v in lock.values() if not v["reviewed"])
    broken = sum(1 for v in lock.values() if v["sha256"] is None)
    failed = bool(new or stale or grown)
    if a.closing:
        for (doc, target), ((_, line, digest), _) in sorted(pending.items()):
            print(f"{doc}:{line}: `{target}` {'vanished' if digest is None else 'changed'} since its pin")
        for (doc, target), pin in sorted(lock.items()):
            if pin["sha256"] is None:
                print(f"{doc}: `{target}` resolves to nothing")
        unread_docs = sorted({doc for (doc, _), pin in lock.items() if not pin["reviewed"]})
        if unread_docs:
            print(f"docs with unread references (`accept --whole` once read): {', '.join(unread_docs)}")
        failed = failed or bool(pending or unread or broken)
    print(f"docrefs: {'FAIL' if failed else 'OK'} -- {len(found)} reference(s), {len(pending)} pending, {unread} unread, "
          f"{broken} broken")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
