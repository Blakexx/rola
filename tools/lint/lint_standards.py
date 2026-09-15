#!/usr/bin/env python3
"""Textual/preprocessor standards checks that ast-grep's AST rules cannot see.

ast-grep (tools/lint/rules/*.yml) covers shapes a parser can express: warp
collectives under a short-circuit, a `__global__` in an anonymous namespace,
a width-shaped `if (kDv == ...)`, a `#include <torch/...>` node. Three checks
in KERNEL_STANDARDS.md are either about the PREPROCESSOR (a `#pragma` has no
AST node next to the statement it modifies in every grammar ast-grep ships)
or about POSITION within one named file (a marker comment, not a syntax
shape) -- those live here as plain line-oriented scans.

Checks:
  1. UNROLL DISCIPLINE (KERNEL_STANDARDS.md §6). A `for`/`while` loop in
     device code (any file under csrc/rola/, excluding csrc/third_party/)
     whose bound is NOT compile-time carries `#pragma unroll 1` immediately
     before it. This is a HEURISTIC, not a compiler: it approximates
     "compile-time bound" by regexing the loop's stop expression, and it can
     only be conservative in one direction --  see HEURISTIC LIMITS below.
  2. NO TORCH INCLUDE IN ARM TUs (belt to the ast-grep rule
     torch_include_in_arm_tu.yml). A `#include` line naming `torch/`
     under csrc/rola/src/instantiations/, found by TEXT rather than AST so a
     macro-guarded or unusually-formatted include cannot hide from either
     check.
  3. NO __syncthreads AFTER THE DECODE PROLOGUE (the decode prologue law, ledger
     2026-08-27: "11 barriers, all in the prologue ... 0 after it"; scope
     fixed 2026-08-29 -- that law is about CTA/grid barriers,
     never about `__syncwarp`, which is intra-warp and which §12 REQUIRES
     around a warp collective). The marker is decode.cu's own section
     comment `// ---- THE WALK`; every `__syncthreads`/`bar.sync` (and any
     cluster/grid barrier) after that line is a finding for Blake to rule on
     -- `__syncwarp` after the marker is NOT a finding, it is §12's fence
     doing its job.

HEURISTIC LIMITS (check 1):
  - "Device code" is found by brace-counting from each `__global__`/`__device__`
    qualifier to its function's matching close brace -- a `.cu` mixes host
    launcher code (std::vector loops, TORCH_CHECK, TensorOptions) with kernels,
    and only the latter is in scope. The brace count does not parse string/char
    literals, so a literal containing an unbalanced `{`/`}` (none observed in
    this codebase) would misplace a span's end.
  - No preprocessor expansion and no constant folding: `for (i = 0; i < N; )`
    is judged by whether `N` LOOKS compile-time (an integer literal, `sizeof`,
    or an ALL_CAPS / `k`-prefixed identifier -- the project's constexpr/
    template-param naming convention), not by resolving what `N` actually is.
    A `constexpr int n = ...;` bound one line above with a lowercase name
    reads as "dynamic" and asks for a pragma it does not need (a false
    positive the script cannot avoid without a real preprocessor+compiler
    front end); a `#define BOUND 64` used as `i < BOUND` reads as constexpr
    correctly only because BOUND matches the ALL_CAPS heuristic.
  - Only an explicit `#pragma unroll <N>` (any measured integer factor, not
    just `1`) directly above the loop counts -- a BARE `#pragma unroll` (no
    factor: full unroll, left to the compiler) does not, and neither does no
    pragma at all. Checked only on the lines immediately above the loop
    (comments in between are skipped, anything else breaks the search). A
    pragma placed elsewhere (e.g. hoisted above an
    unrelated statement) is invisible to this check.
  - Loop headers spanning multiple lines are matched as best-effort: the
    stop-expression regex runs on the concatenation of up to 3 lines from the
    `for`/`while` token, which can under- or over-capture on unusual
    formatting.
  - `while` loops with no explicit relational stop expression (e.g. an
    infinite `while (true)` with a `break`) are skipped: there is no "bound"
    to classify, so nothing is asked of them.
This is a lint, not a gate: every finding is reported for Blake to rule on,
never silently fixed or silenced by this script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC_ROLA = ROOT / "csrc" / "rola"
INSTANTIATIONS = CSRC_ROLA / "src" / "instantiations"
DECODE_CU = CSRC_ROLA / "src" / "decode" / "decode.cu"
TESTS_ROOT = ROOT / "tests"

#: docs/testing.md: "there is no fourth tier". A test answers one of exactly three
#: questions (oracle/integration/unit), so it lives in exactly one of these three
#: directories -- never a fourth bucket, campaign-named or otherwise (`tests/k31/`
#: was exactly that, folded into the tier each of its tests actually answered).
TESTS_ALLOWED_TIERS = {"unit", "integration", "oracle"}

DEVICE_SUFFIXES = (".cu", ".cuh")

#: Identifiers that read as compile-time under this project's naming
#: convention: ALL_CAPS (template params / macros: NC, DV, D) or a
#: `k`-prefixed CamelCase constant (kDecodeWarps, kMaxLevels, kKinds).
CONSTEXPR_LOOKING = re.compile(r"^\s*(?:sizeof\s*\(.*\)|\d+|k[A-Z]\w*|[A-Z][A-Z0-9_]*)\s*$")

FOR_RE = re.compile(r"^\s*for\s*\(")
WHILE_RE = re.compile(r"^\s*while\s*\(")
#: The stop clause of a `for (init; STOP; step)` -- second `;`-separated field.
FOR_STOP_RE = re.compile(r"for\s*\([^;]*;\s*([^;]*);")
#: A `while (COND)` condition, relational-shaped: `X < BOUND`, `X <= BOUND`, ...
WHILE_COND_RE = re.compile(r"while\s*\((.*)\)\s*\{?\s*$")
REL_BOUND_RE = re.compile(r"[<>]=?\s*(.+?)\s*$")

#: `unroll 1` (no unrolling) is the common case, but §6's actual requirement is an
#: EXPLICIT pragma on a dynamic-bound loop, not that specific factor -- a bare
#: `#pragma unroll <N>` for any measured N also satisfies it (decode.cu's
#: `n_live` loop, docs/internals/decode/decode.md#decode-step-kernel-note-l655:
#: "at 1 it is a correctness [performance] defect on this loop, not a style
#: choice" -- unroll 8 there is measured, not a default left to the compiler).
PRAGMA_UNROLL_RE = re.compile(r"^\s*#pragma\s+unroll(\s+\d+)?\s*$")
COMMENT_LINE_RE = re.compile(r"^\s*//")

TORCH_INCLUDE_RE = re.compile(r'#include\s*<\s*torch/')

#: The prologue law is about CTA/grid barriers only (`__syncthreads`, PTX `bar.sync`,
#: and a cluster/grid barrier such as `cluster.sync()`/`grid.sync()`);
#: `__syncwarp` is intra-warp and is §12's REQUIRED fence around a warp
#: collective, never this rule's concern (the rule's one
#: remaining finding after MERGE-D1 was decode.cu:722's `__syncwarp`).
SYNC_RE = re.compile(
    r"__syncthreads\s*\(|\bbar\.sync\b|(?:cluster|grid)\s*\.\s*sync\s*\(")
WALK_MARKER_RE = re.compile(r"----\s*THE WALK")


def device_files():
    for suf in DEVICE_SUFFIXES:
        for p in CSRC_ROLA.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            yield p


DEVICE_QUALIFIER_RE = re.compile(r"\b__(global|device)__\b")


#: A `.cu` mixes HOST launcher code (std::vector loops over `widths`, TORCH_CHECK,
#: TensorOptions -- none of it unroll-relevant) with the device kernels §6 is
#: actually about. Scope the check to the textual span of each `__global__`/
#: `__device__` function body, found by brace-counting from the qualifier line
#: to the matching close -- a heuristic itself (see module docstring): it does
#: not parse strings/char literals, so a stray unbalanced brace in a string
#: literal (none observed in this codebase) would throw off the span.
def device_code_line_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges = []
    n = len(lines)
    i = 0
    while i < n:
        if DEVICE_QUALIFIER_RE.search(lines[i]) and not COMMENT_LINE_RE.match(lines[i]):

            # Find the function's opening brace, scanning forward (signatures
            # with __launch_bounds__(...) or wrapped parameter lists span
            # multiple lines).
            j = i
            open_line = None
            while j < n:
                for ch in lines[j]:
                    if ch == "{":
                        open_line = j
                        break
                if open_line is not None:
                    break
                j += 1
            if open_line is None:
                i += 1
                continue
            depth = 0
            k = open_line
            close_line = None
            started = False
            while k < n:
                for ch in lines[k]:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                        if started and depth == 0:
                            close_line = k
                            break
                if close_line is not None:
                    break
                k += 1
            if close_line is None:
                close_line = n - 1  # unterminated scan -- best effort to EOF
            ranges.append((i, close_line))
            i = close_line + 1
        else:
            i += 1
    return ranges


def in_any_range(idx: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= idx <= hi for lo, hi in ranges)


def bound_looks_constexpr(bound: str) -> bool:
    bound = bound.strip()
    if not bound:
        return True  # nothing to judge (e.g. infinite `while (true)` handled earlier)
    return bool(CONSTEXPR_LOOKING.match(bound))


def preceding_pragma_unroll1(lines: list[str], loop_line_idx: int) -> bool:
    """True iff a `#pragma unroll 1` (explicit 1, not bare) sits directly
    above the loop, skipping over `//` comment lines only."""
    i = loop_line_idx - 1
    while i >= 0 and COMMENT_LINE_RE.match(lines[i]):
        i -= 1
    if i < 0:
        return False
    m = PRAGMA_UNROLL_RE.match(lines[i])
    return bool(m and m.group(1))  # group(1) present => "unroll 1" explicitly


def check_unroll_discipline() -> list[str]:
    findings = []
    for path in device_files():
        rel = path.relative_to(ROOT)
        lines = path.read_text().splitlines()
        ranges = device_code_line_ranges(lines)
        for idx, line in enumerate(lines):
            if not in_any_range(idx, ranges):
                continue  # host launcher code -- §6 is about device code
            header = "\n".join(lines[idx : idx + 3])
            bound = None
            if FOR_RE.match(line):
                m = FOR_STOP_RE.search(header)
                if not m:
                    continue
                stop = m.group(1)
                bm = REL_BOUND_RE.search(stop)
                bound = bm.group(1) if bm else stop
            elif WHILE_RE.match(line):
                m = WHILE_COND_RE.search(line)
                cond = m.group(1) if m else ""
                if not re.search(r"[<>]=?", cond):
                    continue  # no relational stop expression to classify
                bm = REL_BOUND_RE.search(cond)
                bound = bm.group(1) if bm else cond
            else:
                continue

            if bound_looks_constexpr(bound):
                continue
            if preceding_pragma_unroll1(lines, idx):
                continue
            findings.append(
                f"§6: {rel}:{idx + 1}: loop bound `{bound.strip()}` does not read as "
                f"compile-time and no `#pragma unroll 1` precedes it -- "
                f"{line.strip()!r}"
            )
    return findings


def check_no_torch_in_arm_tus() -> list[str]:
    findings = []
    if not INSTANTIATIONS.is_dir():
        return findings
    for path in INSTANTIATIONS.rglob("*"):
        if not path.is_file() or path.suffix not in DEVICE_SUFFIXES:
            continue
        rel = path.relative_to(ROOT)
        for idx, line in enumerate(path.read_text().splitlines(), start=1):
            if TORCH_INCLUDE_RE.search(line):
                findings.append(
                    f"belt to torch_include_in_arm_tu.yml: {rel}:{idx}: "
                    f"torch include in a per-arm TU -- {line.strip()!r}"
                )
    return findings


def check_no_sync_after_decode_prologue() -> list[str]:
    findings = []
    if not DECODE_CU.is_file():
        return findings
    rel = DECODE_CU.relative_to(ROOT)
    lines = DECODE_CU.read_text().splitlines()
    marker_idx = next((i for i, l in enumerate(lines) if WALK_MARKER_RE.search(l)), None)
    if marker_idx is None:
        findings.append(
            f"the decode prologue law: {rel}: could not locate the `---- THE WALK` prologue-end "
            f"marker this check keys on -- the file was restructured and this check "
            f"needs a new anchor."
        )
        return findings
    for idx in range(marker_idx + 1, len(lines)):
        if SYNC_RE.search(lines[idx]):
            findings.append(
                f"the decode prologue law (ledger 2026-08-27, decode.cu#per-cta-prologue): "
                f"{rel}:{idx + 1}: barrier after the prologue marker "
                f"(line {marker_idx + 1}) -- {lines[idx].strip()!r}"
            )
    return findings


#: The 2026-08-04 comments-to-docs mandate names "well
#: under 15%" as the line -- a named constant, not a number buried in an f-string,
#: so a future re-derivation of the threshold has exactly one place to change it.
COMMENT_DENSITY_THRESHOLD = 0.15

#: The mandate's own grain: FILE-HEADER and DECL-BLOCK comments are what stays;
#: granular prose belongs in docs/internals/ mirroring csrc/. A comment RUN this
#: long, found OUTSIDE the file's opening header block, is the shape the mandate
#: calls an antipattern -- a decl-block comment names what a declaration IS, it
#: does not walk a reader through a derivation.
MAX_COMMENT_BLOCK_LINES = 6

_CSRC = ROOT / "csrc"
_SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".h")


def _source_files():
    for suf in _SOURCE_SUFFIXES:
        for p in _CSRC.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            yield p


def _comment_density(lines: list[str]) -> tuple[int, int]:
    """`(comment_lines, total_lines)`, counting `//` line comments and `/* */`
    block comments (a comment line inside an unterminated block still counts,
    even with no leading `//` of its own)."""
    comment = 0
    in_block = False
    for line in lines:
        s = line.strip()
        if in_block:
            comment += 1
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("//"):
            comment += 1
        elif s.startswith("/*"):
            comment += 1
            if "*/" not in s:
                in_block = True
    return comment, len(lines)


def check_comment_density() -> list[str]:
    """Per-file comment-line fraction under `csrc/`, KERNEL_STANDARDS' 2026-08-04
    comments-to-docs mandate. REPORTED, not gated to zero: these are findings for
    Blake to rule on (the prose moves to `docs/internals/`), never silently
    fixed or silenced by this script -- same treatment every other check here
    gets (module docstring)."""
    findings = []
    for p in _source_files():
        lines = p.read_text().splitlines()
        if not lines:
            continue
        comment, total = _comment_density(lines)
        density = comment / total
        if density > COMMENT_DENSITY_THRESHOLD:
            findings.append(
                f"comment density (2026-08-04 comments-to-docs mandate, 'well "
                f"under {COMMENT_DENSITY_THRESHOLD:.0%}'): {p.relative_to(ROOT)}: "
                f"{comment}/{total} lines = {density:.0%} comment -- move the "
                f"prose to docs/internals/, decl-block/file-header comments only"
            )
    return findings


def check_no_long_comment_blocks() -> list[str]:
    """A `//`-comment run longer than `MAX_COMMENT_BLOCK_LINES`, found OUTSIDE
    the file's own opening header block (the copyright/module-docstring run
    starting at line 1), is a NEW prose block the comments-to-docs mandate
    exists to keep out -- reported the same way `check_comment_density` is,
    never auto-fixed."""
    findings = []
    for p in _source_files():
        lines = p.read_text().splitlines()
        n = len(lines)
        header_end = 0
        while header_end < n and (lines[header_end].strip().startswith("//")
                                   or lines[header_end].strip() == ""):
            header_end += 1
        i = header_end
        while i < n:
            if lines[i].strip().startswith("//"):
                start = i
                while i < n and lines[i].strip().startswith("//"):
                    i += 1
                block_len = i - start
                if block_len > MAX_COMMENT_BLOCK_LINES:
                    findings.append(
                        f"comment block (2026-08-04 comments-to-docs mandate, "
                        f"decl-block grain <= {MAX_COMMENT_BLOCK_LINES} lines): "
                        f"{p.relative_to(ROOT)}:{start + 1}-{i}: a {block_len}-line "
                        f"comment run outside the file header -- prose this long "
                        f"belongs in docs/internals/, not a new `//:` block here"
                    )
            else:
                i += 1
    return findings


def check_tests_tier_directories() -> list[str]:
    """docs/testing.md "there is no fourth tier": no directory directly under
    `tests/` other than `unit/`, `integration/` or `oracle/` (a campaign name --
    `tests/k31/` was one -- is exactly the shape of the violation this catches).
    Hidden directories (e.g. `__pycache__`, `.pytest_cache`) are not test tiers
    and are exempt."""
    findings = []
    if not TESTS_ROOT.is_dir():
        return findings
    for child in sorted(TESTS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name == "__pycache__":
            continue
        if child.name not in TESTS_ALLOWED_TIERS:
            findings.append(
                f"docs/testing.md 'there is no fourth tier': tests/{child.name}/ is a "
                f"directory under tests/ that is none of {sorted(TESTS_ALLOWED_TIERS)} -- "
                f"fold its tests into the tier each one answers, by question, not by "
                f"campaign name."
            )
    return findings


#: The `docs/internals/` mirror (2026-08-04 comments-to-docs mandate), ported
#: from `tests/unit/test_docs_mirror.py` (RULED 2026-08-29: a check that does
#: not execute the code under test is a lint, not a test -- docs/testing.md).
#: Deliberately grep-level, no markdown parser: a check a contributor cannot
#: read is a check they will route around.
_DOCS_SRC = ROOT / "csrc" / "rola"
_DOCS_MIRROR = ROOT / "docs" / "internals"
_DOCS_SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".h", ".hpp")

#: A doc reference as it appears in a source comment, e.g.
#: ``docs/internals/chunk/chunk_kernel.md#state-epilogue``. The anchor is optional: a
#: file-header block may point at a whole doc.
_DOCS_REF = re.compile(r"docs/internals/([A-Za-z0-9_/]+\.md)(?:#([A-Za-z0-9_-]+))?")

#: A hazard stub: ``//: HAZARD <slug> -- docs/internals/<doc>.md#<slug>``.
_DOCS_STUB = re.compile(r"//:\s*HAZARD\s+([A-Za-z0-9_-]+)\b")

#: An anchor definition in a mirror doc.
_DOCS_ANCHOR = re.compile(r'<a id="([A-Za-z0-9_-]+)"')

#: GENERATED SOURCES ARE DOCUMENTED BY THEIR GENERATOR, not one doc each -- see
#: `csrc/rola/src/instantiations/README.md`.
_DOCS_GENERATED_DIRS = ("instantiations",)


def _docs_sources() -> list[Path]:
    return sorted(
        p
        for p in _DOCS_SRC.rglob("*")
        if p.suffix in _DOCS_SOURCE_SUFFIXES
        and "third_party" not in p.parts
        and not set(p.parts) & set(_DOCS_GENERATED_DIRS)
    )


def _docs_mirror_for(source: Path) -> Path:
    """``csrc/rola/src/decode/decode.cuh`` -> ``docs/internals/decode/decode.md``.

    The mirror does not repeat the ``src/`` component: the sources it mirrors
    are all under it, so carrying it would be one directory of pure ceremony.
    """
    relative = source.relative_to(_DOCS_SRC)
    if relative.parts[0] == "src":
        relative = Path(*relative.parts[1:])
    return _DOCS_MIRROR / relative.with_suffix(".md")


def _docs_anchors(doc: Path) -> list[str]:
    return _DOCS_ANCHOR.findall(doc.read_text())


def check_generated_directories_are_documented_by_their_readme() -> list[str]:
    """Rule 5. The exemption from rule 1 costs a stricter check, not a weaker one.

    A generated directory that DOES NOT EXIST is not a violation: the generator's own
    `--check` (run below, and by `check_shard_partition`) is what says whether that is
    correct, and at an empty arm list it is. The rule is about a directory
    that exists without the README its exemption is paid for with.
    """
    import subprocess

    findings = []
    for name in _DOCS_GENERATED_DIRS:
        directory = _DOCS_SRC / "src" / name
        if not directory.is_dir():
            continue
        if not (directory / "README.md").is_file():
            findings.append(
                f"{directory.relative_to(ROOT)} is exempt from the per-file mirror rule "
                f"because it is generated; that exemption requires its README.md")
            continue
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "gen_shards.py"), "--check"],
            cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            findings.append(
                f"{directory.relative_to(ROOT)} contains something its generator does "
                f"not produce:\n{proc.stdout}{proc.stderr}")
    return findings


def check_every_source_file_has_a_mirror_doc() -> list[str]:
    """Rule 1. A `.cu` and its `.cuh` share one doc, since they are one component."""
    findings = []
    for source in _docs_sources():
        doc = _docs_mirror_for(source)
        if not doc.is_file():
            findings.append(
                f"{source.relative_to(ROOT)} has no mirror doc.\n"
                f"Expected {doc.relative_to(ROOT)}.\n"
                "Every source file carries its reasons in docs/internals/ at the same "
                "relative path; a .cu and its .cuh share one doc. See "
                "docs/internals/README.md."
            )
    return findings


def check_every_doc_reference_resolves() -> list[str]:
    """Rule 2. Every ``docs/internals/....md#anchor`` in the sources resolves to
    an ``<a id="anchor">`` the named doc actually defines."""
    broken = []
    for source in _docs_sources():
        text = source.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            for doc_name, anchor in _DOCS_REF.findall(line):
                doc = _DOCS_MIRROR / doc_name
                where = f"{source.relative_to(ROOT)}:{lineno}"
                if not doc.is_file():
                    broken.append(f"{where} -> {doc_name} (no such doc)")
                elif anchor and anchor not in _docs_anchors(doc):
                    broken.append(f"{where} -> {doc_name}#{anchor} (no such anchor)")
    return [f"unresolved doc reference: {b}" for b in broken]


def check_hazard_stubs_cite_their_own_slug() -> list[str]:
    """Rule 3. A stub's slug IS its anchor id, so the two cannot drift apart."""
    bad = []
    for source in _docs_sources():
        for lineno, line in enumerate(source.read_text().splitlines(), 1):
            stub = _DOCS_STUB.search(line)
            if stub is None:
                continue
            slug = stub.group(1)
            refs = _DOCS_REF.findall(line)
            where = f"{source.relative_to(ROOT)}:{lineno}"
            if not refs:
                bad.append(f"{where}: HAZARD {slug} cites no docs/internals anchor")
                continue
            anchors = [anchor for _, anchor in refs if anchor]
            if slug not in anchors:
                bad.append(f"{where}: HAZARD {slug} cites {anchors or refs} instead")
    return [f"hazard stub does not cite its own slug: {b}" for b in bad]


def check_no_doc_defines_an_anchor_twice() -> list[str]:
    """Rule 4. A citation to a duplicated anchor is ambiguous."""
    findings = []
    for doc in sorted(_DOCS_MIRROR.rglob("*.md")):
        anchors = _docs_anchors(doc)
        duplicates = sorted({a for a in anchors if anchors.count(a) > 1})
        if duplicates:
            findings.append(
                f"{doc.relative_to(ROOT)} defines these anchors more than once: "
                f"{duplicates}. A citation to a duplicated anchor is ambiguous."
            )
    return findings


#: AN IDENTIFIER, as a mirror page writes one in code font: a C++ or Python name, with
#: optional `::`/`.` qualification and an optional call's parentheses. A page's other
#: backticked spans -- a path, a flag, a number, a phrase, a type expression -- are not
#: identifiers and are not checked, so what this rule reads is exactly the set of names a
#: reader would go looking for in the source.
_MIRROR_IDENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*"
                           r"(?:\(\))?$")

#: The names that are not the source's to carry: language and library vocabulary a page
#: legitimately writes in code font while describing what the code does. This is a
#: VOCABULARY, not an allowlist of sites: it says what kind of token is not an
#: identifier of this repository, never that a particular page may be stale.
#: NOT AN IDENTIFIER OF THIS REPOSITORY, by its SHAPE: a CUDA runtime call, a PTX or
#: intrinsic name, an architecture, an ncu counter, or a math symbol with a subscript
#: (`z_k`, `tau_m`, `R_0`) -- the vocabulary a page writes in code font while describing
#: what the code does. These are structural forms, not an allowlist of sites: none of
#: them says that a particular page may be stale.
_MIRROR_FOREIGN = re.compile(
    r"^(cuda[A-Z]\w*|CU\w+|__\w+|sm_\d+|tcgen05\w*|wgmma\w*|cp_async\w*"
    r"|\w*__\w+\.\w+|[A-Za-z]{1,4}_[A-Za-z0-9]{1,3})$")

_MIRROR_NOT_OURS = frozenset((
    "int", "float", "double", "bool", "void", "char", "unsigned", "long", "short",
    "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "nullptr", "true", "false", "const", "constexpr", "static",
    "inline", "template", "typename", "class", "struct", "enum", "union", "namespace",
    "using", "return", "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "new", "delete", "this", "auto", "public", "private", "protected",
    "__global__", "__device__", "__host__", "__shared__", "__restrict__", "__syncthreads",
    "__syncwarp", "__ldg", "__trap", "threadIdx", "blockIdx", "blockDim", "gridDim",
    "warpSize", "torch", "tensor", "Tensor", "None", "True", "False", "self", "def",
    "import", "from", "lambda", "dict", "list", "tuple", "set", "str", "bytes", "float32",
    "float64", "bfloat16", "int32", "uint8", "cuda", "cpu", "num_warps", "num_stages",
    "static_for", "BOOL_SWITCH", "math_pipe_throttle", "long_scoreboard", "_RS", "_SS",
    "ldmatrix", "stmatrix", "mma", "wgmma"
))


def _repo_identifier_text() -> str:
    """Every first-party source, test, tool and build file, as one blob.

    A mirror page legitimately names identifiers that live in OTHER files: the entry its
    header declares, the tool that writes its generated input, the lint that enforces its
    law. What no page may name is an identifier that exists NOWHERE -- that is the name
    that left with its mechanism.
    """
    global _REPO_TEXT
    if _REPO_TEXT is None:
        chunks = []
        for root, suffixes in ((_DOCS_SRC, _DOCS_SOURCE_SUFFIXES),
                               (ROOT / "rola", (".py",)), (ROOT / "tools", (".py", ".sh")),
                               (ROOT / "tests", (".py",)), (ROOT / "benchmarks", (".py", ".json"))):
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if (path.suffix in suffixes and "third_party" not in path.parts
                        and "__pycache__" not in path.parts):
                    chunks.append(path.read_text(errors="ignore"))
        chunks.append((ROOT / "setup.py").read_text())
        _REPO_TEXT = "\n".join(chunks)
    return _REPO_TEXT


_REPO_TEXT: str | None = None


def _mirror_pairs() -> list[tuple[Path, list[Path]]]:
    """``(mirror page, the sources it mirrors)`` -- the inverse of `_docs_mirror_for`."""
    pages: dict[Path, list[Path]] = {}
    for source in _docs_sources():
        pages.setdefault(_docs_mirror_for(source), []).append(source)
    return sorted((doc, sources) for doc, sources in pages.items() if doc.is_file())


#: A DECLARED ABSENCE: `<!-- ABSENT: name other_name -->`. A page may name what is GONE --
#: an entry that was unbound, a mechanism that retired -- and the absence is often the
#: fact worth keeping. What it may not do is name it by accident, so the names are
#: declared on one line beside the prose that explains them, and the declaration is
#: checked in BOTH directions: a declared-absent name that the tree DOES carry is a
#: finding too, because the passage is then telling a reader the opposite of the truth.
_MIRROR_ABSENT = re.compile(r"<!--\s*ABSENT:\s*([^>]*?)\s*-->")


def check_mirror_pages_name_only_live_identifiers() -> list[str]:
    """Every identifier a mirror page names in code font EXISTS in its source file.

    A mirror page can outlive the mechanism it documents: the branch that replaced a
    fan-in ring was forfeited, the ring was tombstoned, and the page went on describing
    it in full. Nothing in the mirror contract noticed, because the page still existed,
    still resolved its anchors and still sat at the right path -- every check was about
    the page's SHAPE and none about whether its subject was still there.

    The check is textual and deliberately weak in one direction only: a name the source
    does not contain ANYWHERE is a finding, and a name it does contain is accepted
    wherever it appears. That cannot catch a page describing a live name's dead
    BEHAVIOUR, but it catches every name that left with its mechanism, which is the
    failure that was actually paid for.
    """
    findings = []
    for doc, sources in _mirror_pairs():
        text = "\n".join(source.read_text() for source in sources)
        page = doc.read_text()
        absent = {name for line in _MIRROR_ABSENT.findall(page) for name in line.split()}
        for name in sorted(absent):
            if re.search(rf"\b{re.escape(name)}\b", _repo_identifier_text()):
                findings.append(
                    f"{doc.relative_to(ROOT)}: declares `{name}` ABSENT, and the tree "
                    f"carries it. The passage tells a reader the opposite of the truth: "
                    f"drop the name from the declaration, or say what it is now.")
        seen: set[str] = set()
        for lineno, line in enumerate(page.splitlines(), 1):
            if line.lstrip().startswith(("```", "    ")):
                continue
            for span in re.findall(r"`([^`\n]{1,80})`", line):
                match = _MIRROR_IDENT.match(span)
                if not match:
                    continue
                name = match.group(1)
                if name in _MIRROR_NOT_OURS or name in seen or len(name) < 3:
                    continue
                if name in absent:
                    continue
                if _MIRROR_FOREIGN.match(span) or _MIRROR_FOREIGN.match(name):
                    continue
                if re.search(rf"\b{re.escape(name)}\b", text):
                    continue
                if re.search(rf"\b{re.escape(name)}\b", _repo_identifier_text()):
                    continue
                seen.add(name)
                findings.append(
                    f"{doc.relative_to(ROOT)}:{lineno}: names `{span}` in code font, and "
                    f"{', '.join(str(s.relative_to(ROOT)) for s in sources)} contains no "
                    f"`{name}`. A mirror page that outlives its mechanism is a page a "
                    f"reader trusts and the code no longer honours: delete the passage, "
                    f"or say in prose that the mechanism is gone and where it went "
                    f"(docs/internals/DELETIONS.md).")
    return findings


def check_docs_mirror() -> list[str]:
    """The whole `docs/internals/` mirror contract, ported from
    `tests/unit/test_docs_mirror.py` (deleted; RULED 2026-08-29)."""
    findings = []
    findings += check_generated_directories_are_documented_by_their_readme()
    findings += check_every_source_file_has_a_mirror_doc()
    findings += check_every_doc_reference_resolves()
    findings += check_hazard_stubs_cite_their_own_slug()
    findings += check_no_doc_defines_an_anchor_twice()
    findings += check_mirror_pages_name_only_live_identifiers()
    return findings


#: "The word `flock` leaves the process": -- every GPU entry point now takes
#: `tools/gpu_lock.py`'s `gpu_lock()` ITSELF, so no committed doc/skill/script should
#: ever again tell a reader to wrap a command in an external `flock` on the GPU lock
#: path (that is exactly the self-deadlock shape). A COMMAND LINE is what this
#: catches -- a line that, once backticks/leading whitespace are stripped, STARTS
#: with the literal token `flock` naming the GPU lock path; a PROSE mention (a
#: warning against doing this, or a general statement about `flock`'s per-filesystem
#: semantics) is not on a line whose own first token is the command, so it is not a
#: false positive here in practice -- see the exemptions below for the two places
#: that would still trip a purely first-token check.
FLOCK_GPU_LOCK_RE = re.compile(r"^\s*`{0,3}\s*flock\s+\S*rola_gpu")

#: docs/KERNEL_STANDARDS.md carries two legitimate mentions this check would
#: otherwise flag: an ad hoc `cuda-gdb` debugging recipe (a human wrapping an
#: interactive THIRD-PARTY binary that has no `gpu_lock()` of its own to call --
#: outside "every GPU entry point" this rule is about) and a general statement of
#: `flock`'s per-filesystem semantics (still true, unrelated to who calls it). It
#: is the constitution, edited only by ruling, so it is exempted by name rather than
#: taught a narrower regex.
FLOCK_CHECK_EXEMPT_FILES = {ROOT / "docs" / "KERNEL_STANDARDS.md"}

#: "docs, skills, scripts" (the ruling's own words): committed prose and recipes an
#: agent or contributor reads and might copy-paste, not generated/vendored trees.
FLOCK_CHECK_ROOTS = (
    ROOT / "docs",
    ROOT / ".claude" / "skills",
    ROOT / "benchmarks",
    ROOT / "tools",
    ROOT / "CLAUDE.md",
    ROOT / "README.md",
)
FLOCK_CHECK_SUFFIXES = (".md", ".py", ".sh")


def _flock_check_files():
    for root in FLOCK_CHECK_ROOTS:
        if root.is_file():
            yield root
            continue
        if not root.is_dir():
            continue
        for suf in FLOCK_CHECK_SUFFIXES:
            for p in root.rglob(f"*{suf}"):
                if "third_party" in p.parts or "docs/internals" in str(p.relative_to(ROOT)):
                    continue
                yield p


#: A CAMPAIGN WORK CODE: a stage or card identifier -- `K31`, `P80b`, `G5`, `S5c`, and
#: the `D2-b` form a stage takes when it splits. Work codes are EXTERNAL: they name
#: entries in a journal this repository does not carry, so a reader of this tree cannot
#: resolve one, and a reference to one ages into a pointer at nothing.
#:
#: THE STANDARDS-SECTION FORM IS THE EXCEPTION AND IT IS SPELLED WITH ITS SECTION SIGN:
#: `§R13`, `§16`. That is a citation of a document in this tree, at a stable anchor, and
#: the sign is what makes it one rather than a bare pair of characters that happens to
#: look like a rule number.
_WORK_CODE_RE = re.compile(
    r"(?<![\w§#\-.])([A-Z]{1,3}\d{1,3}(?:[a-z]\b|-[a-z]\b|\b))(?!\.\d)")

#: NOT WORK CODES, by shape: a dtype or width (`F32`, `S8`, `BF16`), an architecture
#: (`SM86`), a CUDA/PTX name. A vocabulary, never a list of sites.
_NOT_A_WORK_CODE = re.compile(
    r"^("
    r"F16|F32|F64|S8|S16|S32|S64|U8|U16|U32|U64|BF16|TF32|FP8|FP16|FP32|FP64|INT4|INT8"
    r"|SM\d+|CC\d+"                     # an architecture
    r"|L1|L2|L1TEX"                      # a cache level
    r"|A100|A30|A40|A16|A10|A2|H100|H200|V100|L4|L40|T4|RTX\d+|WSL2|CUDA\d+|PTX\d+"   # a machine or a toolkit
    r"|[A-Z]{1,4}\d{3,4}"                # a linter's own code: E402, BLE001, PLC0415
    r"|R\d{1,3}|UR\d{1,2}|P\d"           # a SASS register or predicate
    r"|Q\d"                             # a quartile
    r"|SXM\d|PCIE\d|HBM\d|GDDR\d|DDR\d"  # a package or memory generation
    r"|BC\d+|BT\d+|DV\d+|BH\d+"       # a shape, written into a cell id
    r"|FA\d|CUTLASS\d|TK\d"           # another project's version
    r")$")

#: TWO BLIND SPOTS, STATED. A hyphen-digit form (`R-6`, `D-1`) is not policed, because
#: `D-1` is arithmetic and `R-6` is a ruling number and no rule can tell them apart. And
#: `R<n>` is a SASS REGISTER -- the ratification tool quotes disassembly by the line --
#: so a bare rung number spelled that way escapes; the standards form is `§R<n>` and is
#: unambiguous. The sweep that turned this check on removed both classes where it found
#: them; a new one is caught by review, not here.
_WORK_CODE_ROOTS = ("csrc", "rola", "docs", "tests", "benchmarks", "tools")
_WORK_CODE_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".md", ".json",
                       ".yaml", ".yml", ".sh", ".toml")


def _work_code_files():
    for name in _WORK_CODE_ROOTS:
        root = ROOT / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            #: a ratchet's baseline quotes the findings it holds (tools/lint/ratchet.py), so it is not prose
            if (path.suffix in _WORK_CODE_SUFFIXES and "third_party" not in path.parts
                    and "__pycache__" not in path.parts and path.parent != ROOT / "tools" / "lint" / "baselines"):
                yield path


#: WHERE A CITATION CAN LIVE. In a source file the rule reads COMMENTS and STRING
#: LITERALS and nothing else: a name the code chose (`A15`, the alpha-1.5 template
#: argument) is subject to the naming rules, not to this one, and only text a human
#: wrote to point somewhere can point outside the tree. Markdown and data files are read
#: whole -- every line of them is prose.
_COMMENT_OR_STRING = re.compile(r"//(.*)$|/\*(.*?)\*/|#(.*)$|\"([^\"\n]*)\"|'([^'\n]*)'")


def _citable_text(path: Path, line: str, in_docstring: bool = False) -> str:
    if path.suffix in (".md", ".json", ".yaml", ".yml", ".toml") or in_docstring:
        return line
    return " ".join(g or "" for m in _COMMENT_OR_STRING.finditer(line) for g in m.groups())


TRIPLE = ('"' * 3, "'" * 3)


def _docstring_lines(path: Path, text: str) -> set[int]:
    """The line numbers inside a triple-quoted string.

    A DOCSTRING IS THIS LANGUAGE'S COMMENT, and it is where the module headers, the
    reasons and therefore the citations actually live. A line-oriented reader that only
    knew `#` and a single-line literal would miss every one of them, which is a gap in
    the medium and not a distinction anybody meant.
    """
    if path.suffix != ".py":
        return set()
    inside = False
    lines: set[int] = set()
    for number, line in enumerate(text.splitlines(), 1):
        marks = sum(line.count(q) for q in TRIPLE)
        if inside:
            lines.add(number)
        if marks % 2:
            inside = not inside
            lines.add(number)
    return lines


def _defined_labels(path: Path, text: str) -> set[str]:
    """The labels a MARKDOWN page defines for itself, in a heading or a bold lead-in.

    A page that writes `## M2 — NON-VACUITY` and then says `M2` afterwards is using its
    own vocabulary, and the reader has the definition on the page. That is the same
    distinction a source file's identifiers make, one medium over.
    """
    if path.suffix != ".md":
        return set()
    labels = re.findall(r"^#+\s+([A-Z]{1,3}\d{1,3}[a-z]?)\b", text, re.M)
    labels += re.findall(r"^\*\*([A-Z]{1,3}\d{1,3}[a-z]?)\b", text, re.M)
    labels += re.findall(r"^\|\s*`?([A-Z]{1,3}\d{1,3}[a-z]?)\b[^|]*\|", text, re.M)
    return set(labels)


def _code_text(path: Path, text: str, docstrings: set[int] | None = None) -> str:
    """The file's CODE, with comments and string literals removed.

    A LABEL THE FILE ITSELF DEFINES IS NOT A CITATION. A test file whose gates are named
    `test_teeth_A1_...` writes `GATE A1` in its own assertion messages, and a reader has
    the definition in front of them; the same two characters in a comment that points at
    a campaign stage have nothing behind them. The difference is whether the name exists
    in this file's code, so that is what is checked.
    """
    if path.suffix in (".md", ".json", ".yaml", ".yml", ".toml"):
        return ""
    docstrings = docstrings or set()
    return "\n".join("" if number in docstrings else _COMMENT_OR_STRING.sub("", line)
                     for number, line in enumerate(text.splitlines(), 1))


def check_no_campaign_work_codes() -> list[str]:
    """No file in this tree names a campaign or stage identifier.

    A work code is a pointer into a journal that lives outside this repository. A reader
    ten years from now has the tree and nothing else, so `K50's axis law` tells them
    only that a rule exists and that its reason is unreachable, while `the axis law
    (docs/ARCHITECTURE.md)` tells them where to read it. The rule is therefore not about
    tidiness: it is what forces a reason into the tree at the moment somebody is still
    able to write it down.

    The standards sections are the one citable short form and they carry their section
    sign (`§R13`); everything else becomes prose, a path, or an anchor.
    """
    findings = []
    for path in _work_code_files():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        docstrings = _docstring_lines(path, text)
        code_text = _code_text(path, text, docstrings)
        defined = _defined_labels(path, text)
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in _WORK_CODE_RE.finditer(
                    _citable_text(path, line, lineno in docstrings)):
                code = match.group(1)
                if _NOT_A_WORK_CODE.match(code):
                    continue
                if re.search(re.escape(code), code_text, re.IGNORECASE):
                    continue
                if code in defined:
                    continue
                findings.append(
                    f"{path.relative_to(ROOT)}:{lineno}: names the work code {code!r}. "
                    f"Work codes are external -- a reader of this tree cannot resolve "
                    f"one. Say the thing itself and cite where it is written down (a "
                    f"docs/ path, an anchor, or a standards section as §R<n>).")
    return findings


def check_no_bare_flock_on_gpu_lock() -> list[str]:
    findings = []
    for path in _flock_check_files():
        if path in FLOCK_CHECK_EXEMPT_FILES:
            continue
        for idx, line in enumerate(path.read_text().splitlines(), start=1):
            if FLOCK_GPU_LOCK_RE.match(line):
                findings.append(
                    f"K46 'the word flock leaves the process': {path.relative_to(ROOT)}:{idx}: "
                    f"a command line wraps the GPU lock in an external `flock` -- every GPU "
                    f"entry point takes `gpu_lock()` itself now (tools/gpu_lock.py); invoke "
                    f"BARE -- {line.strip()!r}"
                )
    return findings


#: K46 item 2 (Blake, 2026-08-29 "lints in test clothing"): a check that reads
#: files/manifests/versions and asserts consistency is a LINT, not a test -- moved
#: here, test file DELETED, from: test_generated_docs.py, test_shard_partition.py,
#: test_arch_coverage.py, test_vendored_pin.py, test_entmax_pin.py
#: (test_docs_mirror.py already moved, S5b -- `check_docs_mirror` above).
_TOOLS_DIR = ROOT / "tools"


def _read_generated_block(text: str, marker: str) -> str | None:
    begin, end = f"<!-- BEGIN GENERATED: {marker} -->", f"<!-- END GENERATED: {marker} -->"
    if begin not in text or end not in text:
        return None
    return text.split(begin, 1)[1].split(end, 1)[0].strip("\n")


def check_supported_table() -> list[str]:
    """Ported from tests/unit/test_generated_docs.py (deleted). `tools/supported.py`
    derives README's `tools/supported.py` block from `arch_caps.cuh` (tabulated) and
    `tools/manifests/` (ratified) -- FILES ONLY, no import of `rola` -- so this checks
    that ONE block matches the generator, that every tabulated arch is listed and
    correctly marked ratified/unratified, and (non-vacuity) that the comparison is
    not vacuous: it fires on a hand-tampered copy.

    A LINT reads files; it never imports the library under test (ITEM 0, 2026-08-29
    -- `tools/supported.py`'s SECOND block, `rola.routing.activations.capability_table`,
    executes real registry code to produce its content, so ITS drift check is a TEST
    (`tests/unit/test_capability_table_docs.py`), not this lint -- a check that
    executes the code under test is a test, not a lint (docs/testing.md). Importing
    `rola` here crashed this hook under pre-commit's isolated environment, which has
    no `rola` installed; this check must be a function of `tools/supported.py`, the
    README, `arch_caps.cuh` and `tools/manifests/` alone."""
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    import supported

    marker = "tools/supported.py"
    findings = []

    def _drift(readme_text: str) -> str | None:
        committed = _read_generated_block(readme_text, marker)
        if committed is None:
            return f"README.md has no generated block for {marker}"
        if committed != supported.table():
            return (f"README.md's {marker} block drifted from tools/supported.py's "
                    f"generator (regenerate with `python tools/supported.py --write`)")
        return None

    readme_text = supported.README.read_text()
    drift = _drift(readme_text)
    if drift is not None:
        findings.append(drift)

    table = supported.table()
    ratified = supported.ratified()
    for arch in supported.tabulated_archs():
        arch_marker = f"`{arch}`"
        if arch_marker not in table:
            findings.append(f"{arch} is tabulated in the kernel but absent from the "
                            f"README support table")
            continue
        row = next(line for line in table.splitlines() if arch_marker in line)
        if arch in ratified:
            if "ratified |" not in row or "not ratified" in row:
                findings.append(f"{arch} is ratified but the README row does not say "
                                f"so: {row}")
        elif "**not ratified — will refuse to load**" not in row:
            findings.append(f"{arch} has no manifest but the README row does not say "
                            f"so: {row}")

    tampered = readme_text.replace(
        "**not ratified — will refuse to load**", "ratified", 1)
    if tampered == readme_text:
        findings.append("check_supported_table non-vacuity probe found nothing "
                        "to tamper -- the README carries no 'not ratified' row to "
                        "flip, so the drift check above was not exercised")
    elif _drift(tampered) is None:
        findings.append("check_supported_table non-vacuity FAILED: a hand-tampered "
                        "README (a 'not ratified' row flipped to 'ratified') did not "
                        "register as drift -- the comparison is vacuous")
    return findings


def _repo_build_config() -> dict | None:
    """THIS TREE's `rola_<toolchain>/_build_config.py`, loaded BY PATH, or None if unbuilt.

    Never `import rola_cu13._build_config`. This module runs as a script, so `sys.path[0]`
    is `tools/lint/` and not the repository root: a plain import resolves `rola` out of
    whatever venv happens to be active -- on a box with a shared base venv, that is a
    DIFFERENT worktree's build, and the two checks below then verdict on a binary this
    repository never produced. (The extension trap's 14th form; the same rule as
    `assert_fresh_binary`: a gate reads the artifact it is gating.)
    """
    import importlib.util

    found = sorted(ROOT.glob("rola_*/_build_config.py"))
    if not found:
        return None
    path = found[0]
    spec = importlib.util.spec_from_file_location("_rola_repo_build_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUILD_CONFIG


def check_shard_partition() -> list[str]:
    """Ported from tests/unit/test_shard_partition.py (deleted). The generated
    instantiation tree is a function of `tools/gen_shards.py`'s arm list alone, and
    the manifest on disk is the ratification the CURRENTLY BUILT binary was gated
    against.

    C0 DELETED THE TWO FAMILIES THAT HAD GENERATED UNITS (card C): the carry arm list
    is empty and the stats passes' shard is gone, so the per-shard membership-hash
    half of this check has nothing to read and went with them. `gen_shards --check`
    still runs and still refuses a directory that disagrees with the arm list -- which
    at zero rows means a directory that is not empty.

    THE FALSE-FAIL THIS FIXES: the retired test recomputed the manifest
    digest over EVERY `tools/manifests/sm_*.json` file found on disk, but
    `BUILD_CONFIG["manifest_sha256"]` is the digest of whatever arch SUBSET this
    build actually compiled (`ROLA_CUDA_ARCHS`) -- a single-arch iteration build
    would then compare a one-arch digest against an all-arches recomputation and
    fail on a build that did nothing wrong. Scoped here to
    `BUILD_CONFIG["archs"]` -- what the build declares -- instead of a directory
    glob.
    """
    import subprocess

    findings = []
    proc = subprocess.run([sys.executable, str(_TOOLS_DIR / "gen_shards.py"), "--check"],
                          cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        findings.append(f"tools/gen_shards.py --check failed:\n{proc.stdout}{proc.stderr}")

    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    BUILD_CONFIG = _repo_build_config()
    if BUILD_CONFIG is None:
        pass  # not built yet -- nothing to check the manifest against
    else:
        built_archs = BUILD_CONFIG.get("archs")
        if not built_archs:
            findings.append("BUILD_CONFIG carries no 'archs' list; cannot scope the "
                            "manifest digest check to what this build declares")
        else:
            arches = sorted(a.removeprefix("sm_") for a in built_archs)
            try:
                import ratify
                import toolchains
                name = toolchains.for_ptxas(BUILD_CONFIG["ptxas"]).name
                got = ratify.manifest_digest(ratify.load_manifest(arches, name))
            except Exception as ex:  # noqa: BLE001
                findings.append(f"could not recompute the manifest digest for the "
                                f"built arch subset {arches}: {ex}")
            else:
                if got != BUILD_CONFIG["manifest_sha256"]:
                    findings.append(
                        f"tools/manifests/{name}/ for {arches} (this build's own declared "
                        f"arch subset) digests to {got}, which is NOT the "
                        f"ratification the shipped binary was gated against "
                        f"({BUILD_CONFIG['manifest_sha256']}) -- every reader of "
                        f"tools/manifests/ then describes a product that was never "
                        f"built.")
    return findings


def check_arch_coverage() -> list[str]:
    """Ported from tests/unit/test_arch_coverage.py (deleted). A tabulated arch with
    no ratified manifest is the "will refuse to load" row, and the generated support
    table must say so -- a hole on one arch is invisible to every other gate, which
    all run on whichever arch the developer's own card happens to be. Device-free:
    reads `tools/manifests/` only.

    THE PER-TOPOLOGY HALF WENT WITH THE STATS PASSES (C0, card C). It asserted that
    every ratified arch's manifest carried a facts arm for every `(D, B)` class any
    other arch carried one for; there are no facts arms. It returns with the kernel,
    keyed on the rebuilt family's own arm list, not restored against an empty one.
    """
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    try:
        import ratify  # noqa: F401 -- the manifest reader the successor half uses
        import supported
    except ImportError as ex:
        return [f"check_arch_coverage: could not import tools/ratify.py or "
                f"tools/supported.py: {ex}"]

    import toolchains

    ratified = {f"sm_{arch}" for t in toolchains.records().values() for arch in t.archs()}
    per_arch = {arch for arch in supported.tabulated_archs() if arch in ratified}
    findings = []
    if not per_arch:
        return ["no tabulated arch has a ratified manifest; nothing was measured"]

    unratified = sorted(set(supported.tabulated_archs()) - per_arch)
    table = supported.table()
    for arch in unratified:
        marker = f"| `{arch}` |"
        if marker not in table:
            findings.append(f"{arch} is tabulated with no manifest but has no row "
                            f"in the README support table at all")
            continue
        tail = table.split(marker, 1)[1].split("\n", 1)[0]
        if "will refuse to load" not in tail:
            findings.append(f"{arch} is tabulated with no manifest, which is the "
                            f"'will refuse to load' row; the generated support table "
                            f"does not say so")
    return findings


def check_vendored_pin() -> list[str]:
    """Ported from tests/unit/test_vendored_pin.py (deleted). Closed-world rule 5:
    the vendored CUTLASS subtree matches `PIN.json`'s digest, the BSD-3-Clause
    redistribution obligations are discharged in-tree, and the extension and the
    ratification probe compile against the vendored headers with the SAME include
    path (a probe compiled against a different include set measures a different
    kernel than the one shipped)."""
    import hashlib
    import json

    cutlass_dir = ROOT / "csrc" / "third_party" / "cutlass"
    pin_path = cutlass_dir / "PIN.json"
    if not pin_path.is_file():
        return []  # nothing vendored to check in this checkout state
    pin = json.loads(pin_path.read_text())

    findings = []
    for key in ("upstream", "tag", "commit", "license", "tree_sha256", "file_count",
               "modifications"):
        if not pin.get(key):
            findings.append(f"{pin_path.relative_to(ROOT)} is missing `{key}`")
    if len(pin.get("commit", "")) != 40:
        findings.append("the pin must be a full commit sha, not a tag alone")
    if not str(pin.get("modifications", "")).startswith("none"):
        findings.append("a MODIFIED vendored subtree cannot be checked against the "
                        "upstream tag, which is the only thing that makes the digest "
                        "more than self-referential")

    def tree_sha256(root: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "PIN.json")
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest(), len(files)

    here, count = tree_sha256(cutlass_dir)
    if count != pin.get("file_count"):
        findings.append(f"vendored file count moved: {count} on disk, "
                        f"{pin.get('file_count')} pinned")
    if here != pin.get("tree_sha256"):
        findings.append("the vendored CUTLASS subtree does not match its pin -- a "
                        "vendored header is code ptxas assembles, so this is a "
                        "codegen change and not a packaging one")

    if "license_file" in pin:
        license_path = cutlass_dir / pin["license_file"]
        if license_path.is_file():
            license_text = license_path.read_text()
            if "Redistribution and use in source and binary forms" not in license_text:
                findings.append(f"{license_path.relative_to(ROOT)} does not carry the "
                                f"BSD-3-Clause redistribution text")
            if "NVIDIA CORPORATION" not in license_text:
                findings.append(f"{license_path.relative_to(ROOT)} does not name "
                                f"NVIDIA CORPORATION")
    notice = ROOT / "NOTICE"
    if notice.is_file():
        notice_text = notice.read_text()
        if pin.get("commit") and pin["commit"] not in notice_text:
            findings.append("NOTICE must name the exact vendored commit")
        if "BSD-3-Clause" not in notice_text:
            findings.append("NOTICE does not name BSD-3-Clause")

    setup_text = (ROOT / "setup.py").read_text()
    ratify_text = (ROOT / "tools" / "ratify.py").read_text()
    needle = '"csrc" / "third_party" / "cutlass"'
    if needle not in setup_text:
        findings.append("setup.py no longer passes the vendored include path")
    if needle not in ratify_text:
        findings.append("tools/ratify.py no longer passes the vendored include path; "
                        "the manifest would then be ratified against a compile the "
                        "extension does not perform")
    if "_gate_vendored" not in setup_text:
        findings.append("closed-world rule 5 gate (_gate_vendored) is gone from setup.py")

    BUILD_CONFIG = _repo_build_config()
    if BUILD_CONFIG is None:
        pass  # not built yet
    else:
        vendored = BUILD_CONFIG.get("vendored")
        if vendored is None:
            findings.append("this .so was built by a setup.py that did not stamp its "
                            "vendored codegen inputs; rebuild it")
        elif vendored.get("cutlass", {}).get("tree_sha256") != pin.get("tree_sha256"):
            findings.append("the built binary's stamped vendored CUTLASS digest does "
                            "not match the current PIN.json")
    return findings


#: K40 (KERNEL_STANDARDS.md §17: "the repository is SELF-CONTAINED"). A tracked file
#: pointing at an ad-hoc place -- another user's home directory, the coordinator's
#: out-of-repo campaign journal, a specific worktree checkout, a session's scratch
#: directory -- cannot be resolved by anyone who clones this repo. NO ALLOWLIST: found
#: 3,927 such paths in the perf ledger alone plus a dozen doc/skill/tool citations,
#: none of them exempt. The four patterns below necessarily SPELL what they catch, so
#: this module's OWN source is excluded from the scan (`_PORTABILITY_LINT_SELF`) --
#: every other tracked file, this one included in spirit, is held to the same text a
#: reader sees when the check fires on someone else's line.
_PORTABILITY_LINT_SELF = Path(__file__).resolve()
HOME_PATH_RE = re.compile("/" + "home/" + r"[\w.\-]+(?:/[\w.\-]+)*")
OUT_OF_REPO_JOURNAL_RE = re.compile("rola" + "-scratch")
#: `/path/to/...` is this repo's OWN documented placeholder convention (docs/setup.md)
#: for "substitute your own location" -- it names nothing ad hoc, so that prefix
#: exempts what follows; a real absolute path under a home directory is already
#: caught by HOME_PATH_RE above.
WORKTREE_PATH_RE = re.compile(r"(?<!/path/to)/" + "worktrees/")
#: A real scratch-directory PATH has a slash touching the word on at least one side;
#: a bracketed placeholder token or plain prose mention has no adjacent slash and is
#: not a finding.
SCRATCH_DIR_PATH_RE = re.compile(r"(?:/scratch" + r"pad(?=/|\b)|\bscratch" + r"pad(?=/))")


#: Third-party/vendored trees and build outputs are not this repo's own tracked
#: authorship; a lockfile's machine-local ephemeral temp path is not an ad-hoc
#: AUTHORED location either, so both are out of this check's scope entirely
#: (scope = git-tracked files, via `git ls-files`, which already excludes both).
def _tracked_files() -> list[Path]:
    import subprocess

    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                        capture_output=True, check=True, text=True)
    return [ROOT / p for p in out.stdout.split("\0") if p]


def check_repo_portability() -> list[str]:
    """§17 lint: no tracked file points at a home-directory path, the coordinator's
    out-of-repo journal, a real worktree path, or a session's scratch-directory path.
    Every tracked file is read as text (the .jsonl ledgers included, UTF-8, same as
    every source file); one that fails to decode is skipped, not flagged."""
    findings = []
    for path in _tracked_files():
        if path.is_symlink() or not path.is_file() or path == _PORTABILITY_LINT_SELF:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("csrc/third_party/"):
            continue  # vendored, not this repo's own authorship (checked by check_vendored_pin)
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if HOME_PATH_RE.search(line):
                findings.append(f"§17 portability: {rel}:{lineno}: absolute home-"
                                f"directory path -- {line.strip()[:120]!r}")
            if OUT_OF_REPO_JOURNAL_RE.search(line):
                findings.append(f"§17 portability: {rel}:{lineno}: cites the "
                                f"coordinator's out-of-repo journal by path -- cite it "
                                f"BY ROLE instead -- {line.strip()[:120]!r}")
            if WORKTREE_PATH_RE.search(line):
                findings.append(f"§17 portability: {rel}:{lineno}: a real worktree path "
                                f"-- {line.strip()[:120]!r}")
            if SCRATCH_DIR_PATH_RE.search(line):
                findings.append(f"§17 portability: {rel}:{lineno}: a session scratch-"
                                f"directory path -- {line.strip()[:120]!r}")
    return findings


_SETTINGS_SHELL_RE = re.compile(r"\$\{?((?:ROLA_[A-Z0-9_]+)|CUDA_HOME|MAX_JOBS|MOLD_PATH|PYTORCH_NVCC)\b")
_MACHINE_PATH_RE = re.compile(r"/usr/local/cuda|/mnt/c/Windows")


def _python_env_reads(text: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """A Python file's environment reads -- `os.environ.get(X)`, `os.getenv(X)`, `os.environ[X]`, `X in os.environ` --
    with X resolved through a module-level string constant or an f-string's literal prefix; and its non-docstring string
    constants (for machine paths)."""
    import ast

    tree = ast.parse(text)
    consts = {t.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
              for t in node.targets if isinstance(t, ast.Name)}

    def name_of(arg: ast.AST) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return consts.get(arg.id)
        if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            return arg.values[0].value
        return None

    def is_environ(node: ast.AST) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "environ"

    reads = []
    for node in ast.walk(tree):
        arg = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            if (node.func.attr == "get" and is_environ(node.func.value)) or node.func.attr == "getenv":
                arg = node.args[0]
        elif isinstance(node, ast.Subscript) and is_environ(node.value) and isinstance(node.ctx, ast.Load):
            arg = node.slice
        elif isinstance(node, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops) \
                and any(is_environ(c) for c in node.comparators):
            arg = node.left
        name = name_of(arg) if arg is not None else None
        if name:
            reads.append((node.lineno, name))
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
    strings = [(node.lineno, node.value) for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings]
    return reads, strings


def check_settings_come_from_the_dev_config() -> list[str]:
    """§23 (1): every machine-dependent value is read through `tools/dev_config.py`. A Python environment read, or a
    shell expansion of a `ROLA_*` variable (or `MOLD_PATH`), that the loader does not list as a handoff or a build
    parameter is a finding, and so is a CUDA or Windows path spelled outside the loader in tools, benchmarks or
    setup.py."""
    sys.path.insert(0, str(ROOT / "tools"))
    import dev_config

    allowed = set(dev_config.HANDOFFS) | set(dev_config.BUILD_PARAMETERS)
    loader = ROOT / "tools" / "dev_config.py"
    findings = []
    for path in _tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if (path.suffix not in (".py", ".sh") or path in (loader, _PORTABILITY_LINT_SELF)
                or rel.startswith("csrc/third_party/") or not path.is_file()):
            continue
        text = path.read_text()
        if path.suffix == ".py":
            try:
                reads, strings = _python_env_reads(text)
            except SyntaxError:
                continue
            names = [(n, s) for n, s in reads if s not in allowed]
            paths = [(n, s) for n, s in strings if _MACHINE_PATH_RE.search(s)]
        else:
            code = [(n, line) for n, line in enumerate(text.splitlines(), 1) if not line.lstrip().startswith("#")]
            names = [(n, m.group(1)) for n, line in code for m in _SETTINGS_SHELL_RE.finditer(line)
                     if m.group(1) not in allowed]
            paths = [(n, line.strip()) for n, line in code if _MACHINE_PATH_RE.search(line)]
        findings += [f"§23 dev config: {rel}:{n}: reads setting {s!r} from the environment -- declare it in "
                     f"tools/dev_config.py SCHEMA and read it with dev_config.get" for n, s in names]
        if not rel.startswith("tests/"):
            findings += [f"§23 dev config: {rel}:{n}: machine path spelled in the tree -- {s[:80]!r}; read it from "
                         f"the dev config" for n, s in paths]
    return findings


def main() -> int:
    #: THESE THIRTEEN are wired as a BLOCKING pre-commit/CI gate
    #: (.pre-commit-config.yaml, .github/workflows/lint.yml) -- a nonzero exit
    #: here fails every commit and every CI run, repo-wide. `check_entmax_pin`
    #: is NOT here (S5c ITEM 0): it executes real code (`pin.gate()`,
    #: monkeypatched refusal shapes) rather than reading files, so it is
    #: restored to tests/unit/test_entmax_pin.py -- a lint reads files; it
    #: never imports the library under test.
    #:
    #: S5c (2026-08-29): `check_comment_density`/`check_no_long_comment_blocks`
    #: MOVE FROM ADVISORY TO GATING here. K39's reason for parking them advisory
    #: (csrc/ ~40 files over the mandate's 15% line, "not to be fixed by you")
    #: is resolved: S5b + S5c moved every remaining `//:` block and file header
    #: to docs/internals/, and every csrc file is under the threshold as of this
    #: commit -- a gate that is green on the tip it lands on is a gate a future
    #: contributor can actually satisfy, not a standing debt notice.
    gating_findings: list[str] = []
    gating_findings += check_unroll_discipline()
    gating_findings += check_no_torch_in_arm_tus()
    gating_findings += check_no_sync_after_decode_prologue()
    gating_findings += check_tests_tier_directories()
    gating_findings += check_docs_mirror()
    gating_findings += check_no_bare_flock_on_gpu_lock()
    gating_findings += check_supported_table()
    gating_findings += check_shard_partition()
    gating_findings += check_arch_coverage()
    gating_findings += check_vendored_pin()
    gating_findings += check_comment_density()
    gating_findings += check_no_long_comment_blocks()
    gating_findings += check_repo_portability()
    gating_findings += check_settings_come_from_the_dev_config()

    if not gating_findings:
        print("lint_standards: no gating findings")
        return 0

    print(f"lint_standards: {len(gating_findings)} gating finding(s)")
    for f in gating_findings:
        print(f"  {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
