#!/usr/bin/env python3
"""THE DRIFT GUARDS, gated by `tools/lint/ratchet.py drift_guards`.

Each law below was ratified with a stage that removed a mechanism, and each one is a
law precisely because the removed mechanism is CHEAP TO REINTRODUCE: a template axis
that "just needs one more parameter", a second kernel body for the case that does not
fit, a `float*` view of a page, an `if` on the depth in a host entry. A law that only
lives in a document is re-broken by the next person who has not read it, so each has a
mechanical check here, run on every commit and printed.

This file prints an INVENTORY and exits 0; the ratchet is the gate. A commit may not add a finding, and the findings
the tree carries are its baseline. A rule that fires on the shipped tree is not thereby wrong; it is the work.

EVERY RULE IS A HEURISTIC OVER TEXT, and says so in its own docstring: a finding is a
question for a reviewer, not a proof. What makes the set worth running is that all of
them are cheap and none of them can be satisfied by a comment.

Usage:

    python tools/lint/drift_guards.py                  # the inventory
    python tools/lint/drift_guards.py --rule <name>    # one rule
    python tools/lint/drift_guards.py --test-fixtures  # each rule fires and stays quiet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "rola"
PYPKG = ROOT / "rola"
TESTS = ROOT / "tests"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: THE DECLARED ARM KEY, read from the declaration rather than restated. A lint that
#: hard-codes the key it enforces is a second declaration of it.
SHIPPED_SET_JSON = ROOT / "tools" / "manifests" / "shipped_set.json"

#: THE GEOMETRY BLOCK -- the ONE place the addressing derivation may be instantiated.
GEOMETRY_BLOCK = ("common/geom.cuh", "common/geom.cu", "common/geom_api.cuh")

#: THE DERIVATION'S OWN NAMES. Instantiating one of these outside the geometry block
#: is a second derivation site whatever it is called.
#: THE KEY GOVERNS THE FAMILY THAT DECLARES IT, here as in `arm_axes`. These are the
#: CARRY addressing derivation's names; decode's own lattice fill is decode's, derived
#: from its own geometry, and folding the two into one site is an open question in
#: `docs/open-work.md` rather than a finding here.
DERIVATION_NAMES = ("derive_carry_geom", "sub_boxes", "ModeOrder", "StreamSpans")


def arm_key() -> tuple[str, ...]:
    if not SHIPPED_SET_JSON.exists():
        return ("D", "DV", "warps_per_cta")
    return tuple(json.loads(SHIPPED_SET_JSON.read_text())["key"])


def _walk(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Every source under `root`, WITH THE FIXTURES EXCLUDED.

    A lint that walks a tree containing its own fixtures reports its own deliberate
    violations as real ones -- checked by running each rule in normal mode and
    grepping its output for `fixtures/`.
    """
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix in suffixes
                  and FIXTURES not in p.parents
                  and "third_party" not in p.parts
                  and "__pycache__" not in p.parts)


def cuda_files() -> list[Path]:
    return _walk(CSRC, (".cu", ".cuh", ".cpp", ".h"))


def python_files(roots=(PYPKG, TESTS)) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        out += _walk(r, (".py",))
    return out


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# the rules

def rule_arm_axes(files=None) -> list[str]:
    """A kernel template's parameter list carries the declared key and nothing else.

    Heuristic: a `template <...>` immediately above a `__global__` declaration, whose
    integer parameters are read as axes. A parameter named outside the declared key is
    a compile-time axis the declaration does not admit -- which is how a three-field
    key becomes a thirteen-field one, one "it only needs one more" at a time.
    Non-integer parameters (types) and parameters spelled as file constants (a leading
    `k`) are not axes and are not reported.

    THE KEY GOVERNS THE FAMILY THAT DECLARES IT. `shipped_set.json` is the carry
    family's declaration, so this rule reads carry kernels: the producer's solves and
    the decode step key on their own axes, and whether those fold into this key is an
    open question in `docs/open-work.md`, not a finding here.
    """
    key = {k.lower() for k in arm_key()}
    family = re.compile(r"\bcarry\w*_kernel\b")
    findings = []
    for path in (cuda_files() if files is None else files):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\s*template\s*<(.+)>\s*$", line)
            if not m:
                continue
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "__global__" not in nxt or not family.search(nxt):
                continue
            for param in m.group(1).split(","):
                pm = re.match(r"\s*(?:int|bool|unsigned)\s+(\w+)\s*$", param)
                if pm is None:
                    continue
                name = pm.group(1)
                if name.startswith("k") or name.lower().rstrip("_") in key:
                    continue
                findings.append(
                    f"{_rel(path)}:{i + 1}: kernel template axis '{name}' is not one "
                    f"of the declared arm key {list(arm_key())}")
    return findings


def rule_state_float_view(files=None) -> list[str]:
    """State storage is never reached through a `float*`.

    A page is split bf16 planes, so a `float` view of one reinterprets two rows as one
    number and reads plausible garbage. Heuristic: a `float*` cast or declaration on a
    line naming a page, a plane or the state.
    """
    subject = r"(?:state|page|plane|pages|planes|backing|arena)\w*"
    #: THE POINTER ITSELF must be the storage: a `float*` DECLARED as a page, or a
    #: cast of a page expression. A line that merely mentions a page while taking a
    #: `float*` register scratch is not a view of it, and reporting it would train a
    #: reader to skim this rule's findings.
    named = re.compile(rf"\bfloat\s*\*\s*{subject}\b", re.IGNORECASE)
    cast = re.compile(rf"reinterpret_cast\s*<\s*(?:const\s+)?float\s*\*\s*>\s*"
                      rf"\(\s*{subject}", re.IGNORECASE)
    findings = []
    for path in (cuda_files() if files is None else files):
        for i, line in enumerate(path.read_text().splitlines()):
            if line.lstrip().startswith("//"):
                continue
            if named.search(line) or cast.search(line):
                findings.append(f"{_rel(path)}:{i + 1}: a float* view of state "
                                f"storage: {line.strip()[:90]}")
    return findings


def rule_bytes_equal(files=None) -> list[str]:
    """A STORAGE IDENTITY claim compares BYTES.

    `torch.equal` compares values after the dtype's own equality, so it calls two
    different bit patterns equal (a signed zero, a NaN payload, a plane that was
    rewritten with the same value by a different path). A claim about STORAGE -- that
    a page, a plane or the state is unchanged -- is a claim about bytes and is made
    with a byte comparison.
    """
    #: THE ARGUMENTS, not the line: a message that happens to say "backing" does not
    #: make an output comparison a storage claim.
    call = re.compile(r"torch\.equal\(([^)]*)\)")
    subject = re.compile(r"\b\w*(state|plane|storage|backing|arena)\w*\b", re.IGNORECASE)
    #: AN INTEGER IS ALREADY ITS OWN BYTES. `torch.equal` is exact on an integer dtype --
    #: there is no signed zero and no NaN payload for it to call equal -- so a page table,
    #: a bitmap, a word array or an id list compared with it IS a byte comparison. The rule
    #: is about a FLOAT view of storage, which is the one place the two differ.
    integral = re.compile(r"\b\w*(bits|table|mask|words|rows|ids|count|index|slots|atoms)\w*\b",
                          re.IGNORECASE)
    findings = []
    for path in (python_files() if files is None else files):
        for i, line in enumerate(path.read_text().splitlines()):
            if line.lstrip().startswith("#"):
                continue
            for args in call.findall(line):
                if subject.search(args) and not integral.search(args):
                    findings.append(f"{_rel(path)}:{i + 1}: a storage identity claim "
                                    f"made with torch.equal: {line.strip()[:90]}")
                    break
    return findings


def rule_one_derivation_site(files=None) -> list[str]:
    """The addressing derivation is instantiated in the geometry block, and there only.

    Heuristic: a CALL to one of the derivation's names from a file outside the block.
    Reading the block's OUTPUT is the normal case and is not a call, so a consumer
    that takes the derived struct by value never fires.
    """
    call = re.compile(r"\b(" + "|".join(DERIVATION_NAMES) + r")\s*[<(]")
    findings = []
    for path in (cuda_files() if files is None else files):
        rel = _rel(path)
        if any(rel.endswith(b) for b in GEOMETRY_BLOCK):
            continue
        for i, line in enumerate(path.read_text().splitlines()):
            if line.lstrip().startswith("//") or "#include" in line:
                continue
            m = call.search(line)
            if m:
                findings.append(f"{rel}:{i + 1}: '{m.group(1)}' is instantiated "
                                f"outside the geometry block ({', '.join(GEOMETRY_BLOCK)})")
    return findings


def rule_one_admission_law(files=None) -> list[str]:
    """One admission law, not one per call path.

    Prefill and decode admit a descriptor by the same check; a branch that asks WHICH
    CALLER is asking is a second law wearing the first one's name. Heuristic: a
    condition testing a decode/prefill flag inside a function whose name mentions
    admission, a descriptor or a check.
    """
    scope = re.compile(r"\b\w*(admit|admission|descriptor|check_shape|validate)\w*\s*\(",
                       re.IGNORECASE)
    branch = re.compile(r"\bif\s*\(.*\b(is_decode|decode_only|for_decode|is_prefill|"
                        r"prefill_only|kDecode|kPrefill)\b")
    findings = []
    for path in (cuda_files() if files is None else files):
        lines = path.read_text().splitlines()
        in_scope, brace = False, 0
        for i, line in enumerate(lines):
            if not in_scope and scope.search(line):
                in_scope, brace = True, line.count("{") - line.count("}")
                continue
            if in_scope:
                brace += line.count("{") - line.count("}")
                if branch.search(line):
                    findings.append(f"{_rel(path)}:{i + 1}: an admission branch on the "
                                    f"CALLER: {line.strip()[:90]}")
                if brace <= 0:
                    in_scope = False
    return findings


def rule_single_value_axis(files=None) -> list[str]:
    """A one-member enumeration is a constant, not a parameter.

    An axis with one legal value costs a field in every signature, a branch in every
    reader and a column in every table, and buys nothing until a second value exists.
    Heuristic: a module-level tuple/list/frozenset named `*_KINDS`, `*_MODES`,
    `*_STRATEGIES` or `*_FLAGS` with exactly one member.
    """
    decl = re.compile(r"^([A-Z_]*(?:KINDS|MODES|STRATEGIES|FLAGS))\s*"
                      r"(?::[^=]+)?=\s*[\(\[\{]([^)\]\}]*)[\)\]\}]")
    findings = []
    for path in (python_files((PYPKG,)) if files is None else files):
        for i, line in enumerate(path.read_text().splitlines()):
            m = decl.match(line)
            if m is None:
                continue
            members = [x for x in m.group(2).split(",") if x.strip()]
            if len(members) == 1:
                findings.append(f"{_rel(path)}:{i + 1}: {m.group(1)} has one member "
                                f"({members[0].strip()}); a one-value axis is a "
                                f"constant")
    return findings


def rule_second_arm_table(files=None) -> list[str]:
    """One enumeration of the arm set: the declaration.

    Heuristic: a module-level name matching `*_ARMS`/`ARM_*` bound to a literal
    container, outside the generator and the declaration's own readers. Two lists of
    one set is the drift shape this repository has already shipped once.
    """
    allowed = {"tools/gen_shards.py"}
    decl = re.compile(r"^(\w*ARMS?\w*)\s*(?::[^=]+)?=\s*[\(\[\{]")
    findings = []
    files = python_files((PYPKG, TESTS, ROOT / "tools", ROOT / "benchmarks")) \
        if files is None else files
    for path in files:
        rel = _rel(path)
        if rel in allowed:
            continue
        for i, line in enumerate(path.read_text().splitlines()):
            m = decl.match(line)
            if m and not line.rstrip().endswith(("()", "[]", "{}")):
                findings.append(f"{rel}:{i + 1}: '{m.group(1)}' is a literal arm table "
                                f"outside {sorted(allowed)[0]}")
    return findings


def rule_one_body_per_family(files=None) -> list[str]:
    """One body per op family.

    A second body for the case the first does not fit is two kernels to keep correct,
    two to measure and two to ratify, and the case it was written for is a point of
    the one body's own parameter space. Heuristic: two `__global__` kernels whose
    names differ only by an infix (`x_kernel` beside `x_<something>_kernel`).
    """
    #: A REVERSE PASS IS ITS OWN OP, not a second body of the forward's: it computes
    #: different quantities from different inputs and is measured on its own.
    reverse = re.compile(r"_(backward|reverse|bwd)_")
    names: dict[str, tuple[str, int]] = {}
    for path in (cuda_files() if files is None else files):
        for i, line in enumerate(path.read_text().splitlines()):
            m = re.search(r"__global__[^;{]*?\b(\w+_kernel)\s*\(", line)
            if m:
                names[m.group(1)] = (_rel(path), i + 1)
    findings = []
    for name, (path, line) in sorted(names.items()):
        head = name[:-len("_kernel")]
        for other in names:
            if other == name:
                continue
            base = other[:-len("_kernel")]
            if head.startswith(base + "_") and base and not reverse.search(name + "_"):
                findings.append(f"{path}:{line}: '{name}' is a second body beside "
                                f"'{other}' in the same family")
    return findings


def rule_device_switch_is_uniform(files=None) -> list[str]:
    """Every device branch on a geometry field is a `uniform_switch`.

    The four layers R9 asks of a structure branch -- a generated set with a count
    assert, a compile-time exhaustiveness assert, the seam's refusal reading the same
    set, a trapping default with a coverage sweep -- are the primitive's contract and
    are not re-implemented per site. Heuristic: a `switch` on a non-constant inside a
    `__device__`/`__global__` function.
    """
    findings = []
    for path in (cuda_files() if files is None else files):
        lines = path.read_text().splitlines()
        device = False
        for i, line in enumerate(lines):
            if "__device__" in line or "__global__" in line:
                device = True
            m = re.match(r"^\s*switch\s*\(\s*(\w+)\s*\)", line)
            if m and device:
                findings.append(f"{_rel(path)}:{i + 1}: a device switch on "
                                f"'{m.group(1)}' that is not a uniform_switch<Set>")
    return findings


def rule_host_dispatch_is_arm_switch(files=None) -> list[str]:
    """Every host dispatch on the arm key goes through `arm_switch<ArmSet>`.

    A host `if`/`switch` chain on a key field selects a body from a set it enumerates
    by hand, so it can name an arm the binary does not carry and miss one it does.
    Heuristic: a host `if` comparing a key field to a literal, in a file that does not
    mention `arm_switch`.
    """
    key = arm_key()
    test = re.compile(r"\bif\s*\(.*\b(" + "|".join(re.escape(k) for k in key) +
                      r"|dv|d_v)\s*==\s*\d+")
    findings = []
    for path in (cuda_files() if files is None else files):
        text = path.read_text()
        if "arm_switch" in text or "__device__" in text or "__global__" in text:
            continue
        for i, line in enumerate(text.splitlines()):
            if line.lstrip().startswith("//"):
                continue
            if test.search(line):
                findings.append(f"{_rel(path)}:{i + 1}: a host dispatch on the arm key "
                                f"outside arm_switch<ArmSet>: {line.strip()[:80]}")
    return findings


#: THE WORDS A COMMENT USES WHEN IT IS DESCRIBING A PAST STATE rather than the code.
_HISTORY_WORDS = re.compile(
    r"\b(used to|previously|no longer|formerly|superseded|replaced by|the old |"
    r"old form|it was |we used|deprecated|legacy|now gone|"
    r"had been|instead of the)\b", re.IGNORECASE)

#: A CAMPAIGN OR STAGE IDENTIFIER: ONE DEFINITION, the standards lint's, imported rather
#: than restated. Two regexes for one law is the drift this file exists to catch, one
#: level up: the rule that refuses work codes tree-wide and the rule that refuses them in
#: a comment are the same rule read at two grains, and they cannot be allowed to disagree
#: about what a work code is.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_standards import _NOT_A_WORK_CODE as _NOT_WORK_CODES  # noqa: E402
from lint_standards import _WORK_CODE_RE as _WORK_CODE  # noqa: E402


def rule_comments_describe_the_present(files=None) -> list[str]:
    """A comment in `csrc/` or `rola/` describes the code AS IT IS.

    A comment about what the code used to be, what it replaced, or which stage changed
    it, is background: it ages into a false statement the moment the next change lands
    (a mirror page outliving its mechanism is the same failure one level up), and a
    reader cannot tell a stale narration from a current one. Background belongs in
    `docs/`; the journal keeps the history.

    Two heuristics, reported separately in the message: a HISTORY PHRASE, and a work
    code (a campaign or stage identifier) standing alone in a comment.
    """
    findings = []
    files = (cuda_files() + python_files((PYPKG,))) if files is None else files
    for path in files:
        for i, line in enumerate(path.read_text().splitlines()):
            stripped = line.strip()
            comment = ""
            if stripped.startswith(("//", "#")):
                comment = stripped
            elif "//" in line and not stripped.startswith("#include"):
                comment = line.split("//", 1)[1]
            elif '"""' in line or "'''" in line:
                comment = line
            if not comment:
                continue
            m = _HISTORY_WORDS.search(comment)
            if m:
                findings.append(f"{_rel(path)}:{i + 1}: a comment describing a past "
                                f"state ({m.group(1).strip()!r}): {stripped[:80]}")
                continue
            for code in (m.group(1) for m in _WORK_CODE.finditer(comment)):
                if _NOT_WORK_CODES.match(code):
                    continue
                findings.append(f"{_rel(path)}:{i + 1}: a comment naming the work code "
                                f"{code!r}: {stripped[:80]}")
                break
    return findings


RULES = {
    "arm_axes": rule_arm_axes,
    "state_float_view": rule_state_float_view,
    "bytes_equal": rule_bytes_equal,
    "one_derivation_site": rule_one_derivation_site,
    "one_admission_law": rule_one_admission_law,
    "single_value_axis": rule_single_value_axis,
    "second_arm_table": rule_second_arm_table,
    "one_body_per_family": rule_one_body_per_family,
    "device_switch_is_uniform": rule_device_switch_is_uniform,
    "host_dispatch_is_arm_switch": rule_host_dispatch_is_arm_switch,
    "comments_describe_the_present": rule_comments_describe_the_present,
}

#: `{rule -> (the fixture that MUST fire, the fixture that must NOT)}`. Every rule
#: owns a pair: a rule that has never been shown to fire is a rule nobody knows is
#: connected, and one that has never been shown to stay quiet is a rule nobody can
#: afford to make gating.
FIXTURE_PAIRS = {
    "arm_axes": ("drift_bad.cuh", "drift_ok.cuh"),
    "state_float_view": ("drift_bad.cuh", "drift_ok.cuh"),
    "one_derivation_site": ("drift_bad.cuh", "drift_ok.cuh"),
    "one_admission_law": ("drift_bad.cuh", "drift_ok.cuh"),
    "one_body_per_family": ("drift_bad.cuh", "drift_ok.cuh"),
    "device_switch_is_uniform": ("drift_bad.cuh", "drift_ok.cuh"),
    "host_dispatch_is_arm_switch": ("drift_host_bad.cpp", "drift_host_ok.cpp"),
    "bytes_equal": ("drift_bad.py", "drift_ok.py"),
    "single_value_axis": ("drift_bad.py", "drift_ok.py"),
    "second_arm_table": ("drift_bad.py", "drift_ok.py"),
    "comments_describe_the_present": ("drift_bad.py", "drift_ok.py"),
}


def self_test() -> int:
    bad = []
    for name, (fires, quiet) in sorted(FIXTURE_PAIRS.items()):
        rule = RULES[name]
        hot = rule([FIXTURES / fires])
        cold = rule([FIXTURES / quiet])
        if not hot:
            bad.append(f"    {name}: did not fire on {fires}")
        if cold:
            bad.append(f"    {name}: fired on {quiet}: {cold}")
    #: AND THE FIXTURES MUST NOT REACH THE REAL SCAN. A lint that walks a tree holding
    #: its own deliberate violations reports them as findings; this has happened to
    #: two lints in this directory already, so it is checked rather than remembered.
    for name, rule in sorted(RULES.items()):
        leaked = [f for f in rule() if "lint/fixtures/" in f]
        if leaked:
            bad.append(f"    {name}: its own fixtures leaked into the real scan: "
                       f"{leaked[:2]}")
    if bad:
        print("SELF-TEST FAILED:")
        print("\n".join(bad))
        return 1
    print(f"drift_guards.py --test-fixtures: PASS ({len(FIXTURE_PAIRS)} rules fire on "
          f"their positive fixture and stay quiet on their negative; no rule leaks a "
          f"fixture into the real scan)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", action="append", choices=sorted(RULES),
                    help="run only this rule, repeatable")
    ap.add_argument("--test-fixtures", action="store_true",
                    help="prove each rule fires on its positive fixture and stays "
                         "quiet on its negative")
    a = ap.parse_args()
    if a.test_fixtures:
        return self_test()

    total = 0
    for name in (a.rule or sorted(RULES)):
        findings = RULES[name]()
        total += len(findings)
        print(f"drift_guards[{name}]: {len(findings)} finding(s)")
        for f in findings:
            print(f"  {f}")
    print(f"drift_guards: {total} finding(s) over "
          f"{len(a.rule or RULES)} rule(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
