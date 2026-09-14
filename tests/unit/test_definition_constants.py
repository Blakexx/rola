"""A constant that changes the FUNCTION has exactly one home, and this checks it.

The readout epsilon appeared four times as a bare literal -- in the oracle, the
fused consumer, decode and the then-public `rola_func` -- and the four defaults
merely happened to agree (audit finding F6). Four agreeing literals are
indistinguishable from four independent decisions until one of them is edited, at
which point the oracle and the kernel compute different functions and every Tier 1
gate reports the difference as a kernel bug.

Three sites remain: the API contract took the `eps` knob off the public
surface entirely, which is the same conclusion reached from the other end -- a
number that changes the FUNCTION is not a caller's argument. The grep rule below
is the one that still covers the whole package, including the entry point that no
longer has a parameter to check.

Two claims, and the second is the one with teeth:

1. **EVERY SITE READS THE SAME OBJECT** -- checked at runtime, through the
   signatures a caller actually gets.
2. **NO SITE COULD DRIFT BACK** -- checked statically. A future edit that
   re-inlines the literal fails here rather than passing silently, because the
   default's AST node must be the NAME, not a number.

No device, no build: this reads source and signatures.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from rola.ops.constants import READOUT_EPS

ROOT = Path(__file__).resolve().parents[2]

#: Every entry point that takes the readout epsilon, as (module path, function).
#: Adding a fifth is fine; adding one that does not appear here is what rule 2
#: below refuses.
EPS_SITES = (
    ("rola/ops/naive.py", "naive_rola"),
    ("rola/ops/decode.py", "derive_decode_geometry"),
)

#: The retired tiled consumer's row is dropped. Its replacement is NOT
#: another row here, because the chunk arm
#: takes no epsilon PARAMETER -- the executor passes the constant straight
#: into the launch. Rule 3 below is what covers it: a literal epsilon anywhere in
#: the package is refused wherever it appears, parameter or not.


def _function_defs(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _eps_default_node(path: Path, function: str):
    for node in _function_defs(path):
        if node.name != function:
            continue
        args = node.args
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if arg.arg == "eps":
                return default
        positional = args.posonlyargs + args.args
        defaults = args.defaults
        for arg, default in zip(positional[len(positional) - len(defaults):], defaults):
            if arg.arg == "eps":
                return default
    raise AssertionError(f"{path}::{function} has no `eps` parameter")


@pytest.mark.parametrize("relative,function", EPS_SITES)
def test_every_eps_default_is_the_named_constant_not_a_literal(relative, function):
    """Rule 2: the default is the NAME. A re-inlined literal fails right here."""
    node = _eps_default_node(ROOT / relative, function)
    assert isinstance(node, ast.Name), (
        f"{relative}::{function} defaults `eps` to "
        f"{ast.dump(node)} -- it must be the imported name `READOUT_EPS`, so the "
        f"definition has one home (audit F6)"
    )
    assert node.id == "READOUT_EPS"


def test_every_eps_default_is_the_same_value_at_runtime():
    """Rule 1, through the signature a caller actually binds against."""
    from rola.ops.decode import derive_decode_geometry
    from rola.ops.naive import naive_rola

    for fn in (naive_rola, derive_decode_geometry):
        default = inspect.signature(fn).parameters["eps"].default
        assert default == READOUT_EPS, f"{fn.__qualname__} binds eps={default}"


def test_no_module_under_rola_inlines_the_readout_epsilon():
    """The literal is gone from the package, not merely unused at the four sites.

    Deliberately grep-level over `rola/`: a fifth call site that spells the number
    instead of importing it is the drift this file exists to stop, and it would
    not be in `EPS_SITES` to be found by the AST rule.
    """
    literal = repr(READOUT_EPS)
    offenders = []
    for path in sorted((ROOT / "rola").rglob("*.py")):
        if path == ROOT / "rola" / "ops" / "constants.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if literal in line and "eps" in line.lower():
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "the readout epsilon is inlined again; import it from rola.ops.constants:\n"
        + "\n".join(offenders)
    )
