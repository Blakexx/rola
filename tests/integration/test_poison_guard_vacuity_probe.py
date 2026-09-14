"""NON-VACUITY self-gate for `tests/conftest.py`'s NaN-poison guard.

The coverage-closure audit found the guard gating on `tests/ops/` and
`tests/modules/` -- the FORK's layout, neither of which exists in this tree -- so
the uninitialized-read tripwire had NEVER run on any test. A path-gated fixture
can go silently dead again the same way (a tier rename, a conftest refactor), and
its death is invisible: every test still passes, just unprotected. So this file
asserts the guard ENGAGES, from inside one of the tiers it must cover.

The probe compiles a one-line `torch.empty` call with a code object whose frame
resolves into the `rola` package (filename + globals of a real rola module),
because that is the guard's own activation predicate (`_is_called_from_fla`):
plain test-frame calls are deliberately unguarded. If this fails, the guard is
dead tree-wide and every poison finding it would have produced is being missed.
"""

import torch

#: An ARBITRARY real `rola` module -- the probe needs a filename and globals
#: that resolve inside the package, and nothing about the claim depends on
#: which one. It pointed at `rola.ops.consumer` when `rola.routing.
#: sequence_parallel` was deleted, then at the chunk arm; today it points at
#: decode for the same reason and by the same precedent.
from rola.ops import decode as probe_mod


def test_the_nan_poison_guard_engages_in_this_tier():
    code = compile(
        "def _poison_probe(torch):\n    return torch.empty(8, dtype=torch.float32)",
        probe_mod.__file__, "exec")
    exec(code, probe_mod.__dict__)
    try:
        out = probe_mod.__dict__["_poison_probe"](torch)
    finally:
        probe_mod.__dict__.pop("_poison_probe", None)
    assert torch.isnan(out).all(), (
        "torch.empty called from a rola frame was NOT NaN-poisoned: the conftest "
        "guard is dead in tests/integration (it must cover tests/oracle/ and "
        "tests/integration/ -- see _POISON_TIERS)")
