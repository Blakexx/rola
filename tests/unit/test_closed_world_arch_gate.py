# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CLOSED-WORLD RUNTIME ARCH REFUSAL, checked where it can be checked.

``setup.py``'s rule R4 forbids *a launch on a device whose arch was never measured*.
Unlike the build-time rules it cannot be enforced at build time, because the device is
not present then; it is enforced in C++ by a pair of clauses, and this file gates that
the pair is intact.

**THE PAIR.** The compile-time half is ``csrc/rola/src/common/arch_caps.cuh``'s
``static_assert(tabulated(kArch))``, which fixes the capability row every arm's staging
depth, CTA target and shared-memory fit was DERIVED from. The runtime half is
``check_arch_table()``, which refuses -- on the first launch, per device -- an
architecture the table has no row for, or a row the driver disagrees with. Neither half
can substitute for the other: a derivation is only as good as the facts it was closed
over, and only a fact read off the DEVICE can establish that this is the device they
were read for.

**THE RUNTIME HALF HAS NO DEFINITION ON THIS LINE, AND THIS FILE IS RED FOR THAT ONE
REASON.** Its single definition lived in the stats pass's dispatch translation unit,
which the clean slate cleared (card ``development/queue/C_CLEAN_SLATE.md``); the
surviving families -- the liveness pass, decode, intra, entmax -- kept their entries and
lost the clause. The gate is stated here over EVERY declared entry surface rather than
over one family's, because the property is per entry and the next entry added must
inherit it. Making it green is a csrc change to the surviving families, and the day it
passes is the day the refusal is back.

**WHY THIS SHAPE OF TEST.** The refusal's subject is an architecture this machine is
not, so the refusal itself cannot be provoked on real hardware: an untabulated device is
exactly the thing that is unavailable. What CAN be checked is the plumbing -- that the
clause exists, that no entry which launches can be reached without it, and that on the
device that IS here the check passes and a launch follows.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "csrc" / "rola" / "src"
ARCH_CAPS = SRC / "common" / "arch_caps.cuh"

#: THE ENTRY LIST IS THE REGISTRATION TRANSLATION UNIT'S OWN. Every operator the extension
#: offers is an ``m.impl`` here and nowhere else, so a family cannot join the library
#: without appearing in this scan -- which a per-header glob could not promise, since a
#: family may declare its entries in its kernel header rather than in an ``*_api.cuh``.
REGISTRATION_TU = REPO / "csrc" / "rola" / "rola_api.cpp"

#: The device translation units an entry's definition may live in.
DEVICE_TUS = tuple(sorted(SRC.glob("*/*.cu")))

#: THE ENTRIES THAT ISSUE NO LAUNCH OF THEIR OWN, each with the reason it issues none.
#: Named rather than pattern-matched: "it looked like a query" is how an entry that
#: does launch acquires an exemption.
_LAUNCHLESS = {
    "carry_arms": "reads the built arm set out of compile-time constants",
    "intra_arms": "reads the built arm set out of compile-time constants",
    "rola_decode_arms": "reads the built arm set out of compile-time constants",
    "carry_build_stamp": "returns a compiled-in stamp",
    "intra_build_stamp": "returns a compiled-in stamp",
    "rola_decode_build_stamp": "returns a compiled-in stamp",
    "csrc_build_stamp": "returns a compiled-in stamp",
    "carry_geometry": "a host derivation over the descriptor; no device work",
    "carry_sub_boxes": "a host derivation over the descriptor; no device work",
    "rola_decode_capacity": "a host derivation over the descriptor; no device work",
    "rola_decode_producer_width_mirror": "returns a compiled-in width",
    "vmm_create": "a driver allocation, not a kernel launch",
    "vmm_probe": "a driver capability probe, not a kernel launch",
    "vmm_release": "drops the registry's reference to a driver allocation",
    "vmm_base": "a tensor view over a driver allocation, not a kernel launch",
    "vmm_grow": "maps driver memory, not a kernel launch",
    "vmm_rollback_to": "unmaps driver memory, not a kernel launch",
    "vmm_reset": "a logical reset of a driver allocation",
    "vmm_close": "releases a driver allocation",
    "vmm_facts": "reads a driver allocation's bookkeeping",
    "carry_census": "a cudaFuncGetAttributes query of the compiled arms, not a kernel launch",
    "carry_ledger_bind": "binds the device buffer the next carry launch adds into; issues no launch",
}


def _registered_entries() -> list[tuple[str, str]]:
    """``(operator name, C++ function)`` for every ``m.impl``, in registration order. An operator's implementation is
    always a named function (``TORCH_BOX(&fn)``), in a device translation unit or in this one."""
    text = REGISTRATION_TU.read_text()
    return [(m.group(1), m.group(2)) for m in re.finditer(r'm\.impl\(\s*"(\w+)"\s*,\s*TORCH_BOX\(&([\w:]+)\)\)', text)]


ENTRIES = [(name, fn.rsplit("::", 1)[-1]) for name, fn in _registered_entries() if name not in _LAUNCHLESS]


def test_the_entry_surfaces_are_declared_where_this_gate_looks():
    """The scan has something to scan: an entry list that came back empty would
    otherwise make this whole file vacuously green."""
    assert _registered_entries(), f"no m.impl entries found in {REGISTRATION_TU}"
    assert ENTRIES, "every declared entry is exempt; the gate would be vacuous"


def test_the_compile_time_half_is_a_table_and_a_static_assert():
    """``tabulated()`` gates the compiling target, and nothing softens it."""
    text = ARCH_CAPS.read_text()
    assert "constexpr bool tabulated(int cc)" in text
    assert re.search(r"static_assert\(\s*tabulated\(kArch\)", text), (
        "arch_caps.cuh no longer fails the BUILD on an untabulated compile target; "
        "that is half of the closed-world arch rule and it is not optional")


def _check_arch_table_definitions() -> list[pathlib.Path]:
    return [tu for tu in DEVICE_TUS if "void check_arch_table() {" in tu.read_text()]


def test_the_runtime_half_refuses_an_untabulated_architecture():
    """``check_arch_table()`` keys on ``tabulated`` and on the driver's own numbers."""
    defs = _check_arch_table_definitions()
    assert len(defs) == 1, (
        f"check_arch_table has {len(defs)} definitions "
        f"({[str(p.relative_to(REPO)) for p in defs]}); it needs exactly one -- a second "
        f"copy is a second thing that can drift from the capability table it checks, and "
        f"none at all is the runtime half of the closed-world arch rule missing")
    text = defs[0].read_text()
    body = text[text.index("void check_arch_table() {"):]
    body = body[:body.index("\n}\n") + 3]
    assert "arch::tabulated(cc)" in body, (
        "the runtime half must refuse an arch the TABLE does not carry -- an unbuilt "
        "ARM and an untabulated ARCHITECTURE are different refusals")
    assert "cudaGetDeviceProperties" in body and "cudaDeviceGetAttribute" in body, (
        "the check must read the DEVICE; a path or a hash cannot establish which "
        "device a compiled derivation was derived for")
    for field in ("smem_per_sm", "smem_per_cta_max", "regs_per_sm", "max_threads_per_sm"):
        assert f"want.{field}" in body, (
            f"the table's {field} is an input to a compiled derivation and must be "
            "compared against the driver")


@pytest.mark.parametrize(("name", "entry"), ENTRIES, ids=[n for n, _ in ENTRIES])
def test_every_launching_entry_passes_the_gate_first(name: str, entry: str):
    """No entry that reaches a kernel can be called without the refusal running.

    There is no single launcher to hang the check on -- the families' entries are
    independent -- so the property is per entry, and this is what keeps the next one
    from being added without it.
    """
    match = None
    for path in (*DEVICE_TUS, REGISTRATION_TU):
        match = re.search(rf"^[\w:<>,\s&*]+?\b{entry}\(.*?\)\s*\{{\n(.*?)\n^\}}",
                          path.read_text(), re.M | re.S)
        if match:
            break
    assert match, (
        f"cannot find the definition of {entry}; it is registered as an operator, so it is "
        f"a named function in a device translation unit or in {REGISTRATION_TU.name}")
    first = next(line.strip() for line in match.group(1).splitlines() if line.strip())
    assert first == "check_arch_table();", (
        f"{name} launches, so the closed-world arch refusal must be its "
        f"FIRST statement; found {first!r}")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the live half needs the device")
def test_this_device_is_tabulated_and_the_gate_admits_it():
    """The positive direction: the check ran, said yes, and a launch followed.

    A gate that only ever refuses is indistinguishable from a gate that is never
    reached, so the admitted case is asserted too -- and asserted through a real entry,
    because the entry is where the call site is. The liveness pass is that entry: it is
    the one surviving family whose launch takes no arm and no state, so what this cell
    exercises is the gate and nothing else.
    """
    from rola.engine.facts import liveness as lv
    from rola.ops.liveness import liveness_words

    major, minor = torch.cuda.get_device_capability()
    cc = major * 100 + minor * 10
    rows = re.search(r"constexpr bool tabulated\(int cc\) \{\s*return ([^;]+);",
                     ARCH_CAPS.read_text(), re.S).group(1)
    assert f"cc == {cc}" in rows, (
        f"this device is compute capability {major}.{minor}, which the capability table "
        "does not carry -- the refusal is then the correct behaviour, and this row is "
        "the one that must be added")
    widths, L = (16, 16), 32
    plane = torch.zeros(1, L, sum(widths), dtype=torch.bfloat16, device="cuda")
    plane[..., 0] = 1.0
    layout = lv.LivenessLayout(D=len(widths), B=widths, L=L)
    statics = lv.side_statics(layout, (False,) * layout.D, layout.B)
    words = liveness_words(plane, plane, layout, statics, statics)
    assert words.shape[0] == 1
