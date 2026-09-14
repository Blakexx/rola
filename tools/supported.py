"""Generate the README's "Supported configurations" table. **A REPORT, NOT A CLAIM.**

A hand-maintained support table says what someone hoped was true when
they last edited it; this one is derived from the two artifacts that decide the
answer, and CI fails on drift:

* ``csrc/rola/src/common/arch_caps.cuh`` -- the architectures the KERNEL knows how to
  size itself for. ``sm_arch_tabulated()`` is a ``static_assert`` at compile time
  and the chunk dispatch's first-launch check a refusal at run time, so an arch
  missing from it cannot execute whatever the table says.
* ``tools/manifests/<toolchain>/sm_XX.json`` -- the architectures MEASURED, per toolchain (``tools/toolchains.py``).
  One file per arch, so "is it ratified?" is a file-existence question.

The two are not the same set, and **the difference is the product statement**. An
arch that is tabulated but not ratified is listed with an empty toolchain column
and the status *not ratified -- will refuse to load*. A support claim is worth
nothing if the table quietly implies coverage nobody measured.

    python tools/supported.py            # print the table
    python tools/supported.py --check    # fail if README.md has drifted

GPU names are the only hand-written column, in ``GPUS`` below, because a compute
capability does not know its own marketing name. They are labels; the ratification
is what the table is actually reporting.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import toolchains

ROOT = Path(__file__).resolve().parents[1]
ARCH_TABLE = ROOT / "csrc" / "rola" / "src" / "common" / "arch_caps.cuh"
README = ROOT / "README.md"

#: The README's generated regions: `marker -> producer`. TWO of them, because two
#: different artifacts decide two different tables and neither may be hand-edited.
#: `capability_table()` is the activation registry's own generator -- this
#: script only places it, so there is still exactly one table in the codebase.
BLOCKS = {
    "tools/supported.py": lambda: table(),
    "rola.routing.activations.capability_table": lambda: _capability_table(),
}


def _capability_table() -> str:
    from rola.routing.activations import capability_table

    return capability_table().rstrip()


#: Marketing names per compute capability. The ONLY hand-written column.
GPUS = {
    "sm_80": "A100, A30",
    "sm_86": "A10, A40, RTX 30xx",
    "sm_87": "Jetson Orin",
    "sm_89": "L40S, RTX 40xx",
}

_TABULATED = re.compile(r"return\s+((?:cc\s*==\s*\d+\s*\|\|\s*)*cc\s*==\s*\d+)\s*;")


def tabulated_archs() -> list[str]:
    """The `sm_XX` the kernel can size itself for, read out of `tabulated`.

    Parsed from the header rather than restated here, because a second list is a
    list that can disagree with the `static_assert`.
    """
    text = ARCH_TABLE.read_text()
    body = text.split("constexpr bool tabulated", 1)[1]
    match = _TABULATED.search(body)
    if match is None:
        raise SystemExit(
            "cannot read `tabulated()` from common/arch_caps.cuh. The supported "
            "table is GENERATED from that function; if its shape changed, this "
            "parser must change with it rather than the table being hand-written.")
    codes = [int(n) for n in re.findall(r"\d+", match.group(1))]
    return [f"sm_{c // 10}" for c in codes]


def ratified() -> dict[str, list[dict]]:
    """`sm_XX -> [{toolchain, ptxas, instantiations}]`, one entry per toolchain with a committed manifest for it."""
    out: dict[str, list[dict]] = {}
    for toolchain in toolchains.records().values():
        for arch in toolchain.archs():
            blob = json.loads(toolchain.manifest_path(arch).read_text())
            out.setdefault(f"sm_{arch}", []).append({
                "toolchain": toolchain.name,
                "ptxas": blob["toolchain"]["ptxas"],
                "instantiations": len(blob["entries"]),
            })
    return out


def _ptxas_short(version: str) -> str:
    m = re.search(r"release ([\d.]+), V([\d.]+)", version)
    return f"ptxas {m.group(2)}" if m else version


def table() -> str:
    rat = ratified()
    rows = ["| Compute capability | GPUs | Toolchain | Ratified under | Instantiations | Status |",
            "|---|---|---|---|---|---|"]
    for arch in tabulated_archs():
        gpus = GPUS.get(arch, "—")
        for r in rat.get(arch, []):
            rows.append(f"| `{arch}` | {gpus} | `{r['toolchain']}` | {_ptxas_short(r['ptxas'])} | "
                        f"{r['instantiations']} | ratified |")
        if arch not in rat:
            rows.append(f"| `{arch}` | {gpus} | — | — | — | "
                        f"**not ratified — will refuse to load** |")
    #: Anything ratified but NOT tabulated would be a manifest for an arch the
    #: kernel cannot size itself for. It cannot happen through the normal path, and
    #: if it ever does the table must say so rather than quietly drop the row.
    for arch in sorted(set(rat) - set(tabulated_archs())):
        for r in rat[arch]:
            rows.append(f"| `{arch}` | {GPUS.get(arch, '—')} | `{r['toolchain']}` | "
                        f"{_ptxas_short(r['ptxas'])} | {r['instantiations']} | "
                        f"**ratified but NOT in the kernel's arch table — a defect** |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if README.md's generated block has drifted")
    ap.add_argument("--write", action="store_true", help="rewrite README.md's block")
    a = ap.parse_args()

    if not (a.check or a.write):
        for marker, produce in BLOCKS.items():
            print(f"--- {marker}")
            print(produce())
        return 0

    text = new = README.read_text()
    for marker, produce in BLOCKS.items():
        begin, end = f"<!-- BEGIN GENERATED: {marker} -->", f"<!-- END GENERATED: {marker} -->"
        if begin not in new or end not in new:
            raise SystemExit(f"README.md has no generated block for {marker}")
        head, rest = new.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = f"{head}{begin}\n{produce()}\n{end}{tail}"

    if a.write:
        README.write_text(new)
        print(f"README.md: {len(BLOCKS)} generated blocks rewritten")
        return 0
    if new != text:
        sys.stderr.write(
            "README.md's GENERATED tables have DRIFTED from the artifacts that decide "
            "them (the kernel's arch table, the committed manifests, the activation "
            "registry). Regenerate:\n"
            "    python tools/supported.py --write\n"
            "A support table edited by hand is a claim, not a report.\n")
        return 1
    print(f"README.md: {len(BLOCKS)} generated blocks are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
