# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LINKER PIN -- mold is not an option, it is THE linker (Blake ruling, 2026-08-28).

"A tool that can run one of two ways must REFUSE, not choose." A build must not
silently differ by whether `mold` happened to be on a given host's PATH, the way
`ptxas`/`sccache` are already pinned and gated (`tools/sccache_toolchain.py`,
closed-world rule 1). `tools/mold_pin.json` pins the exact upstream release
(the SAME artifact `Dockerfile`'s `MOLD_VERSION`/`MOLD_SHA256` ARGs install, so
the host-side gate here and the container's install agree on one pin, never two).

mold LINKS the extension; it does not touch codegen (`tools/build_flags.py`'s
module docstring: link flags are explicitly not part of the ratified,
codegen-affecting flag list a `ptxas` swap can move). So `gate()` is still a HARD
build-time refusal on a miss (no fallback to `ld.bfd`/`ld.gold` -- the
constitution's NO FALLBACKS AT THE ATOM rule extended to the toolchain), but its
result is recorded in `tools/ratify.py`'s manifest `toolchain.mold` block for
PROVENANCE -- which linker produced the shipped `.so` -- not as part of the
SASS-body ratchet: a linker version drift is reported, never a ratchet failure.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev_config  # noqa: E402 -- path insert must precede this import

PIN_PATH = HERE / "mold_pin.json"


def pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def resolve_mold() -> str | None:
    """`toolchain.mold` in the dev config -- no search path; `tools/dev.py init` finds it once."""
    return dev_config.get("toolchain.mold")


def mold_version(path: str) -> str:
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{path} --version` failed (rc={proc.returncode}); a "
                           f"closed-world build cannot gate a linker it cannot ask")
    #: `mold --version` prints e.g. "mold 2.42.0 (441dc1d...; compatible with GNU
    #: ld)" -- the pin matches on the LEADING "mold X.Y.Z" token, not the whole
    #: line, since the trailing commit hash is not part of the release identity
    #: the pin names.
    return proc.stdout.strip().split(" (")[0]


def binary_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link_flags() -> list[str]:
    """`-fuse-ld=mold` for the extension link. Raises on any pin miss -- no
    fallback to the default linker, per the constitution's NO FALLBACKS rule
    extended to the toolchain (Blake ruling 2026-08-28)."""
    p = pin()
    path = resolve_mold()
    if path is None or not Path(path).exists():
        raise RuntimeError(
            f"THE LINKER PIN (tools/mold_pin.json): no `mold` binary found (checked "
            f"toolchain.mold in the dev config; `python tools/dev.py init` installs it). Pinned build is {p['version_string']} "
            f"({p['release_asset']}, {p['release_url']}). mold is the extension's "
            f"linker unconditionally -- there is no fallback to the default linker; "
            f"install the pinned release (docs/setup.md names the host-local "
            f"vendoring recipe for a box with no apt package for it).")

    version = mold_version(path)
    if version != p["version_string"]:
        raise RuntimeError(
            "THE LINKER PIN (tools/mold_pin.json): the resolved `mold` does not "
            f"match the pin.\n  pinned  : {p['version_string']}\n"
            f"  resolved: {version}  ({path})\n"
            "Install the pinned release, or move the pin DELIBERATELY after "
            "re-verifying the link still produces the same extension (mold does "
            "not touch codegen, so this is a provenance question, not a SASS one, "
            "but the pin itself is still a hard gate: no build may differ by "
            "which linker version happened to be on a host's PATH).")

    #: `-fuse-ld=mold` resolves `ld.mold` by NAME via `-B<dir>`, not by absolute
    #: path (gcc's `-fuse-ld=` only accepts a small fixed name set on the
    #: toolchain versions this repo pins; an absolute path is not portable across
    #: them) -- the release tarball ships `bin/mold` alongside a `bin/ld.mold`
    #: symlink for exactly this invocation.
    bindir = Path(path).resolve().parent
    if not (bindir / "ld.mold").exists():
        raise RuntimeError(
            f"THE LINKER PIN: found `mold` at {path} but no `ld.mold` beside it in "
            f"{bindir} -- the pinned release layout is `bin/{{mold,ld.mold}}`; a "
            f"bare `mold` binary with no `ld.mold` sibling cannot be selected via "
            f"`-fuse-ld=mold`.")
    return [f"-B{bindir}", "-fuse-ld=mold"]


def provenance() -> dict:
    """`{"version", "sha256", "path"}` of the resolved, pin-checked `mold` --
    called AFTER `link_flags()` has already gated it, for `tools/ratify.py`'s
    manifest `toolchain.mold` block."""
    path = resolve_mold()
    return {"version": mold_version(path), "sha256": binary_sha256(path), "path": path}
