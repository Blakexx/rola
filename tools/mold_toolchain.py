# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LINKER'S PROVENANCE: mold is the extension's linker unconditionally (Blake ruling, 2026-08-28), the graph's
`@mold` at `tools/mold_pin.json`'s release, fetched by its digest -- no fallback to `ld.bfd`/`ld.gold`, the
constitution's NO FALLBACKS AT THE ATOM rule extended to the toolchain.

mold LINKS the extension; it does not touch codegen (`tools/build_flags.py`'s module docstring: link flags are not part
of the ratified flag list a `ptxas` swap can move). So `tools/ratify.py` records the graph's mold in the manifest's
`toolchain.mold` block for PROVENANCE -- which linker produced the measured library -- and never compares it against a
SASS body.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def mold_version(path: str) -> str:
    """The release `mold --version` names, e.g. `mold 2.42.0`."""
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{path} --version` failed (rc={proc.returncode}); a closed-world build cannot record a "
                           f"linker it cannot ask")
    #: `mold --version` prints e.g. "mold 2.42.0 (441dc1d...; compatible with GNU ld)": the release is the leading
    #: "mold X.Y.Z" token, the trailing commit hash not part of the release identity the pin names
    return proc.stdout.strip().split(" (")[0]


def binary_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
