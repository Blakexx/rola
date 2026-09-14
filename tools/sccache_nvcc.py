#!/usr/bin/env python3
"""THE nvcc<->sccache SEAM -- a pure pass-through, never a resolver.

Both call sites that use this wrapper (`setup.py`'s `PYTORCH_NVCC`, and
`tools/ratify.py`'s post-build recompile) have ALREADY resolved and gated the real
`nvcc` they want -- `setup.py` via `cuda_bin("nvcc")` under closed-world rule 1's
`ptxas --version` check, `ratify.py` the same way. This wrapper does not locate
`nvcc` itself: if it did, it and its caller could each resolve a different
`CUDA_HOME` (a stale `PATH`, a second toolkit installed later, ...) and the cache
would key an object under a toolchain identity nobody gated. Instead the caller
hands the exact resolved path through `ROLA_REAL_NVCC`, and this script does exactly
one thing: `exec sccache <that nvcc> <args>`.

That is also why this file has no logic to unit-test: its only claim is "call
these two things, in this order, with these arguments," and that claim is checked
by the bit-identity gate the caller runs (`docs/build.md#sccache`), not by a test
of this file in isolation.
"""
import os
import sys

REAL_NVCC = os.environ.get("ROLA_REAL_NVCC")
if not REAL_NVCC:
    sys.exit(
        "tools/sccache_nvcc.py: ROLA_REAL_NVCC is not set. This wrapper never "
        "resolves nvcc on its own (closed-world rule 1: the resolved toolchain "
        "must come from the already-gated caller, not be re-derived here) -- set "
        "it to the exact nvcc path the caller gated, or invoke through "
        "tools/sccache_toolchain.py's helpers instead of directly."
    )

SCCACHE_BIN = os.environ.get("ROLA_SCCACHE_BIN")
if not SCCACHE_BIN:
    sys.exit("tools/sccache_nvcc.py: ROLA_SCCACHE_BIN is not set -- the gated sccache is handed over by the caller "
             "(tools/sccache_toolchain.py wrap), never resolved here.")
os.execvp(SCCACHE_BIN, [SCCACHE_BIN, REAL_NVCC] + sys.argv[1:])
