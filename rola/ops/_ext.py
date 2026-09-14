# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""Access to the compiled CUDA extension. **This module imports; it does not build.**

That sentence is the whole point of the extraction. In the fork this file called
``torch.utils.cpp_extension.load(...)`` at first use, which meant the binary a user
ran was compiled by *their* toolchain, from source, at import time — the exact
thing the closed-world codegen policy exists to forbid. The register and spill
counts in ``tools/manifests/`` describe a binary that was *measured*; a JIT build
produces a different binary every time the environment moves, and no manifest can
speak for it.

So the extension is built ahead of time by ``setup.py`` (which runs the
ratification gates as part of the build) and is merely *imported* here. Every
build-environment knob that used to live in this file — ``CUDA_HOME``,
``TORCH_CUDA_ARCH_LIST``, ``MAX_JOBS``, the WSL ``libcuda`` search path — now
lives in ``setup.py``, where a build's environment belongs.

There is deliberately **no pure-torch fallback**. ``rola.ops.naive`` is an fp64
oracle, not a fast path: silently routing to it when the wheel is missing would
turn an installation problem into a ~100x slowdown that no test would catch.
The import failure is loud and names the remedy instead.
"""

from __future__ import annotations

from functools import lru_cache

#: PACKAGE-RELATIVE, and that is the whole of the ``rola._C`` namespacing: the binary
#: lives INSIDE the package in both the installed and the in-place-built layout, so
#: this import resolves the same way in both and can never pick up a stray top-level
#: ``.so`` that happens to sit in the working directory. See ``setup.py``'s
#: ``MODULE_NAME``.
_IMPORT_ERROR: ImportError | None = None
try:
    from rola import _C as _rola_cuda
except ImportError as e:  # noqa: BLE001
    _rola_cuda = None
    _IMPORT_ERROR = e


_MISSING = """\
The `rola` CUDA extension (`rola._C`) is not available, so no RoLA kernel can run.

  original import error: {err}

Install a ratified wheel for your (python, torch, CUDA) combination from the
project's releases, or build from source:

    pip install rola --no-build-isolation

A source build compiles ahead of time and re-runs the ratification gates; it will
refuse outright to build an architecture that has no measured manifest under
`tools/manifests/`. See docs/bringup.md to ratify a new architecture.

There is no CPU or pure-torch fallback by design: `rola.ops.naive.naive_rola` is a
float64 reference for checking the kernel, not a substitute for it.\
"""


@lru_cache(maxsize=1)
def extension():
    """Return the compiled extension module, or raise with an actionable message."""
    if _rola_cuda is None:
        raise ImportError(_MISSING.format(err=_IMPORT_ERROR))
    return _rola_cuda


def is_available() -> bool:
    """True when the compiled extension imported successfully.

    For probing (tests, capability reporting). Never branch the math on this —
    see the module docstring on why there is no fallback path.
    """
    return _rola_cuda is not None


__all__ = ["extension", "is_available"]
