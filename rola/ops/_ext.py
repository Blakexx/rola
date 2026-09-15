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

import importlib
from functools import lru_cache

from rola import __version__

#: THE BINARY PLUGIN (wheel naming D, docs/build.md#wheels): `rola` is pure Python and the build of each toolchain is
#: its own distribution, `rola-cu13`, whose package holds the extension, its build record and its manifests. A
#: checkout's in-place build lays out the same package, so an installed wheel and a checkout take this one import.
PLUGIN, EXTRA = "rola_cu13", "cu13"


def _load():
    """The extension module and None, or None and why not. The plugin's build record is read BEFORE its library
    loads: a binary built for another `rola` never registers its operators in this process."""
    try:
        built = importlib.import_module(f"{PLUGIN}._build_config").BUILD_CONFIG
    except ImportError as e:
        return None, f"no `{PLUGIN}` binary plugin is installed ({e})"
    if built["version"] != __version__:
        return None, (f"`{PLUGIN}` {built['version']} is not this `rola` ({__version__}); the two are one version, "
                      f"installed together")
    try:
        return importlib.import_module(f"{PLUGIN}._C"), None  # loading the library registers torch.ops.rola
    except ImportError as e:
        return None, f"`{PLUGIN}`'s extension did not load ({e})"


_C, _WHY = _load()


class RoLAVmmOwner:
    """The CUDA-driver VMM backing of a page arena (``csrc/rola/src/paging/vmm_owner.cuh``), held by the extension under
    an integer handle: the operators take the handle, and this object is the handle's Python face. Releasing it drops
    the extension's reference; a tensor view of the arena keeps the owner alive past that."""

    backing_kind = "cuda_driver_vmm"
    _FACTS = ("closed", "mapped_capacity_pages", "dense_limit_pages", "committed_bytes", "virtual_bytes",
              "allocation_granularity", "chunk_pages")

    def __init__(self, ops, handle: int) -> None:
        self._ops, self._handle = ops, handle

    def _facts(self) -> dict:
        return dict(zip(self._FACTS, self._ops.vmm_facts(self._handle), strict=True))

    def base(self):
        return self._ops.vmm_base(self._handle)

    def grow(self, required_pages: int) -> None:
        self._ops.vmm_grow(self._handle, required_pages)

    def rollback_to(self, mapped_capacity_pages: int) -> None:
        self._ops.vmm_rollback_to(self._handle, mapped_capacity_pages)

    def reset(self) -> None:
        self._ops.vmm_reset(self._handle)

    def close(self) -> None:
        self._ops.vmm_close(self._handle)

    def __del__(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle is not None:
            self._ops.vmm_release(handle)

    closed = property(lambda self: bool(self._facts()["closed"]))
    mapped_capacity_pages = property(lambda self: self._facts()["mapped_capacity_pages"])
    dense_limit_pages = property(lambda self: self._facts()["dense_limit_pages"])
    committed_bytes = property(lambda self: self._facts()["committed_bytes"])
    virtual_bytes = property(lambda self: self._facts()["virtual_bytes"])
    allocation_granularity = property(lambda self: self._facts()["allocation_granularity"])
    chunk_pages = property(lambda self: self._facts()["chunk_pages"])


class _Extension:
    """rola's operators (``torch.ops.rola``) under the names the package calls them by, and the VMM owner's handle as an
    object."""

    def __init__(self, ops) -> None:
        self._ops = ops

    def __getattr__(self, name: str):
        return getattr(self._ops, name)

    def rola_vmm_probe(self, device: int) -> dict:
        found_device, supported, granularity, reason = self._ops.vmm_probe(device)
        return {"device": found_device, "supported": supported, "allocation_granularity": granularity,
                "reason": reason}

    def rola_vmm_create(self, device: int, dense_limit_pages: int, page_rows: int, page_cols: int,
                        target_chunk_bytes: int = 64 * 1024 * 1024) -> RoLAVmmOwner:
        return RoLAVmmOwner(self._ops, self._ops.vmm_create(device, dense_limit_pages, page_rows, page_cols,
                                                            target_chunk_bytes))


_rola_cuda: _Extension | None = None
if _C is not None:
    import torch

    _rola_cuda = _Extension(torch.ops.rola)


_MISSING = """\
No RoLA kernel can run: {why}.

Install the binary for your CUDA toolchain at this version:

    pip install "rola[{extra}]=={version}"

It carries the kernels for every ratified architecture and the manifests they were measured against. A checkout builds
the same package in place (`pip install -e . --no-build-isolation`, docs/build.md), and refuses an architecture that
has no measured manifest (docs/bringup.md).

There is no CPU or pure-torch fallback by design: `rola.ops.naive.naive_rola` is a float64 reference for checking the
kernel, not a substitute for it.\
"""


@lru_cache(maxsize=1)
def extension():
    """Return the compiled extension module, or raise with an actionable message."""
    if _rola_cuda is None:
        raise ImportError(_MISSING.format(why=_WHY, extra=EXTRA, version=__version__))
    return _rola_cuda


def is_available() -> bool:
    """True when the compiled extension imported successfully.

    For probing (tests, capability reporting). Never branch the math on this —
    see the module docstring on why there is no fallback path.
    """
    return _rola_cuda is not None


__all__ = ["extension", "is_available"]
