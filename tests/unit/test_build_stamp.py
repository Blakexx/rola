# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The extension's freshness fact, against the tree it is supposed to describe.

A stale binary is this project's most-repeated failure, and every check that sits one
level above it -- the import path, the file's hash, the mtime -- passes on the exact
shape that keeps recurring: the right file, at the right path, built from sources that
have since moved. The only check that cannot be fooled is a fact the DEVICE produces
out of a constant the compile put there.

`csrc_build_stamp()` is that fact, and this file is what makes it mean something: the
value the loaded fatbin returns must equal the content hash of `csrc/rola/src` as this
tree hashes it right now. A stamp nothing compares to the tree is a number, not a gate.

The digest function is read from `tools/ratify.py` BY PATH rather than imported from an
installed package: the fact wanted here is a property of THIS repository, and a tool
that imports the package to learn it can be answered by another worktree's build.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

from rola.ops._ext import extension

pytestmark = pytest.mark.cuda

REPO = pathlib.Path(__file__).resolve().parents[2]


def _ratify():
    spec = importlib.util.spec_from_file_location(
        "rola_ratify_for_stamp", REPO / "tools" / "ratify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_the_device_stamp_is_this_trees_csrc_content_hash():
    ratify = _ratify()
    got = extension().csrc_build_stamp()
    want = ratify.csrc_stamp(ratify.csrc_digest())
    assert got == want, (
        f"the loaded fatbin reports the csrc stamp {got}, this tree hashes to {want}. "
        f"The binary was built from other sources than the ones being tested; rebuild "
        f"before trusting any measurement taken through it.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_the_stamp_moves_when_the_sources_do():
    """The gate is only worth its run if a moved tree fails it.

    Hashing a perturbed copy of the tree stands in for the edit: it exercises the
    comparison this file makes rather than merely asserting the number that is there.
    """
    ratify = _ratify()
    real = ratify.csrc_digest()
    perturbed = ratify.csrc_stamp("0" * 63 + "1")
    assert extension().csrc_build_stamp() != perturbed
    assert ratify.csrc_stamp(real) != perturbed
