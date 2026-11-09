# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""AN OCI IMAGE LAYOUT, READ (the devbox image's, `//:image`, and its base's): the one image's manifest digest -- the
image's identity, which `tools/devbox_image.json` names -- the tag the tree gives it, and its layers. Every reader of a
layout reads it here: the container commands (`tools/dev.py`), the push's tag (`//:image_tag`), the package layer's
rewrite and check (`tools/usrmerge_layer.py`, `tools/devbox_packages.py`). Docs: docs/setup.md#the-devbox-image.

    oci_layout.py --tag LAYOUT_DIR      # the tag: the digest's first 12 hex digits
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path


def _manifest(layout: Path) -> dict:
    return json.loads((layout / "index.json").read_text())["manifests"][0]


def digest(layout: Path) -> str:
    """The layout's one image's manifest digest (`sha256:...`), which a registry serves it by."""
    return _manifest(layout)["digest"]


def tag(image_digest: str) -> str:
    """The tag the tree gives an image: its digest's first 12 hex digits."""
    return image_digest.removeprefix("sha256:")[:12]


def layers(layout: Path) -> Iterator[Path]:
    """The layer archives of the layout's one image, base first."""
    blob = lambda name: layout / "blobs" / "sha256" / name.removeprefix("sha256:")  # noqa: E731
    for layer in json.loads(blob(digest(layout)).read_text())["layers"]:
        yield blob(layer["digest"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", type=Path, required=True, help="the layout whose tag to print")
    print(tag(digest(ap.parse_args().tag)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
