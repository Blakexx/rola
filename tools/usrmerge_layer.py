# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A PACKAGE LAYER FOR A MERGED-/USR BASE (the root package's `devbox_debs`): on Ubuntu 22.04 `/bin`, `/lib`, `/lib32`,
`/lib64`, `/libx32` and `/sbin` are links into `/usr`, and a layer that carries those paths as directories replaces the
links, so `/bin/sh` and the loader's own path go. The links are read from the base image itself; every entry under one
of them moves under `usr/`, as dpkg writes it through the link, and the entries for the directories themselves go.
Members keep their metadata, in order; a later one at a path wins. Docs: docs/setup.md#the-devbox-image.

    usrmerge_layer.py --base LAYOUT_DIR --in flat.tar --out usrmerged.tar
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

#: the layout's reader, from beside this file as Bazel lays it out (its declared input)
sys.path.insert(0, str(Path(__file__).absolute().parent))
from oci_layout import layers  # noqa: E402


def merged_links(layout: Path) -> frozenset[str]:
    """The base image's merged directories: its top-level links into `usr/` (`bin -> usr/bin`), as its layers leave
    them."""
    links: dict[str, bool] = {}
    for archive in layers(layout):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                name = member.name.removeprefix("./").rstrip("/")
                if name and "/" not in name:
                    links[name] = member.issym() and member.linkname.removeprefix("./").startswith("usr/")
    return frozenset(name for name, linked in links.items() if linked)


def merged_name(name: str, merged: frozenset[str]) -> str | None:
    """Where an entry lands on the merged-/usr base: under `usr/` when it is under a merged directory, None for one of
    those directories itself, else where it was."""
    head, _, rest = name.removeprefix("./").partition("/")
    if head in merged:
        return f"./usr/{head}/{rest}" if rest else None
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="the base image's OCI layout")
    ap.add_argument("--in", dest="source", type=Path, required=True, help="the flattened package layer")
    ap.add_argument("--out", type=Path, required=True, help="the layer for a merged-/usr base")
    a = ap.parse_args()
    merged = merged_links(a.base)
    if not merged:
        raise SystemExit(f"{a.base}: the base image links no top-level directory into usr/")
    with tarfile.open(a.source) as src:
        members = []
        for member in src.getmembers():
            name = merged_name(member.name, merged)
            if name is not None:
                members.append((name, member))
        last = {name: i for i, (name, _) in enumerate(members)}
        with tarfile.open(a.out, "w", format=tarfile.PAX_FORMAT) as out:
            for i, (name, member) in enumerate(members):
                if last[name] != i:
                    continue
                payload = src.extractfile(member) if member.isfile() else None
                member.name = name
                #: a hard link names its target by archive path, which moved as its target did
                if member.islnk():
                    member.linkname = merged_name(member.linkname, merged) or member.linkname
                out.addfile(member, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
