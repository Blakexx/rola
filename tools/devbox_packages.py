# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEVBOX IMAGE'S PACKAGE LAYER ADDS PACKAGES, NEVER CHANGES THE BASE'S (the root package's `devbox_packages`):
rules_distroless lays every package of the closure over the base image, its own core packages among them, so every
package the base already has must be at the base's version, or the layer would swap a library under the base's
programs. The base's packages come from its dpkg status, in its OCI layout; the layer's from its packages' repositories,
each named by rules_distroless's sanitized key (`<suite>_<name>-<arch>_<version>`), so the base's are keyed the same
way. Docs: docs/setup.md#the-devbox-image.

    devbox_packages.py --base LAYOUT_DIR --layer LAYER_TAR -- PACKAGE_FILE...

The layer must also carry nothing at the base's merged-/usr links (`/bin`, `/lib`, ...: the base's own top-level links
into `/usr`), which a directory entry there would replace (tools/usrmerge_layer.py).
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path

#: the siblings it reads the base's layers and links with, from beside this file as Bazel lays it out (its inputs)
sys.path.insert(0, str(Path(__file__).absolute().parent))
from oci_layout import layers  # noqa: E402
from usrmerge_layer import merged_links  # noqa: E402

STATUS = "var/lib/dpkg/status"


def installed(status: str) -> list[tuple[str, str, str]]:
    """`(package, architecture, version)` of a dpkg status file's installed stanzas."""
    found = []
    for stanza in status.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in stanza.splitlines() if ": " in line and not line.startswith(" "))
        if {"Package", "Architecture", "Version"} <= set(fields) and fields.get("Status", "").endswith(" installed"):
            found.append((fields["Package"], fields["Architecture"], fields["Version"]))
    return found


def sanitize(text: str) -> str:
    """rules_distroless's repository-name sanitizer (apt/private/util.bzl's `_sanitize`), so a key compares as it does."""
    return text.removeprefix("/").replace("+", "-").replace(":", "-").replace("~", "_").replace("/", "_").replace("=", "_")


#: a package repository's name: rules_distroless's extension prefix, then its sanitized `/<suite>/<name>:<arch>=<version>`
REPOSITORY = re.compile(r"rules_distroless\+\+apt\+(jammy(?:-updates|-security)?)_(.+)$")


def _status_in(archive: Path) -> str | None:
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if member.name.lstrip("./") == STATUS and member.isfile():
                return tf.extractfile(member).read().decode()
    return None


def base_status(layout: Path) -> str:
    """The base image's dpkg status: the last of its layers that writes one."""
    status = None
    for layer in layers(layout):
        status = _status_in(layer) or status
    if status is None:
        raise SystemExit(f"{layout}: no layer of the base image writes {STATUS}")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="the base image's OCI layout")
    ap.add_argument("--layer", type=Path, required=True, help="the package layer the image carries")
    ap.add_argument("packages", nargs="+", type=Path, help="the layer's packages' files, one repository a package")
    a = ap.parse_args()
    with tarfile.open(a.layer) as tf:
        linked = sorted({n.removeprefix("./").split("/", 1)[0] for n in tf.getnames()} & merged_links(a.base))
    if linked:
        print(f"devbox packages: the layer carries {linked}, which would replace the base's links into /usr")
        return 1
    #: the base's packages keyed as the layer's are, by `<name>-<arch>_<version>` and by `<name>-<arch>_` alone; an
    #: `all` package under amd64 too, as rules_distroless keys it once an architecture
    base: set[str] = set()
    prefixes: dict[str, str] = {}
    for name, arch, version in installed(base_status(a.base)):
        for keyed in {arch, "amd64"} if arch == "all" else {arch}:
            base.add(sanitize(f"{name}:{keyed}={version}"))
            prefixes[sanitize(f"{name}:{keyed}=")] = f"{name} {version}"
    layer = sorted({m.group(2) for path in a.packages for part in path.parts if (m := REPOSITORY.match(part))})
    same = [key for key in layer if key in base]
    moved = sorted(f"{key} over the base's {prefixes[prefix]}" for key in layer if key not in base
                   for prefix in prefixes if key.startswith(prefix))
    print(f"devbox packages: {len(layer)} in the layer, {len(same)} of them the base's at its version, "
          f"{len(moved)} the base's at another version")
    for line in moved:
        print(f"  {line}")
    return 1 if moved or not layer else 0


if __name__ == "__main__":
    sys.exit(main())
