"""THE WHEELS: one gate build of a clean commit, split into `rola` (pure Python) and `rola-cu13` (the binary plugin).

    python tools/wheels.py                 # dist/rola-<v>-py3-none-any.whl, dist/rola_cu13-<v>-cp310-abi3-linux_x86_64.whl
    python tools/wheels.py --check dist    # install both into a scratch venv and prove what each one does alone

Wheel naming D (docs/build.md#wheels): `pip install "rola[cu13]"` installs both at one version. The split is a split
of ONE build's output, so the binary in `rola-cu13` is the binary the post-build gates passed: the build runs with every
iteration knob cleared and `ROLA_STRICT_MANIFEST=1`, and its record must say it built every ratified arch at this
version. The plugin carries the manifests that record was gated against, and their digest must be the one it holds.
Docs: docs/internals/tools/wheels.md.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ratify  # noqa: E402
import toolchains  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = "cu13"
PLUGIN = f"rola_{TOOLCHAIN}"
#: The knobs that make an iteration build or skip a gate; a wheel's build runs with none of them.
ITERATION_KNOBS = ("ROLA_CUDA_ARCHS", "ROLA_CARRY_ARMS", "ROLA_DECODE_ARMS", "ROLA_CARRY_PARTS", "ROLA_BUILD_PARTS",
                   "ROLA_NO_EXTENSION", "ROLA_SKIP_POST_BUILD_RATIFY")
PLUGIN_TAG = "cp310-abi3-linux_x86_64"  #: the limited API torch builds against (3.10), and the host's platform
EPOCH = (1980, 1, 1, 0, 0, 0)  #: every member's timestamp, so one commit's wheels are the same bytes


def _clean_commit() -> str:
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        raise SystemExit(f"refusing: a wheel is a commit, and this tree has changes:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _build(scratch: Path) -> Path:
    """The gate build as one combined wheel (both packages, pip's own metadata)."""
    env = {k: v for k, v in os.environ.items() if k not in ITERATION_KNOBS}
    env["ROLA_STRICT_MANIFEST"] = "1"
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(scratch)],
                   cwd=ROOT, env=env, check=True)
    (combined,) = scratch.glob("rola-*.whl")
    return combined


def _record(source: zipfile.ZipFile, name: str) -> dict:
    scope: dict = {}
    exec(source.read(name).decode(), scope)  # noqa: S102 -- the build's own generated record
    return scope["BUILD_CONFIG"]


def _verify(source: zipfile.ZipFile, version: str) -> tuple[dict, list[Path]]:
    """The combined wheel holds the one extension, no instrument, and a record of a shippable build of this version."""
    names = source.namelist()
    binaries = [n for n in names if n.endswith(".so")]
    if binaries != [f"{PLUGIN}/_C.abi3.so"]:
        raise SystemExit(f"refusing: the build's binaries are {binaries}; a wheel carries {PLUGIN}/_C.abi3.so alone")
    record = _record(source, f"{PLUGIN}/_build_config.py")
    toolchain = toolchains.named(TOOLCHAIN)
    archs = toolchain.archs()
    manifests = [toolchain.manifest_path(a) for a in archs]
    digest = ratify.manifest_digest(ratify.load_manifest(archs, TOOLCHAIN))
    wrong = {"version": (record["version"], version), "iteration": (record["iteration"], False),
             "archs": (record["archs"], [f"sm_{a}" for a in archs]), "toolchain": (record["toolchain"], TOOLCHAIN),
             "manifest_sha256": (record["manifest_sha256"], digest)}
    wrong = {k: v for k, v in wrong.items() if v[0] != v[1]}
    if wrong:
        raise SystemExit(f"refusing: the build's record is not a shippable {TOOLCHAIN} build of rola {version} "
                         f"(field: recorded, required): {wrong}")
    return record, manifests


def _wheel_file(members: dict[str, bytes], dist_info: str, out: Path) -> Path:
    """`members` plus a RECORD, stored in name order with fixed timestamps."""
    rows = [[name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
             str(len(data))] for name, data in sorted(members.items())]
    table = io.StringIO()
    csv.writer(table, lineterminator="\n").writerows([*rows, [f"{dist_info}/RECORD", "", ""]])
    members = {**members, f"{dist_info}/RECORD": table.getvalue().encode()}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as whl:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, EPOCH)
            info.external_attr = (0o755 if name.endswith(".so") else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            whl.writestr(info, members[name])
    return out


def _wheel_meta(tag: str, purelib: bool) -> bytes:
    return (f"Wheel-Version: 1.0\nGenerator: rola tools/wheels.py\nRoot-Is-Purelib: {str(purelib).lower()}\n"
            f"Tag: {tag}\n").encode()


def split(combined: Path, out: Path, version: str) -> list[Path]:
    """`rola` takes the `rola` package and the build's metadata; `rola-cu13` takes the plugin package, the manifests and
    metadata of its own. Neither takes anything else of the other's."""
    with zipfile.ZipFile(combined) as source:
        _, manifests = _verify(source, version)
        info = f"rola-{version}.dist-info"
        files = {n: source.read(n) for n in source.namelist() if not n.endswith("/")}
    metadata = files[f"{info}/METADATA"].decode()
    licenses = {n: d for n, d in files.items() if n.startswith(f"{info}/licenses/")}

    core = {n: d for n, d in files.items() if n.startswith("rola/")}
    core.update({n: d for n, d in files.items() if n.startswith(f"{info}/") and n.rsplit("/", 1)[1] != "RECORD"})
    core[f"{info}/WHEEL"] = _wheel_meta("py3-none-any", purelib=True)
    core[f"{info}/top_level.txt"] = b"rola\n"

    plugin_info = f"{PLUGIN}-{version}.dist-info"
    #: the build's single-line fields that describe the plugin as well; `License` is not one (setuptools writes the
    #: licence's whole text into it), the License-File entries name the same files
    lines = metadata.splitlines()
    header = [line for line in lines if line.split(":", 1)[0] in ("Author", "Author-email", "License-File", "Classifier",
                                                                   "Requires-Python")]
    torch = next(line for line in lines if re.match(r"Requires-Dist: torch\b", line))
    plugin = {n: d for n, d in files.items() if n.startswith(f"{PLUGIN}/")}
    plugin.update({f"{PLUGIN}/manifests/{m.name}": m.read_bytes() for m in manifests})
    plugin.update({n.replace(info, plugin_info, 1): d for n, d in licenses.items()})
    plugin[f"{plugin_info}/METADATA"] = "\n".join([
        lines[0], f"Name: rola-{TOOLCHAIN}", f"Version: {version}",
        f"Summary: RoLA's kernels built by the {TOOLCHAIN} toolchain, with the manifests they were ratified against; "
        f"install as rola[{TOOLCHAIN}]", *header, f"Requires-Dist: rola=={version}", torch, ""]).encode()
    plugin[f"{plugin_info}/WHEEL"] = _wheel_meta(PLUGIN_TAG, purelib=False)
    plugin[f"{plugin_info}/top_level.txt"] = f"{PLUGIN}\n".encode()

    out.mkdir(parents=True, exist_ok=True)
    return [_wheel_file(core, info, out / f"rola-{version}-py3-none-any.whl"),
            _wheel_file(plugin, plugin_info, out / f"{PLUGIN}-{version}-{PLUGIN_TAG}.whl")]


_PROBE = """
import rola
from rola.ops import _ext
try:
    _ext.extension()
    print("LOADED", sorted(map(tuple, _ext.extension().carry_arms())))
except ImportError as e:
    print("REFUSED", str(e).splitlines()[0], "|", [l.strip() for l in str(e).splitlines() if "pip install" in l])
"""


def check(wheels: Path) -> int:
    """A scratch venv over this interpreter's torch: `rola` alone refuses a kernel naming the extra, and with
    `rola-cu13` beside it the binary loads and names its carry arms."""
    (core,) = wheels.glob("rola-*-py3-none-any.whl")
    (plugin,) = wheels.glob(f"{PLUGIN}-*.whl")
    torch_site = Path(subprocess.run([sys.executable, "-c", "import torch, pathlib; print(pathlib.Path(torch.__file__)"
                                      ".parents[1])"], capture_output=True, text=True, check=True).stdout.strip())
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run(["uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", "--seed",
                        str(venv)], check=True, capture_output=True)
        py = str(venv / "bin" / "python")
        site = Path(subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                   capture_output=True, text=True, check=True).stdout.strip())
        (site / "zz_torch.pth").write_text(f"{torch_site}\n")
        verdicts = []
        for wheel in (core, plugin):
            subprocess.run([py, "-m", "pip", "install", "--no-deps", "-q", str(wheel)], check=True)
            probe = subprocess.run([py, "-c", _PROBE], cwd=tmp, capture_output=True, text=True)
            verdicts.append(probe.stdout.strip() or probe.stderr.strip()[-600:])
            print(f"{wheel.name}: {verdicts[-1]}")
    ok = verdicts[0].startswith("REFUSED") and f"rola[{TOOLCHAIN}]" in verdicts[0] and verdicts[1].startswith("LOADED")
    print("wheels check", "OK" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--check", type=Path, metavar="DIR", help="check the wheels in DIR instead of building")
    a = ap.parse_args()
    if a.check:
        return check(a.check)
    sha = _clean_commit()
    version = re.search(r'^__version__\s*=\s*"([^"]+)"', (ROOT / "rola" / "__init__.py").read_text(), re.M).group(1)
    with tempfile.TemporaryDirectory() as tmp:
        for wheel in split(_build(Path(tmp)), a.out, version):
            print(f"{wheel}  ({wheel.stat().st_size / 1e6:.1f} MB)  commit {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
