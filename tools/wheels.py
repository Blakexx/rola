"""THE WHEELS: the graph's library of a clean, ratified commit, packaged as `rola` (pure Python) and `rola-cu13` (the
binary plugin).

    bazel run //:wheels                          # dist/rola-<v>-py3-none-any.whl, dist/rola_cu13-<v>-cp310-abi3-linux_x86_64.whl
    bazel run //tools:wheels -- --check dist     # install both into a scratch venv and prove what each one does alone

Wheel naming D (docs/build.md#wheels): `pip install "rola[cu13]"` installs both at one version. The binary in
`rola-cu13` is `//:library`'s, which its gates passed; ratify's verdict on that build must pass first (it refuses an
iteration build), and the build's record must name this version, the toolchain and the digest of the manifests the
plugin carries. The metadata is `pyproject.toml`'s `[project]`, with the `cu13` extra
pinned to this version. Docs: docs/internals/tools/wheels.md.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_record  # noqa: E402
import ratify  # noqa: E402
import toolchains  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = "cu13"
PLUGIN = f"rola_{TOOLCHAIN}"
PLUGIN_TAG = "cp310-abi3-linux_x86_64"  #: the limited API torch builds against (3.10), and the host's platform
EPOCH = (1980, 1, 1, 0, 0, 0)  #: every member's timestamp, so one commit's wheels are the same bytes
#: a package's files are its tracked files but these placeholders, which keep an empty directory in git
PLACEHOLDERS = (".gitkeep",)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _clean_commit() -> str:
    dirty = _git("status", "--porcelain")
    if dirty.strip():
        raise SystemExit(f"refusing: a wheel is a commit, and this tree has changes:\n{dirty}")
    return _git("rev-parse", "HEAD").strip()


def _ratified(census: Path, runfiles: Path) -> None:
    """Ratify's verdict on the build the wheels package (`bazel test //:ratified`'s, in the checkout)."""
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "ratify.py"), "--graph", str(census), "--runfiles",
                          str(runfiles)], cwd=ROOT)
    if run.returncode:
        raise SystemExit("refusing: ratify's verdict on this build is not a pass; a wheel ships a ratified build")


def _record(path: Path) -> dict:
    scope: dict = {}
    exec(path.read_text(), scope)  # noqa: S102 -- the build's own generated record
    return scope["BUILD_CONFIG"]


def _verify(record: dict, version: str) -> list[Path]:
    """The record is this version's build under this toolchain, and the manifests the plugin will carry are the ones it
    was gated against. That it is no iteration build is ratify's verdict's, which refuses one (`_ratified`)."""
    toolchain = toolchains.named(TOOLCHAIN)
    archs = toolchain.archs()
    manifests = [toolchain.manifest_path(a) for a in archs]
    digest = ratify.manifest_digest(ratify.load_manifest(archs, TOOLCHAIN))
    wrong = {"version": (record["version"], version), "toolchain": (record["toolchain"], TOOLCHAIN),
             "manifest_sha256": (record["manifest_sha256"], digest)}
    wrong = {k: v for k, v in wrong.items() if v[0] != v[1]}
    if wrong:
        raise SystemExit(f"refusing: the build's record is not a shippable {TOOLCHAIN} build of rola {version} "
                         f"(field: recorded, required): {wrong}")
    return manifests


def _package(name: str) -> dict[str, bytes]:
    """A package's tracked files, its placeholders left out."""
    return {path: (ROOT / path).read_bytes() for path in _git("ls-files", "--", name).splitlines()
            if Path(path).name not in PLACEHOLDERS}


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


#: the licence files a project's metadata names, by setuptools' patterns, in its order
LICENCE_PATTERNS = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")


def _licence_files() -> list[str]:
    found: list[str] = []
    for pattern in LICENCE_PATTERNS:
        found += sorted(p.name for p in ROOT.glob(pattern) if p.is_file() and p.name not in found)
    return found


def metadata(project: dict, version: str) -> tuple[str, list[str]]:
    """`rola`'s METADATA from `[project]` (core metadata 2.4, as setuptools writes it: the licence's text folded under
    `License`, the README its body), with the `cu13` extra pinned to this version first; and the licence files it
    names."""
    licences = _licence_files()
    extras = {TOOLCHAIN: [f"{PLUGIN.replace('_', '-')}=={version}"], **project.get("optional-dependencies", {})}
    licence = (ROOT / project["license"]["file"]).read_text().replace("\n", "\n" + " " * 8)
    lines = ["Metadata-Version: 2.4", f"Name: {project['name']}", f"Version: {version}",
             f"Summary: {project['description']}", *(f"Author: {a['name']}" for a in project["authors"]),
             f"License: {licence}", f"Keywords: {','.join(project['keywords'])}",
             *(f"Classifier: {c}" for c in project["classifiers"]), f"Requires-Python: {project['requires-python']}",
             "Description-Content-Type: text/markdown", *(f"License-File: {name}" for name in licences),
             *(f"Requires-Dist: {d}" for d in project["dependencies"])]
    for extra, requirements in extras.items():
        lines += [f"Provides-Extra: {extra}", *(f'Requires-Dist: {r}; extra == "{extra}"' for r in requirements)]
    lines += ["Dynamic: license-file", "Dynamic: provides-extra", "", (ROOT / project["readme"]).read_text()]
    return "\n".join(lines), licences


def build(library: Path, record_path: Path, out: Path, version: str) -> list[Path]:
    """`rola` takes the `rola` package and the project's metadata; `rola-cu13` takes the plugin package, the graph's
    library and record, the manifests, and metadata of its own."""
    record = _record(record_path)
    manifests = _verify(record, version)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    text, licence_files = metadata(project, version)

    info = f"rola-{version}.dist-info"
    core = _package("rola")
    core.update({f"{info}/licenses/{name}": (ROOT / name).read_bytes() for name in licence_files})
    core[f"{info}/METADATA"] = text.encode()
    core[f"{info}/WHEEL"] = _wheel_meta("py3-none-any", purelib=True)
    core[f"{info}/top_level.txt"] = b"rola\n"

    plugin_info = f"{PLUGIN}-{version}.dist-info"
    plugin = _package(PLUGIN)
    plugin[f"{PLUGIN}/_C.abi3.so"] = library.read_bytes()
    plugin[f"{PLUGIN}/_build_config.py"] = record_path.read_bytes()
    plugin.update({f"{PLUGIN}/manifests/{m.name}": m.read_bytes() for m in manifests})
    plugin.update({f"{plugin_info}/licenses/{name}": (ROOT / name).read_bytes() for name in licence_files})
    #: the core's single-line fields that describe the plugin as well, in the core's order; `License` is not one (it
    #: holds the licence's whole text), the License-File entries name the same files
    lines = text.split("\n\n", 1)[0].splitlines()
    header = [line for line in lines if line.split(":", 1)[0] in ("Author", "Author-email", "License-File", "Classifier",
                                                                   "Requires-Python")]
    torch = next(line for line in lines if re.match(r"Requires-Dist: torch\b", line))
    plugin[f"{plugin_info}/METADATA"] = "\n".join([
        "Metadata-Version: 2.4", f"Name: rola-{TOOLCHAIN}", f"Version: {version}",
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
    ap.add_argument("--library", type=Path, help="the graph's library (`bazel run //:wheels` passes it)")
    ap.add_argument("--record", type=Path, help="the graph's build record")
    ap.add_argument("--census", type=Path, help="the build's census, for ratify's verdict")
    ap.add_argument("--runfiles", type=Path, help="the directory the census record's paths are relative to")
    a = ap.parse_args()
    if a.check:
        return check(a.check)
    if not (a.library and a.record and a.census and a.runfiles):
        raise SystemExit("the wheels package the graph's build: `bazel run //:wheels`")
    sha = _clean_commit()
    _ratified(a.census, a.runfiles)
    version = build_record.version(ROOT / "rola" / "__init__.py")
    for wheel in build(a.library, a.record, a.out, version):
        print(f"{wheel}  ({wheel.stat().st_size / 1e6:.1f} MB)  commit {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
