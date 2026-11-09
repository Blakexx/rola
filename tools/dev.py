#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEV ENVIRONMENT (KERNEL_STANDARDS §23): the commands that set up, check and describe a host or container, each
run as the graph's target of the same name (`bazel/defs.bzl`'s `dev_run`), its arguments after `--`:

    python3 tools/bootstrap.py                   # the one step outside the graph: the pinned bazelisk and `bazel`
    bazel run //:init [-- --force]               # write this machine's dev config, install the pinned tools, wire hooks
    bazel run //:check                           # every value, tool, pin and lock this tree depends on; exit 1 on a red
    bazel run //:show                            # every resolved setting and the file it came from
    bazel run //:get -- <section.key>            # one value, for a person or a shell
    bazel run //:clock -- --mhz 1665             # lock this host's SM clock once, prove the round trip, record it
    bazel run //:dev_env                         # this checkout's venv over the graph's environment, its hooks' venv
    bazel run //:worktree -- <name> <sha>        # a worktree, its venv and gate, its extension (//:library)
    bazel run //:container_check                 # the devbox image's clean room: the host's loop inside it, recorded
    bazel run //:container_compose               # the devbox image for an editor's dev container, on its own volumes

The settings live in the dev config directory (`tools/dev_config.py`). Docs: docs/internals/tools/dev.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import
import oci_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: THE DEVBOX'S TOOLS outside the graph's environment: a developer's commands, each a pinned release in its own
#: versioned folder under `host.tools_dir` (`dev_config.pinned`), linked into every checkout's venv (`dev-env`) and found by
#: its pin, never recorded as a machine key -- see docs/setup.md#the-devbox
DEV_TOOLS = ("hyperfine", "difftastic", "nsight_compute", "nsight_systems")
#: THE CUDA TOOLKIT the tree's tools outside the graph use (`dev_config.cuda_home`): NVIDIA's redistributable components at
#: the graph's toolkit manifest, installed by init as a devbox tool is, on no PATH -- see docs/setup.md#the-devbox
TOOLKIT = "cuda_toolkit"
PINS = {"bazelisk": ROOT / "tools" / "bazelisk_pin.json",
        **{name: ROOT / "tools" / f"{name}_pin.json" for name in (TOOLKIT, *DEV_TOOLS)}}
#: THE DEVBOX IMAGE (`//:image`, docs/setup.md#the-devbox-image): the name Docker knows it by, and the digest the tree
#: names, which a commit changing it carries a passing container check for (KERNEL_STANDARDS §23 (3))
DEVBOX_REPO = "rola-devbox"
DEVBOX_DIGEST = ROOT / "tools" / "devbox_image.json"
DEVBOX_DIGEST_ROLE = ("the devbox image the tree names (`//:image`'s content digest), which `bazel run //:container_check` "
                      "writes when green; the commit gate refuses a tree whose image is another, or a change to this "
                      "digest without a passing check for it (KERNEL_STANDARDS 23 (3))")
#: the clean room's container and the prefix of its volumes
CHECK_VOLUMES = "rola-devbox-check"
#: the `rola_results` location of the container checks, one record per image digest
RESULTS_ENVIRONMENT = "environment"


def _bazelisk_version(path: str) -> str:
    """The pin's version string when the binary is the pinned release, by its digest (it prints no version alone)."""
    pin = json.loads(PINS["bazelisk"].read_text())
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return pin["version_string"] if got == pin["asset_sha256"] else f"bazelisk sha256 {got}"


#: a pinned tool's own version reader, where its `--version` cannot say its pin's version string
VERSION_OF = {"bazelisk": _bazelisk_version}


def ensure_pinned(name: str) -> None:
    """A pin installed where it is missing or does not report its pin's version."""
    if _pinned_version(name, str(dev_config.pinned(name))) != json.loads(PINS[name].read_text())["version_string"]:
        install_pinned(name)


def _write_section(section: str, values: dict, force: bool) -> None:
    """Merge `values` into the section's file: keys already there are kept unless `force`."""
    path = dev_config.directory() / f"{section}.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update({k: v for k, v in values.items() if force or k not in merged})
    if not merged:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    dev_config.reload()
    print(f"wrote {path}")


#: a release host's transient failures (a gateway timeout, a dropped connection): the same artifact asked again, from
#: the release's URL and then its mirrors (`mirror_urls`), the digest still the check -- see docs/internals/tools/dev.md
DOWNLOAD_ATTEMPTS = 6


def _download(urls: list[str], to: Path) -> None:
    """Fetch the first of `urls` that answers to `to`, asking again after a server error or a dropped connection (2, 4,
    8, ... s apart). An API asset URL serves the file's bytes under `Accept: application/octet-stream`."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
                with urllib.request.urlopen(request, timeout=300) as response, open(to, "wb") as out:
                    shutil.copyfileobj(response, out)
                return
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError, TimeoutError) as ex:
                if isinstance(ex, urllib.error.HTTPError) and ex.code < 500:
                    raise
                print(f"{url}: {ex}; asking again ({attempt}/{DOWNLOAD_ATTEMPTS})", file=sys.stderr)
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(2 ** attempt)
    raise SystemExit(f"none of {urls} answered in {DOWNLOAD_ATTEMPTS} attempts")


def install_bazelisk() -> None:
    """The pinned bazelisk under `host.tools_dir` and `bazel` linked to it: the machine's one step outside the graph
    (`tools/bootstrap.py`, on a host and in the devbox alike), since every other command is a `bazel run`."""
    ensure_pinned("bazelisk")
    link_bazel()


def link_bazel() -> None:
    """`toolchain.bazel` linked to the pinned bazelisk: `bazel` is the command the docs name, and only the pinned
    launcher reads `.bazeliskrc` and runs the graph's front door. A link it made to another pinned release (under
    `host.tools_dir`) is replaced; anything else at the path is refused, named, never overwritten."""
    link, bazelisk = Path(dev_config.get("toolchain.bazel")), dev_config.pinned("bazelisk")
    if link.is_symlink() and link.resolve() == bazelisk.resolve():
        return
    tools = Path(dev_config.get("host.tools_dir")).resolve()
    if link.exists() or link.is_symlink():
        if not (link.is_symlink() and link.resolve().is_relative_to(tools)):
            raise SystemExit(f"{link} is not the bootstrap's link to a pinned bazelisk ({link.resolve()}): move it, or set "
                             f"toolchain.bazel to a free path on PATH, then python3 tools/bootstrap.py")
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(bazelisk)
    print(f"linked {link} -> {bazelisk}")


def _verify(what: str, path: Path, sha256: str) -> None:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha256:
        raise SystemExit(f"{what}: sha256 {got} is not the pin's {sha256}")


def _without_top(component: str):
    """The extraction filter of a component's archive: a member under the pin's folder instead of the archive's own (the
    archive's folder dropped from its name, and from a hard link's target, which names an archive path), its LICENSE
    at `licenses/<component>/LICENSE`, since every component carries one at the same path -- the graph's copy's layout
    (rola-devtools' environment extension's `cuda_toolkit` tag, `@cuda_tools`)."""
    def keep(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
        _, _, name = member.name.removeprefix("./").partition("/")
        if not name:
            return None
        name = f"licenses/{component}/LICENSE" if name == "LICENSE" else name
        link = member.linkname.removeprefix("./").partition("/")[2] if member.islnk() else member.linkname
        return tarfile.tar_filter(member.replace(name=name, linkname=link, deep=False), dest)
    return keep


def _install_components(name: str, pin: dict) -> None:
    """A toolkit of NVIDIA's redistributable components: the manifest by the pin's digest, each component by the digest
    the manifest lists, all unpacked into the pin's one folder, then the pin's links (`lib64` for nvcc.profile). Unpacked
    beside the folder and moved into place, so a folder there is always a whole toolkit."""
    tools = Path(dev_config.get("host.tools_dir"))
    dest, staging = tools / pin["archive_root"], tools / f".{pin['archive_root']}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix=f"rola_{name}_") as tmp:
        index = Path(tmp) / "manifest.json"
        _download([pin["manifest_url"]], index)
        _verify(f"{name} manifest", index, pin["manifest_sha256"])
        manifest, base = json.loads(index.read_text()), pin["manifest_url"].rsplit("/", 1)[0]
        for component in pin["components"]:
            entry = manifest[component]["linux-x86_64"]
            archive = Path(tmp) / Path(entry["relative_path"]).name
            _download([f"{base}/{entry['relative_path']}"], archive)
            _verify(component, archive, entry["sha256"])
            with tarfile.open(archive) as tf:
                tf.extractall(staging, filter=_without_top(component))
            archive.unlink()
    for link, target in pin["links"].items():
        (staging / link).symlink_to(target)
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)


def install_pinned(name: str) -> str:
    """Download the pinned release, verify its sha256 (and the binary's, where the pin has one), and unpack the whole
    release under `dest` -- its layout is part of the tool (the toolkit's nvcc finds cicc and ptxas by its own place). A release is a tarball,
    a bare binary (`asset_kind`) installed under its release's name, or a manifest's components (`components`)."""
    pin = json.loads(PINS[name].read_text())
    root, rel = pin["archive_root"], pin["binary_path"]
    dest, binary = Path(dev_config.get("host.tools_dir")), dev_config.pinned(name)
    dest.mkdir(parents=True, exist_ok=True)
    if "components" in pin:
        _install_components(name, pin)
    else:
        with tempfile.TemporaryDirectory(prefix=f"rola_{name}_") as tmp:
            asset = Path(tmp) / pin["release_asset"]
            _download([pin["release_url"], *pin.get("mirror_urls", [])], asset)
            _verify(f"{name} release", asset, pin["asset_sha256"])
            if pin.get("asset_kind") == "binary":
                #: a bare binary: installed in its pin's folder, as an archive's tree is
                binary.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asset, binary)
                binary.chmod(0o755)
            else:
                with tarfile.open(asset) as tf:
                    #: an archive with no folder of its own unpacks into its pin's, so every version has one
                    own = all(n == root or n.startswith(root + "/") for n in tf.getnames())
                    tf.getmember(f"{root}/{rel}" if own else rel)
                    tf.extractall(dest if own else dest / root, filter="tar")
    if "binary_sha256" in pin and hashlib.sha256(binary.read_bytes()).hexdigest() != pin["binary_sha256"]:
        raise SystemExit(f"{name}: {binary} sha256 is not the pin's {pin['binary_sha256']}")
    print(f"installed {pin['version_string']} -> {binary}")
    return str(binary)


def site_packages(venv: Path) -> Path:
    """A venv's own site-packages, as its interpreter reports it: the Python running this tool may be another version."""
    out = subprocess.run([str(venv / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"{venv}: no interpreter answers for its site-packages ({out.stderr.strip()[-200:]})")
    return Path(out.stdout.strip())


#: the shared checkouts every venv here imports, by a one-line `.pth` naming each: the store's `rola_results` and
#: rola-devtools' `rola_devtools`
LINKED = {"rola_results.pth": "store.root", "rola_devtools.pth": "workspace.devtools"}


def link_packages() -> list[Path]:
    """`rola_results` and `rola_devtools` importable from every interpreter that runs here, by the `LINKED` `.pth` files:
    each checkout's own venv (`checkout_venv`: it names the environment as a plain path, which reads none of its .pth
    files). Never the machine's own python: nothing here runs from it."""
    sites = [site_packages(v) for v in (checkout_venv(c) for c in _worktrees()) if v is not None]
    for site in sites:
        site.mkdir(parents=True, exist_ok=True)
        for pth, key in LINKED.items():
            (site / pth).write_text(f"{dev_config.get(key)}\n")
    return sites


def wire_hooks(checkout: Path) -> None:
    """The commit gate as a property of the checkout: `core.hooksPath` set per worktree to the tracked hooks (the gate
    is `bazel test //:gate`), and `rola.venv` naming the checkout's own venv, which the post-commit review runs from, so
    no hook reaches the host's python. Refuses a checkout with no venv of its own."""
    venv = checkout_venv(checkout)
    if venv is None:
        raise SystemExit(f"{checkout} has no venv of its own: bazel run //:dev_env in it")
    subprocess.run(["git", "-C", str(checkout), "config", "extensions.worktreeConfig", "true"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "--worktree", "core.hooksPath", "tools/git-hooks"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "--worktree", "rola.venv", str(venv)], check=True)


#: the marker of a venv `dev-env` wrote: its `.pth` naming the graph's environment
ENV_PTH = "zz_rola_env.pth"


def dev_venv(checkout: Path) -> Path:
    """The venv `dev-env` writes for a checkout: `venv-<checkout>` beside it."""
    return checkout.parent / f"venv-{checkout.name}"


def checkout_venv(checkout: Path) -> Path | None:
    """A checkout's own venv, which `dev-env` wrote (its `ENV_PTH`), or None until `bazel run //:dev_env` writes it."""
    own = dev_venv(Path(checkout))
    return own if list(own.glob(f"lib/python*/site-packages/{ENV_PTH}")) else None


#: THE AGENT SKILL ROUTER (KERNEL_STANDARDS section 31): the user-level bootstrap that runs the touched checkout's
#: router from a session started anywhere -- the repo's copy installed as the user's, its PreToolUse entry in their
#: settings
ROUTER = ROOT / ".claude" / "hooks" / "skill_router_bootstrap.py"
ROUTER_HOOKS = {"PreToolUse": {"matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash", "hooks": [
                    {"type": "command", "command": 'python3 "$HOME/.claude/hooks/skill_router.py"', "timeout": 5}]},
                "SessionStart": {"matcher": "compact|resume", "hooks": [
                    {"type": "command", "command": 'python3 "$HOME/.claude/hooks/skill_router.py" --forget', "timeout": 5}]}}


def router_state() -> tuple[bool, bool]:
    """(the user's bootstrap is this checkout's, the user's settings run it)."""
    home = Path.home() / ".claude"
    installed = home / "hooks" / "skill_router.py"
    try:
        settings = json.loads((home / "settings.json").read_text())
    except (OSError, ValueError):
        settings = {}
    wired = all(any("skill_router.py" in hook.get("command", "")
                    for group in settings.get("hooks", {}).get(event, []) for hook in group.get("hooks", []))
                for event in ROUTER_HOOKS)
    return installed.is_file() and installed.read_bytes() == ROUTER.read_bytes(), wired


def install_router() -> None:
    """The bootstrap installed where it differs and its hook entry added where it is absent, on a machine where Claude
    Code keeps a user configuration (`~/.claude`; the image has none); every other user setting is kept, and a settings
    file that does not parse is refused, never rewritten."""
    home = Path.home() / ".claude"
    if not home.is_dir():
        print("agent skill router: no Claude Code configuration here, nothing to install")
        return
    current, wired = router_state()
    if not current:
        (home / "hooks").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROUTER, home / "hooks" / "skill_router.py")
        print(f"agent skill router: installed {home / 'hooks' / 'skill_router.py'}")
    if not wired:
        path = home / "settings.json"
        try:
            settings = json.loads(path.read_text()) if path.is_file() else {}
        except ValueError as ex:
            raise SystemExit(f"{path} does not parse ({ex}); fix it, then rerun init") from ex
        for event, entry in ROUTER_HOOKS.items():
            groups = settings.setdefault("hooks", {}).setdefault(event, [])
            if not any("skill_router.py" in hook.get("command", "") for group in groups for hook in group.get("hooks", [])):
                groups.append(entry)
        path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"agent skill router: wired into {path}")


def _worktrees() -> list[Path]:
    """Every checkout of this repository: the main one and each linked worktree."""
    listing = subprocess.run(["git", "-C", str(ROOT), "worktree", "list", "--porcelain"], capture_output=True, text=True,
                             check=True).stdout
    return [Path(line.split(" ", 1)[1]) for line in listing.splitlines() if line.startswith("worktree ")]


def _refuse_unwired() -> None:
    """The repository's shared hook refuses: a checkout reaches it only when its own gate is not wired, and a commit
    must never pass unchecked or through a hook that runs the host's python."""
    common = Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                 capture_output=True, text=True, check=True).stdout.strip())
    hook = common / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/usr/bin/env bash\n# Written by rola's tools/dev.py init: a checkout reaches this hook only when "
                    "its own commit gate\n# (core.hooksPath -> tools/git-hooks) is not wired, so it refuses.\n"
                    'echo "pre-commit: refusing -- this checkout\'s commit gate is not wired: bazel run //:dev_env" >&2\n'
                    "exit 1\n")
    hook.chmod(0o755)


def gate_venv(checkout: Path | str) -> str:
    """The venv a checkout's commit hooks run from (`git config rola.venv`), or "" when none is named."""
    done = subprocess.run(["git", "-C", str(checkout), "config", "--get", "rola.venv"], capture_output=True, text=True)
    return done.stdout.strip()


def cmd_init(a) -> int:
    for section, values in dev_config.detect().items():
        _write_section(section, values, a.force)

    for name in (TOOLKIT, *DEV_TOOLS):
        ensure_pinned(name)
    for site in link_packages():
        print(f"rola_results, rola_devtools: linked into {site}")
    for checkout in _worktrees():
        if not os.access(checkout / "tools" / "git-hooks" / "pre-commit", os.X_OK):
            continue
        if checkout_venv(checkout) is None:
            print(f"{checkout}: no venv of its own, its gate left as it is -- bazel run //:dev_env in it")
        else:
            wire_hooks(checkout)
    _refuse_unwired()
    install_router()
    if dev_config.get("clock.ghz") is None:
        print("clock: unset -- this host measures UNLOCKED until `bazel run //:clock -- --mhz <MHz>` runs")
    return cmd_check(a)


def _pinned_version(name: str, path: str) -> str:
    """What the binary at `path` reports, as its pin's `version_string` is written: its own reader's answer, else the
    pin's text when a line of its `--version` says it."""
    try:
        if name in VERSION_OF:
            return VERSION_OF[name](path)
    except (OSError, RuntimeError):
        return ""
    want = json.loads(PINS[name].read_text())["version_string"]
    return want if want in _version(path).splitlines() else ""


def _version(tool: str) -> str:
    try:
        return subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except OSError:
        return ""


def cmd_check(a) -> int:
    rows: list[tuple[bool, str, str]] = []

    def row(ok: bool, what: str, fix: str = "") -> None:
        rows.append((ok, what, fix))

    try:
        dev_config.load()
        row(True, f"config {dev_config.directory()}")
    except SystemExit as ex:
        row(False, f"config: {ex}", "fix the named file")
        return _report(rows)
    want = json.loads(PINS[TOOLKIT].read_text())["version_string"]
    row(_pinned_version(TOOLKIT, str(dev_config.pinned(TOOLKIT))) == want,
        f"toolchain cuda: {dev_config.pinned(TOOLKIT).parents[1]} is {want}", "bazel run //:init")
    for name in DEV_TOOLS:
        want = json.loads(PINS[name].read_text())["version_string"]
        row(_pinned_version(name, str(dev_config.pinned(name))) == want, f"devbox {name}: {dev_config.pinned(name)} is {want}",
            "bazel run //:init")
    bazelisk = dev_config.pinned("bazelisk")
    pin = json.loads(PINS["bazelisk"].read_text())["version_string"]
    row(_pinned_version("bazelisk", str(bazelisk)) == pin, f"bazelisk: {bazelisk} is {pin}", "python3 tools/bootstrap.py")
    #: bazelisk puts the front door's directory first on PATH for what it runs (`bazel run //:check`): the user's `bazel`
    #: is the first one past it
    front_door = (ROOT / "bazel").resolve()
    on_path = shutil.which("bazel", path=os.pathsep.join(d for d in os.environ.get("PATH", "").split(os.pathsep)
                                                         if d and Path(d).resolve() != front_door))
    row(on_path is not None and Path(on_path).resolve() == bazelisk.resolve(),
        f"`bazel` on PATH is the pinned bazelisk: {on_path}",
        f"python3 tools/bootstrap.py (links toolchain.bazel, {dev_config.get('toolchain.bazel')}); its directory on PATH")
    readable = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"], capture_output=True).returncode == 0
    hooks = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "core.hooksPath"], capture_output=True, text=True)
    own = checkout_venv(ROOT)
    row(not readable or (hooks.stdout.strip() == "tools/git-hooks" and own is not None and gate_venv(ROOT) == str(own)),
        (f"commit gate wired in {ROOT.name} (`bazel test //:gate`), its review run from {own}" if own else
         f"{ROOT.name} has no venv of its own for its commit gate") if readable else
        f"{ROOT}: no git repository visible here; commits are made where it is",
        "bazel run //:dev_env" if readable and own is None else "bazel run //:init")
    current, wired = router_state()
    if (Path.home() / ".claude").is_dir():
        row(current and wired, "agent skill router: " + ("installed and current" if current else "missing or stale")
            + ("" if wired else ", not in the user's settings"), "bazel run //:init")
    else:
        row(True, "agent skill router: no Claude Code configuration here")
    for key in ("host.lock_dir", "host.scratch"):
        path = Path(dev_config.get(key))
        path.mkdir(parents=True, exist_ok=True)
        row(os.access(path, os.W_OK), f"{key} writable: {path}")
    row(os.access(Path(dev_config.get("host.gpu_lock")).parent, os.W_OK), f"host.gpu_lock: {dev_config.get('host.gpu_lock')}")
    ghz = dev_config.get("clock.ghz")
    row(True, f"clock: {'locks at ' + str(ghz) + ' GHz' if ghz else 'unlocked (rows carry their measured clock)'}")
    suite = Path(dev_config.get("workspace.suite"))
    row((suite / "declare.py").is_file() and (suite / "rola_bench").is_dir(),
        f"workspace.suite holds the measurement suite's root: {suite}",
        "point workspace.suite at a rola-bench checkout carrying its root declare.py")
    row(Path(dev_config.get("workspace.worktrees")).is_dir(), f"workspace.worktrees: {dev_config.get('workspace.worktrees')}")
    root = Path(dev_config.get("store.root"))
    row((root / "rola_results" / "store.py").is_file(), f"store.root is a rola-results checkout: {root}",
        "clone rola-results to store.root, or point store.root at a checkout")
    devtools = Path(dev_config.get("workspace.devtools"))
    row((devtools / "rola_devtools" / "mirror.py").is_file(), f"workspace.devtools is a rola-devtools checkout: {devtools}",
        "clone rola-devtools to workspace.devtools, or point workspace.devtools at a checkout")
    #: the co-developed checkouts import in the checkout's own venv (its hooks, `container`)
    if readable and own is not None:
        python = own / "bin" / "python"
        found = subprocess.run([str(python), "-c", "import rola_results, rola_devtools; "
                                "print(rola_results.ROOT); print(rola_devtools.__file__)"], capture_output=True, text=True)
        lines = found.stdout.split()
        row(lines[:1] == [str(root / "records")], f"rola_results imports from store.root in {python}",
            "bazel run //:init")
        row(lines[1:2] == [str(devtools / "rola_devtools" / "__init__.py")],
            f"rola_devtools imports from workspace.devtools in {python}", "bazel run //:init")
    return _report(rows)


def _report(rows: list[tuple[bool, str, str]]) -> int:
    for ok, what, fix in rows:
        print(f"{'ok ' if ok else 'RED'}  {what}" + ("" if ok or not fix else f"  -- {fix}"))
    red = sum(1 for ok, _, _ in rows if not ok)
    print(f"dev check: {'green' if not red else f'{red} red'}")
    return 1 if red else 0


def cmd_show(_a) -> int:
    for section, keys in dev_config.load().items():
        for name, (value, source) in keys.items():
            print(f"{section}.{name} = {json.dumps(value)}  [{source}]")
    return 0


def cmd_get(a) -> int:
    print(json.dumps(dev_config.get(a.key)).strip('"'))
    return 0


def _windows(rel: str) -> Path:
    tool = dev_config.windows_tool(rel)
    if tool is None or not tool.exists():
        raise SystemExit(f"host.windows_system32 does not hold {rel}; bazel run //:init detects it under WSL")
    return tool


def _clock_wsl(mhz: int) -> dict:
    """An elevated scheduled task a non-administrator may start: registered once, one administrator prompt."""
    smi, ps = _windows("nvidia-smi.exe"), _windows("WindowsPowerShell/v1.0/powershell.exe")
    appdata = subprocess.run([str(ps), "-NoProfile", "-Command", "$env:LOCALAPPDATA"],
                             capture_output=True, text=True, check=True).stdout.strip()
    home = Path(f"/mnt/{appdata[0].lower()}{appdata[2:].replace(chr(92), '/')}") / "rola"
    home.mkdir(parents=True, exist_ok=True)

    def to_windows(path: Path) -> str:
        s = str(path)
        return f"{s[5].upper()}:{s[6:]}".replace("/", "\\")

    lines = ["$ErrorActionPreference = 'Stop'"]
    for name, arg in (("gpu-lock", f"-lgc {mhz},{mhz}"), ("gpu-unlock", "-rgc")):
        vbs = home / f"{name}.vbs"
        #: the launcher exits with nvidia-smi's own code, so the task's last result is the driver's answer
        vbs.write_text(f'WScript.Quit CreateObject("WScript.Shell").Run("C:\\Windows\\System32\\nvidia-smi.exe {arg}", 0, '
                       f'True)\r\n')
        lines.append(f"schtasks /create /tn '{name}' /sc once /st 00:00 /rl highest /f /tr 'wscript.exe \"{to_windows(vbs)}\"'")
    script = home / "register-gpu-clock-tasks.ps1"
    script.write_text("\r\n".join(lines) + "\r\n")
    print(f"registering the tasks (one administrator prompt): {to_windows(script)}")
    subprocess.run([str(ps), "-NoProfile", "-Command", f"Start-Process powershell -Verb RunAs -Wait -ArgumentList "
                    f"'-NoProfile -ExecutionPolicy Bypass -File \"{to_windows(script)}\"'"], check=True)
    return {"ghz": mhz / 1000.0, "lock": _task_run(ps, "gpu-lock"), "unlock": _task_run(ps, "gpu-unlock"),
            "read": [str(smi), "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


def _task_run(ps: Path, name: str) -> list[str]:
    """Start a scheduled task and return when it has FINISHED, failing with its result: `schtasks /run` returns before the
    task runs, so a lock read at once raced it and one run's unlock could land after the next run's lock."""
    script = (f"$t = '{name}'; Start-ScheduledTask -TaskName $t; $w = [Diagnostics.Stopwatch]::StartNew(); "
              "while ((Get-ScheduledTask -TaskName $t).State -ne 'Running' -and $w.ElapsedMilliseconds -lt 3000) "
              "{ Start-Sleep -Milliseconds 50 }; "
              "while ((Get-ScheduledTask -TaskName $t).State -eq 'Running') { Start-Sleep -Milliseconds 50 }; "
              "$r = (Get-ScheduledTaskInfo -TaskName $t).LastTaskResult; "
              "if ($r -ne 0) { [Console]::Error.WriteLine(\"$t exited $r (nvidia-smi's code)\"); exit 1 }")
    return [str(ps), "-NoProfile", "-NonInteractive", "-Command", script]


def _clock_linux(mhz: int) -> dict:
    """`nvidia-smi -lgc` needs root: the commands run through `sudo -n`, and the sudoers rule is printed."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise SystemExit("no nvidia-smi on PATH")
    print(textwrap.dedent(f"""
        the lock runs through sudo without a password; give it the rule (visudo):
            {os.environ.get("USER", "USER")} ALL=(root) NOPASSWD: {smi} -lgc *, {smi} -rgc
        """))
    return {"ghz": mhz / 1000.0, "lock": ["sudo", "-n", smi, "-lgc", f"{mhz},{mhz}"],
            "unlock": ["sudo", "-n", smi, "-rgc"], "read": [smi, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


def cmd_clock(a) -> int:
    """Lock the SM clock, read the driver's clock under the lock, unlock: the round trip the harness makes, proven."""
    cfg = _clock_wsl(a.mhz) if dev_config.get("host.windows_system32") else _clock_linux(a.mhz)
    subprocess.run(cfg["lock"], check=True, capture_output=True, timeout=60)
    time.sleep(2)
    read = subprocess.run(cfg["read"], capture_output=True, text=True, timeout=60).stdout.strip()
    subprocess.run(cfg["unlock"], check=True, capture_output=True, timeout=60)
    if not read or abs(int(read.split()[0]) - a.mhz) > 0.01 * a.mhz:
        raise SystemExit(f"the lock did not hold: the driver reports {read!r} MHz, wanted {a.mhz}")
    print(f"lock proven: {read} MHz under lock, released")
    _write_section("clock", {k: cfg[k] for k in ("ghz", "lock", "unlock")}, force=True)
    return 0


def cmd_dev_env(a) -> int:
    """THIS CHECKOUT'S VENV OVER THE GRAPH'S ENVIRONMENT (`bazel run //:dev_env`): `venv-<checkout>` beside the checkout
    on the environment's interpreter. Its site names the checkout, the sibling checkouts this machine co-develops
    (`LINKED`) and the environment's packages, in that order, and moves bytecode out of the checkout
    (`sys.pycache_prefix`); the checkout's commit hooks then run from it."""
    checkout = Path(a.checkout).resolve()
    venv = dev_venv(checkout)
    if venv.exists():
        if not list(venv.glob("lib/python*/site-packages/zz_rola_*.pth")):
            raise SystemExit(f"refusing: {venv} exists and is not a venv this tool wrote")
        shutil.rmtree(venv)
    subprocess.run([a.uv, "venv", "--no-config", "--python", a.python, str(venv)], check=True)
    site = site_packages(venv)
    (site / f"aa_rola_{checkout.name}.pth").write_text(f"{checkout}\n")
    for pth, key in LINKED.items():
        (site / pth).write_text(f"{dev_config.get(key)}\n")
    (site / ENV_PTH).write_text(f"{site_packages(Path(a.env))}\n")
    prefix = Path(dev_config.get("host.build_cache")) / "pycache"
    (site / "zz_rola_pycache.pth").write_text(f"import sys; sys.pycache_prefix = {str(prefix)!r}\n")

    #: the environment's commands (ruff, pytest, ast-grep, ...) on the venv's own bin, run by its interpreter: a Python
    #: entry script copied with this venv's interpreter on its first line, a native one linked; never pip, which would
    #: install into a venv that holds nothing of its own
    for tool in sorted((Path(a.env) / "bin").iterdir()):
        target = venv / "bin" / tool.name
        if target.exists() or tool.name.startswith("pip") or not tool.is_file():
            continue
        first, _, rest = tool.read_bytes().partition(b"\n")
        if first.startswith(b"#!") and b"python" in first:
            target.write_bytes(f"#!{venv / 'bin' / 'python'}".encode() + b"\n" + rest)
            target.chmod(0o755)
        else:
            target.symlink_to(tool)
    #: and the devbox tools, by their pins' commands
    for name in DEV_TOOLS:
        if not dev_config.pinned(name).is_file():
            raise SystemExit(f"{venv}: the devbox's {name} is not installed at {dev_config.pinned(name)}: bazel run //:init")
        (venv / "bin" / json.loads(PINS[name].read_text())["command"]).symlink_to(dev_config.pinned(name))

    wire_hooks(checkout)
    found = subprocess.run([str(venv / "bin" / "python"), "-c", "import json, rola, rola_devtools, rola_results, torch; "
                            "print(json.dumps({m.__name__: m.__file__ for m in (rola, rola_devtools, rola_results, torch)}))"],
                           capture_output=True, text=True, cwd=venv.parent)
    if found.returncode:
        raise SystemExit(f"{venv}: its interpreter cannot import the tree's packages:\n{found.stderr[-2000:]}")
    origins = json.loads(found.stdout)
    want = {"rola": checkout, "rola_devtools": Path(dev_config.get("workspace.devtools")),
            "rola_results": Path(dev_config.get("store.root")), "torch": site_packages(Path(a.env))}
    wrong = [f"{name} from {origins[name]}, not {root}" for name, root in want.items()
             if not Path(origins[name]).resolve().is_relative_to(root.resolve())]
    if wrong:
        raise SystemExit(f"{venv}: " + "; ".join(wrong))
    print(f"venv {venv}: rola from {checkout}, the environment {a.env}; the commit hooks run from it")
    return 0


def cmd_worktree(a) -> int:
    """A worktree in `workspace.worktrees` at `sha`, then its venv and commit gate (`bazel run //:dev_env`) and its
    extension (`bazel run //:library`) from the graph, which serves a commit's library from its cache when any
    checkout built it."""
    main = Path(subprocess.run(["git", "-C", str(ROOT), "worktree", "list", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.split("\n", 1)[0].split(" ", 1)[1])
    worktree = Path(dev_config.get("workspace.worktrees")) / a.name
    for p in (worktree, dev_venv(worktree)):
        if p.exists():
            raise SystemExit(f"refusing: {p} already exists")
    subprocess.run(["git", "-C", str(main), "worktree", "add", str(worktree), a.sha], check=True)
    if not os.access(worktree / "tools" / "git-hooks" / "pre-commit", os.X_OK):
        raise SystemExit(f"refusing: {worktree}/tools/git-hooks/pre-commit is missing or not executable")
    for target in ("//:dev_env", "//:library"):
        subprocess.run(["bazel", "run", target], cwd=worktree, check=True)
    venv = dev_venv(worktree)
    subprocess.run([str(venv / "bin" / "python"), "-c", "import importlib, rola; from rola.ops._ext import PLUGIN, extension; "
                    "extension(); print('rola:', rola.__file__, 'archs:', "
                    "importlib.import_module(PLUGIN + '._build_config').BUILD_CONFIG['archs'])"], cwd=worktree, check=True)
    print(f"worktree {worktree}  venv {venv}")
    return 0


def _image(layout: str) -> tuple[str, str]:
    """The devbox image's identity, its content digest (`//:image`'s manifest), and the name Docker knows it by."""
    digest = oci_layout.digest(Path(layout))
    return digest, f"{DEVBOX_REPO}:{oci_layout.tag(digest)}"


def _load(tarball: str, tag: str) -> None:
    """The graph's image handed to Docker (it only runs it), tagged by its digest."""
    loaded = subprocess.run(["docker", "load", "-i", tarball], capture_output=True, text=True)
    if loaded.returncode:
        raise SystemExit(f"docker load: {loaded.stderr.strip()[-400:]}")
    subprocess.run(["docker", "tag", f"{DEVBOX_REPO}:latest", tag], check=True)


#: THE ONE LOCK PATH: a container's lock directory and GPU lock, the dev config's defaults, where the host's are mounted
LOCK_PATH = Path(dev_config.SCHEMA["host"]["lock_dir"].default)
LOCK_GPU = Path(dev_config.SCHEMA["host"]["gpu_lock"].default)


def _mounts(named: Path, digest: str) -> list[str]:
    """What a container started here shares with the host, as `source:target[:ro]`: the host's lock directory at the
    one lock path (`host.lock_dir`'s default, which a container's own config keeps), so one lock governs the GPU both
    use; the image's digest, written to `named`, where `dev_config.devbox_image()` reads it; and under WSL the driver's
    libraries at their own path. Nothing else: its checkouts and installs are its own."""
    locks, gpu = Path(dev_config.get("host.lock_dir")), Path(dev_config.get("host.gpu_lock"))
    if gpu != locks / LOCK_GPU.name:
        raise SystemExit(f"host.gpu_lock {gpu} must be {locks / LOCK_GPU.name} for a container to share it: the "
                         "container's locks are one directory")
    #: a missing source would be made by the daemon, as root
    locks.mkdir(parents=True, exist_ok=True)
    named.write_text(digest + "\n")
    mounts = [f"{locks}:{LOCK_PATH}", f"{named}:{dev_config.CONTAINER_IMAGE}:ro"]
    wsl_lib = dev_config.get("host.wsl_lib")
    #: the whole WSL directory, not only lib/: its libcuda is a loader that opens the host's driver store beside it
    return mounts + ([f"{Path(wsl_lib).parent}:{Path(wsl_lib).parent}:ro"] if wsl_lib else [])


#: the checkouts a container clones, by the dev config key that names each on the host, and the folder it clones into
#: (beside the checkout, where the reader's defaults look); this checkout's own is `CONTAINER_CHECKOUT`
CLONED = {"workspace.devtools": "rola-devtools", "store.root": "rola-results", "workspace.suite": "rola-bench"}
CONTAINER_CHECKOUT = Path("/workspace/rola")
#: the checkout's venv in the container, by the rule a host's follows
CONTAINER_PYTHON = dev_venv(CONTAINER_CHECKOUT) / "bin" / "python"

#: THE CLEAN ROOM (`bazel run //:container_check`): the host's loop, step by step, in a container on fresh volumes of
#: its own, each step timed. init's own check runs before the checkout's venv exists, so its gate row is the one red it
#: may report (any other fails the step); `check`, after `dev_env`, is the verdict on the machine.
CONTAINER_LOOP = {
    "clone the checkouts": f"""set -e
git config --global --add safe.directory '*'
mkdir -p {CONTAINER_CHECKOUT.parent}/worktrees
for name in {' '.join([*CLONED.values(), CONTAINER_CHECKOUT.name])}; do
  git clone -q "/mnt/git/$name" "{CONTAINER_CHECKOUT.parent}/$name"
  git -C "{CONTAINER_CHECKOUT.parent}/$name" checkout -q "$(cat "/mnt/git/$name.sha")"
done
if [ -s /mnt/git/{CONTAINER_CHECKOUT.name}.staged ]; then
  git -C {CONTAINER_CHECKOUT} apply --index /mnt/git/{CONTAINER_CHECKOUT.name}.staged
fi""",
    "bootstrap": "python3 tools/bootstrap.py",
    "init": """bazel run //:init > /tmp/init.log 2>&1; rc=$?; cat /tmp/init.log
[ $rc -eq 0 ] || { [ "$(grep -c '^RED' /tmp/init.log)" = 1 ] && grep -q '^RED  .* has no venv of its own' /tmp/init.log; }""",
    "dev_env": "bazel run //:dev_env",
    "check": "bazel run //:check",
    "the commit gate": "bazel test $(bazel query 'tests(//:gate) except //:dev_container')",
    "GPU, torch and its flash backend": f"""{CONTAINER_PYTHON} - <<'PY'
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
assert torch.cuda.is_available(), "no GPU"
q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    F.scaled_dot_product_attention(q, q, q, is_causal=True)
print(torch.cuda.get_device_name(0), torch.__version__)
PY""",
    "a host-held lock is held here": f"""{CONTAINER_PYTHON} - {LOCK_PATH}/rola_container_probe.lock <<'PY'
import fcntl, sys
f = open(sys.argv[1], "w")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("held by the host: one lock governs both")
    sys.exit(0)
print("acquired: the container does not see the host's lock")
sys.exit(1)
PY""",
    "the library builds and imports": f"""set -e
arch=$({CONTAINER_PYTHON} -c 'import torch; print("%d%d" % torch.cuda.get_device_capability())')
bazel run //:library --//:archs=$arch --//:carry_arms=0
{CONTAINER_PYTHON} -c 'import rola; from rola_cu13._build_config import BUILD_CONFIG as c; print("built", c["archs"])'""",
}
#: the steps a fresh machine takes before it can work: the cold start a rented box pays
COLD_START = ("clone the checkouts", "bootstrap", "init", "dev_env")


def _git_mounts(staging: Path) -> list[str]:
    """Each checkout's git directory, read-only, for the container to clone from, its commit beside it; this checkout's
    staged change too, so the container's tree is the commit as git will write it."""
    mounts = []
    for name, path in [*((folder, Path(dev_config.get(key))) for key, folder in CLONED.items()), (CONTAINER_CHECKOUT.name, ROOT)]:
        common = subprocess.run(["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        (staging / f"{name}.sha").write_text(sha + "\n")
        mounts.append(f"{common}:/mnt/git/{name}:ro")
    staged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--binary"], capture_output=True, check=True)
    (staging / f"{CONTAINER_CHECKOUT.name}.staged").write_bytes(staged.stdout)
    for entry in sorted(staging.iterdir()):
        mounts.append(f"{entry}:/mnt/git/{entry.name}:ro")
    return mounts


def _container_check(digest: str, tag: str) -> int:
    from rola_results import Store, checkout, portable

    volumes = {f"{CHECK_VOLUMES}-home": "/root", f"{CHECK_VOLUMES}-work": str(CONTAINER_CHECKOUT.parent)}
    for volume in volumes:
        subprocess.run(["docker", "volume", "rm", "-f", volume], capture_output=True)
    lock = Path(dev_config.get("host.lock_dir")) / "rola_container_probe.lock"
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, sys; f = open(sys.argv[1], 'w'); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); sys.stdin.read()", str(lock)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    holder.stdout.readline()
    rows = []
    with tempfile.TemporaryDirectory(prefix="rola_check_") as staging:
        named = Path(staging) / "image"
        args = [x for mount in [*_git_mounts(Path(staging)), *_mounts(named, digest)] for x in ("-v", mount)]
        args += [x for volume, at in volumes.items() for x in ("-v", f"{volume}:{at}")]
        subprocess.run(["docker", "rm", "-f", CHECK_VOLUMES], capture_output=True)
        subprocess.run(["docker", "run", "-d", "--name", CHECK_VOLUMES, "--gpus", "all", "--ipc", "host", *args,
                        tag, "sleep", "infinity"],
                       capture_output=True, check=True)
        try:
            for what, script in CONTAINER_LOOP.items():
                start = time.monotonic()
                done = subprocess.run(["docker", "exec", "-w", "/" if what == "clone the checkouts" else str(CONTAINER_CHECKOUT),
                                       CHECK_VOLUMES, "bash", "-c", script], capture_output=True, text=True)
                seconds = round(time.monotonic() - start, 1)
                tail = [portable(line, ROOT) for line in (done.stdout + done.stderr).strip().splitlines()[-3:]]
                rows.append({"step": what, "ok": done.returncode == 0, "seconds": seconds, "tail": tail})
                print(f"{'ok ' if done.returncode == 0 else 'RED'}  {what} ({seconds} s)" +
                      ("" if done.returncode == 0 else f"  -- {tail}"), flush=True)
                if done.returncode:
                    break
        finally:
            subprocess.run(["docker", "rm", "-f", CHECK_VOLUMES], capture_output=True)
            holder.stdin.close()
            holder.wait()
    ok = len(rows) == len(CONTAINER_LOOP) and all(r["ok"] for r in rows)
    cold = round(sum(r["seconds"] for r in rows if r["step"] in COLD_START), 1)
    result = {"image": tag, "steps": rows, "cold_start_seconds": cold}
    store = Store(RESULTS_ENVIRONMENT)
    if ok:
        store.put({"image_digest": digest}, output=result, provenance=checkout(ROOT))
        DEVBOX_DIGEST.write_text(json.dumps({"digest": digest, "role": DEVBOX_DIGEST_ROLE}, indent=4) + "\n")
        #: a green run's volumes are its own and go with it; a red run's stay for a look until the next run
        for volume in volumes:
            subprocess.run(["docker", "volume", "rm", "-f", volume], capture_output=True)
    else:
        store.put({"image_digest": digest}, error=f"RED: {[r['step'] for r in rows if not r['ok']]}",
                  provenance=checkout(ROOT), **result)
    print(f"container check: {'green' if ok else 'RED'} (cold start {cold} s); recorded under {store.dir}" +
          (f"; wrote {DEVBOX_DIGEST.relative_to(ROOT)}" if ok else ""))
    return 0 if ok else 1


def cmd_container(a) -> int:
    digest, tag = _image(a.layout)
    if a.action == "gate":
        return _container_gate(digest)
    _load(a.tarball, tag)
    if a.action == "check":
        return _container_check(digest, tag)
    local = ROOT / ".devcontainer" / "compose.local.yaml"
    volumes = [f"{DEVBOX_REPO}-home:/root", f"{DEVBOX_REPO}-work:{CONTAINER_CHECKOUT.parent}"]
    doc = {"services": {"rola": {"image": tag, "volumes": [*volumes, *_mounts(local.with_name("devbox_image.local"), digest)]}}}
    local.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {local} ({tag})")
    return 0


def _container_gate(digest: str) -> int:
    """KERNEL_STANDARDS §23 (3): the image the graph builds is the one the tree names (`tools/devbox_image.json`), and a
    commit that changes the name carries a passing container check for the new digest. So a commit that leaves the
    image alone needs no Docker (a rented box is itself a container)."""
    from rola_results import Store
    from rola_results import key as record_key

    def named(rev: str) -> str | None:
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{DEVBOX_DIGEST.relative_to(ROOT)}"],
                              capture_output=True, text=True)
        return json.loads(blob.stdout)["digest"] if blob.returncode == 0 else None

    staged = named("")
    if staged != digest:
        print(f"§23 (3): the graph builds the devbox image {digest[:19]}, and the commit names {str(staged)[:19]}: "
              f"bazel run //:container_check (it writes {DEVBOX_DIGEST.relative_to(ROOT)} when green), then stage it")
        return 1
    if staged != named("HEAD") and not Store.complete(Store(RESULTS_ENVIRONMENT).get(record_key({"image_digest": digest}))):
        print(f"§23 (3): the commit changes the devbox image to {digest[:19]}, which has no passing container check: "
              f"bazel run //:container_check")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write this machine's dev config, install pinned tools, wire hooks, then check")
    p.add_argument("--force", action="store_true", help="overwrite detected values already in the config")
    p.set_defaults(fn=cmd_init)
    sub.add_parser("check", help="verify every setting, tool, pin and lock").set_defaults(fn=cmd_check)
    sub.add_parser("show", help="every resolved setting and its source").set_defaults(fn=cmd_show)
    p = sub.add_parser("get", help="print one value: <section.key>")
    p.add_argument("key")
    p.set_defaults(fn=cmd_get)
    p = sub.add_parser("clock", help="lock this host's SM clock once and record it")
    p.add_argument("--mhz", type=int, required=True, help="a clock the card holds under the kernel's power")
    p.set_defaults(fn=cmd_clock)

    p = sub.add_parser("dev-env", help="this checkout's venv over the graph's environment (`bazel run //:dev_env`)")
    p.add_argument("--checkout", required=True, help="the checkout ($BUILD_WORKSPACE_DIRECTORY)")
    p.add_argument("--env", required=True, help="the graph's environment, its store venv")
    p.add_argument("--python", required=True, help="the environment's interpreter")
    p.add_argument("--uv", required=True, help="the environment's uv")
    p.set_defaults(fn=cmd_dev_env)
    p = sub.add_parser("worktree", help="a worktree with its venv, commit gate and extension, from the graph")
    p.add_argument("name")
    p.add_argument("sha")
    p.set_defaults(fn=cmd_worktree)

    p = sub.add_parser("container", help="the devbox image: its clean-room check, an editor's container, the commit gate")
    p.add_argument("action", choices=("check", "compose", "gate"))
    p.add_argument("--layout", required=True, help="the image's OCI layout (`//:image`)")
    p.add_argument("--tarball", help="check, compose: the image as Docker loads it (`//:image_tarball`)")
    p.set_defaults(fn=cmd_container)
    a = ap.parse_args()
    if a.cmd not in BOOTSTRAP:
        _from_own_venv()
    return a.fn(a)


#: the commands that make or read the environment itself, and so run from whichever python starts them
BOOTSTRAP = ("init", "check", "show", "get", "clock", "dev-env", "worktree")


def _from_own_venv() -> None:
    """Every other command runs from this checkout's own venv, re-executing this file there when another python started
    it: the container reads `rola_results`, which the venv links, never the host's own python."""
    own = checkout_venv(ROOT)
    if own is None:
        raise SystemExit(f"{ROOT.name} has no venv of its own to run this from: bazel run //:dev_env")
    if Path(sys.prefix).resolve() != own.resolve():
        python = own / "bin" / "python"
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
