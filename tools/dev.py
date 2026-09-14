#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEV ENVIRONMENT (KERNEL_STANDARDS §23): the one command that sets up, checks and describes a host or container.

    python tools/dev.py init [--force]          # write this machine's dev config, install the pinned tools, wire hooks
    python tools/dev.py check                   # every value, tool, pin and lock this tree depends on; exit 1 on a red
    python tools/dev.py show                    # every resolved setting and the file it came from
    python tools/dev.py get <section.key>       # one value, for a shell caller (cuda-home resolves the toolkit)
    python tools/dev.py clock --mhz 1665        # lock this host's SM clock once, prove the round trip, record it
    python tools/dev.py worktree <name> <sha> [--build]   # a worktree, its pointer venv, its gate, its extension
    python tools/dev.py container compose|build|check|gate # the dev container: mounts, image, proof, commit gate
    python tools/dev.py store commit -m <msg>   # the measurements store (a rola-results checkout)

The settings live in the dev config directory (`tools/dev_config.py`). Docs: docs/internals/tools/dev.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import
import mold_toolchain  # noqa: E402
import sccache_toolchain  # noqa: E402
import toolchains  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PINS = {"sccache": ROOT / "tools" / "sccache_pin.json", "mold": ROOT / "tools" / "mold_pin.json"}
#: THE ENVIRONMENT'S INPUTS: what the dev image is built from and how it is run. Their hash is the environment key; a
#: commit changing any of them carries a passing container check for its key (KERNEL_STANDARDS §23 (3)).
ENV_INPUTS = ("Dockerfile", ".devcontainer/devcontainer.json", ".devcontainer/compose.yaml", "requirements.lock",
              "tools/sccache_pin.json", "tools/mold_pin.json", "tools/dev.py", "tools/dev_config.py",
              "tools/sccache_toolchain.py", "tools/mold_toolchain.py",
              "tools/toolchains.py",
              *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools" / "toolchains").glob("*.json")))
IMAGE_REPO = "rola-dev"
#: the `rola_results` location of the container checks, one record per environment key
RESULTS_ENVIRONMENT = "environment"
IMAGE_LAYOUT, WSL_LIB_IN_IMAGE = dev_config.IMAGE_LAYOUT, dev_config.WSL_LIB_IN_IMAGE
#: each pinned tool's own version reader, the one its build gate compares against the pin
VERSION_OF = {"sccache": sccache_toolchain.sccache_version, "mold": mold_toolchain.mold_version}


def _write_section(section: str, values: dict, force: bool) -> None:
    """Merge `values` into the section's file: keys already there are kept unless `force`."""
    path = dev_config.directory() / f"{section}.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update({k: v for k, v in values.items() if force or k not in merged})
    if not merged:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    dev_config.load.cache_clear()
    print(f"wrote {path}")


def install_pinned(name: str, dest: Path) -> str:
    """Download the pinned release, verify its tarball sha256 (and the binary's, where the pin has one), and unpack the
    whole release under `dest` -- its layout is part of the tool (mold selects `ld.mold` beside itself)."""
    pin = json.loads(PINS[name].read_text())
    with tempfile.TemporaryDirectory(prefix=f"rola_{name}_") as tmp:
        tar_path = Path(tmp) / pin["release_asset"]
        urllib.request.urlretrieve(pin["release_url"], tar_path)
        got = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if got != pin["tarball_sha256"]:
            raise SystemExit(f"{name}: tarball sha256 {got} is not the pin's {pin['tarball_sha256']}")
        with tarfile.open(tar_path) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith(f"/{name}") and m.isfile())
            dest.mkdir(parents=True, exist_ok=True)
            tf.extractall(dest, filter="tar")
    binary = dest / member.name
    if "binary_sha256" in pin and hashlib.sha256(binary.read_bytes()).hexdigest() != pin["binary_sha256"]:
        raise SystemExit(f"{name}: {binary} sha256 is not the pin's {pin['binary_sha256']}")
    print(f"installed {pin['version_string']} -> {binary}")
    return str(binary)


def base_venv_of(prefix: Path) -> str | None:
    """The shared venv a pointer venv borrows from, or the venv itself when it carries torch directly."""
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    pointer = site / "zz_rola_base.pth"
    if pointer.exists():
        return str(Path(pointer.read_text().strip()).parents[2])
    return str(prefix) if (site / "torch").is_dir() else None


def site_packages(venv: Path) -> Path:
    """A venv's own site-packages, as its interpreter reports it: the Python running this tool may be another version."""
    out = subprocess.run([str(venv / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"{venv}: no interpreter answers for its site-packages ({out.stderr.strip()[-200:]})")
    return Path(out.stdout.strip())


def link_results(image: bool) -> list[Path]:
    """`rola_results` importable from every interpreter that measures here, by a one-line `rola_results.pth` naming
    `store.root`: the base venv (the image's at its build, naming the store's mount point), and on a host each worktree's
    pointer venv (it borrows the base as a plain path, which reads none of the base's .pth files). Never the host's own
    python: nothing here runs from it."""
    sites = [site_packages(Path(dev_config.get("workspace.base_venv")))]
    if not image:
        sites += [site_packages(v) for v in sorted(Path(dev_config.get("workspace.worktrees")).glob("venv-*"))
                  if (site_packages(v) / "zz_rola_base.pth").exists()]
    for site in sites:
        site.mkdir(parents=True, exist_ok=True)
        (site / "rola_results.pth").write_text(f"{dev_config.get('store.root')}\n")
    return sites


#: the other checkouts whose tracked commit hooks run from this machine's base venv: the suite and the store
GATED_CONSUMERS = ("workspace.suite", "store.root")


def wire_hooks(checkout: Path) -> None:
    """The commit gate as a property of the checkout: `core.hooksPath` set per worktree to the tracked hooks, and
    `rola.venv` naming the venv they run from (the lock pins every gate tool), so no hook reaches the host's python."""
    subprocess.run(["git", "-C", str(checkout), "config", "extensions.worktreeConfig", "true"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "--worktree", "core.hooksPath", "tools/git-hooks"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "--worktree", "rola.venv", dev_config.get("workspace.base_venv")],
                   check=True)


def gate_venv(checkout: Path | str) -> str:
    """The venv a checkout's commit hooks run from (`git config rola.venv`), or "" when none is named."""
    done = subprocess.run(["git", "-C", str(checkout), "config", "--get", "rola.venv"], capture_output=True, text=True)
    return done.stdout.strip()


def env_key(read=lambda rel: (ROOT / rel).read_bytes()) -> str:
    """sha256 over the environment's inputs, path and bytes each; `read` takes the staged blob for the commit gate."""
    h = hashlib.sha256()
    for rel in ENV_INPUTS:
        h.update(rel.encode() + b"\0" + read(rel) + b"\0")
    return h.hexdigest()


def lock_pins() -> dict[str, str]:
    """`requirements.lock`'s pins as {normalized name: version}; a URL pin's version is its wheel's."""
    pins = {}
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        if m := re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line):
            name, version = m.groups()
        elif m := re.match(r"^([A-Za-z0-9_.-]+) @ (\S+)", line):
            name, version = m.group(1), urllib.parse.unquote(m.group(2).split("#")[0].rsplit("/", 1)[1]).split("-")[1]
        else:
            continue
        pins[re.sub(r"[-_.]+", "-", name).lower()] = version
    return pins


def venv_drift(python: str) -> list[str]:
    """The lock's pins a venv lacks or carries at another version (extras the lock does not name are not drift)."""
    out = subprocess.run([python, "-m", "pip", "list", "--format=json"], capture_output=True, text=True, check=True)
    have = {re.sub(r"[-_.]+", "-", p["name"]).lower(): p["version"].split("+")[0] for p in json.loads(out.stdout)}
    return sorted(f"{name}=={v}" for name, v in lock_pins().items() if have.get(name) != v.split("+")[0])


def host_toolchain() -> toolchains.Toolchain:
    """The declared toolchain of this machine's configured toolkit."""
    return toolchains.for_ptxas(_version(dev_config.cuda_bin("ptxas")))


def cmd_init(a) -> int:
    if a.image:
        _write_section("toolchain", {"ncu": _newest_ncu()}, force=True)
        for section, values in IMAGE_LAYOUT.items():
            _write_section(section, values, force=True)
        _write_section("environment", {"image_digest": a.env_key}, force=True)
    for section, values in ({} if a.image else dev_config.detect()).items():
        _write_section(section, values, a.force)
    if dev_config.get("toolchain.cuda_home") is None:
        _write_section("toolchain", {"cuda_home": dev_config.cuda_home()}, a.force)
    base = base_venv_of(Path(sys.prefix))
    if base and not a.image:
        _write_section("workspace", {"base_venv": base}, a.force)
    tools_dir = Path(dev_config.get("host.tools_dir"))
    for name in PINS:
        current = dev_config.get(f"toolchain.{name}")
        pin = json.loads(PINS[name].read_text())
        if current is None or _pinned_version(name, current) != pin["version_string"]:
            _write_section("toolchain", {name: install_pinned(name, tools_dir)}, force=True)
    base = dev_config.get("workspace.base_venv")
    drift = venv_drift(str(Path(base) / "bin" / "python")) if base else []
    if drift:
        toolchain = host_toolchain()
        print(f"base venv {base}: installing the lock's {len(drift)} missing pin(s): {' '.join(drift)}")
        subprocess.run(["uv", "pip", "install", "--python", str(Path(base) / "bin" / "python"),
                        "-r", str(ROOT / "requirements.lock"), "--extra-index-url", toolchain.torch_index,
                        "--index-strategy", "unsafe-best-match"], check=True)
    for site in link_results(a.image):
        print(f"rola_results: linked into {site}")
    if not a.image:
        wire_hooks(ROOT)
        for key in GATED_CONSUMERS:
            if (Path(dev_config.get(key)) / ".git").exists():
                subprocess.run(["git", "-C", dev_config.get(key), "config", "rola.venv", dev_config.get("workspace.base_venv")],
                               check=True)
        if dev_config.get("clock.ghz") is None:
            print("clock: unset -- this host measures UNLOCKED until `python tools/dev.py clock --mhz <MHz>` runs")
    return cmd_check(a)


def _newest_ncu() -> str:
    found = sorted(dev_config.ncu_candidates(), key=lambda p: _ncu_version(str(p)))
    if not found:
        raise SystemExit(f"no Nsight Compute CLI under {' or '.join(dev_config.NCU_SEARCH)}")
    return str(found[-1])


def cmd_store(a) -> int:
    if a.action == "commit":
        return subprocess.run([sys.executable, "-m", "rola_results", "commit", "-m", a.message]).returncode


def _pinned_version(name: str, path: str) -> str:
    try:
        return VERSION_OF[name](path)
    except (OSError, RuntimeError):
        return ""


def _version(tool: str) -> str:
    try:
        return subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except OSError:
        return ""


def _ncu_version(ncu: str) -> tuple[int, ...]:
    m = re.search(r"Version (\d+)\.(\d+)", _version(ncu))
    return tuple(int(x) for x in m.groups()) if m else ()


def cmd_check(a) -> int:
    image = getattr(a, "image", False)
    rows: list[tuple[bool, str, str]] = []

    def row(ok: bool, what: str, fix: str = "") -> None:
        rows.append((ok, what, fix))

    try:
        dev_config.load()
        row(True, f"config {dev_config.directory()}")
    except SystemExit as ex:
        row(False, f"config: {ex}", "fix the named file")
        return _report(rows)
    try:
        home = dev_config.cuda_home()
        for tool in ("nvcc", "ptxas", "nvdisasm", "cuobjdump"):
            path = Path(home) / "bin" / tool
            row(path.exists(), f"toolchain {tool}: {path}", "set toolchain.cuda_home to a CUDA toolkit with it")
    except SystemExit as ex:
        row(False, f"toolchain: {ex}", "python tools/dev.py init")
    ncu = dev_config.get("toolchain.ncu")
    row(_ncu_version(ncu) >= (2025, 3), f"toolchain ncu {'.'.join(map(str, _ncu_version(ncu))) or 'missing'}: {ncu}",
        "install Nsight Compute 2025.3+ and set toolchain.ncu")
    for name, gate in (("sccache", sccache_toolchain.gate), ("mold", mold_toolchain.link_flags)):
        try:
            gate()
            row(True, f"toolchain {name}: {dev_config.get(f'toolchain.{name}')} matches its pin")
        except RuntimeError as ex:
            row(False, f"toolchain {name}: {str(ex).splitlines()[0][:140]}", "python tools/dev.py init")
    base = dev_config.get("workspace.base_venv")
    python = Path(base or "") / "bin" / "python"
    row(bool(base) and python.exists(), f"workspace.base_venv: {base}", "python tools/dev.py init")
    if base and python.exists():
        drift = venv_drift(str(python))
        row(not drift, f"base venv carries requirements.lock{': drift ' + ' '.join(drift) if drift else ''}",
            "python tools/dev.py init")
    if image:
        return _report(rows)
    readable = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"], capture_output=True).returncode == 0
    hooks = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "core.hooksPath"], capture_output=True, text=True)
    base = dev_config.get("workspace.base_venv")
    row(not readable or (hooks.stdout.strip() == "tools/git-hooks" and gate_venv(ROOT) == base),
        f"commit gate wired in {ROOT.name}, run from {base}" if readable else f"{ROOT}: no git repository visible here; "
        "commits are made where it is", "python tools/dev.py init")
    #: commits are made on the host (the image sees no repository), so the consumers' gates are the host's to check
    for key in GATED_CONSUMERS if readable else ():
        if (Path(dev_config.get(key)) / ".git").exists():
            row(gate_venv(dev_config.get(key)) == base, f"{key}'s commit gate runs from {base}", "python tools/dev.py init")
    for key in ("host.lock_dir", "host.scratch"):
        path = Path(dev_config.get(key))
        path.mkdir(parents=True, exist_ok=True)
        row(os.access(path, os.W_OK), f"{key} writable: {path}")
    row(os.access(Path(dev_config.get("host.gpu_lock")).parent, os.W_OK), f"host.gpu_lock: {dev_config.get('host.gpu_lock')}")
    ghz = dev_config.get("clock.ghz")
    row(True, f"clock: {'locks at ' + str(ghz) + ' GHz' if ghz else 'unlocked (rows carry their measured clock)'}")
    suite = Path(dev_config.get("workspace.suite"))
    row((suite / "rola_bench" / "measure" / "engine.py").exists(), f"workspace.suite holds the measurement suite: {suite}",
        "point workspace.suite at a rola-bench checkout carrying rola_bench/measure")
    row(Path(dev_config.get("workspace.worktrees")).is_dir(), f"workspace.worktrees: {dev_config.get('workspace.worktrees')}")
    root = Path(dev_config.get("store.root"))
    row((root / "rola_results" / "store.py").is_file(), f"store.root is a rola-results checkout: {root}",
        "clone rola-results to store.root, or point store.root at a checkout")
    found = subprocess.run([str(python), "-c", "import rola_results; print(rola_results.ROOT)"], capture_output=True,
                           text=True)
    row(found.stdout.strip() == str(root / "records"), f"rola_results imports from store.root in {python}",
        "python tools/dev.py init")
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
    print(dev_config.cuda_home() if a.key == "cuda-home" else json.dumps(dev_config.get(a.key)).strip('"'))
    return 0


def _windows(rel: str) -> Path:
    tool = dev_config.windows_tool(rel)
    if tool is None or not tool.exists():
        raise SystemExit(f"host.windows_system32 does not hold {rel}; python tools/dev.py init detects it under WSL")
    return tool


def _clock_wsl(mhz: int) -> dict:
    """An elevated scheduled task a non-administrator may start: registered once, one administrator prompt."""
    smi, tasks, ps = _windows("nvidia-smi.exe"), _windows("schtasks.exe"), _windows("WindowsPowerShell/v1.0/powershell.exe")
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
        vbs.write_text(f'CreateObject("WScript.Shell").Run "C:\\Windows\\System32\\nvidia-smi.exe {arg}", 0, True\r\n')
        lines.append(f"schtasks /create /tn '{name}' /sc once /st 00:00 /rl highest /f /tr 'wscript.exe \"{to_windows(vbs)}\"'")
    script = home / "register-gpu-clock-tasks.ps1"
    script.write_text("\r\n".join(lines) + "\r\n")
    print(f"registering the tasks (one administrator prompt): {to_windows(script)}")
    subprocess.run([str(ps), "-NoProfile", "-Command", f"Start-Process powershell -Verb RunAs -Wait -ArgumentList "
                    f"'-NoProfile -ExecutionPolicy Bypass -File \"{to_windows(script)}\"'"], check=True)
    run = [str(tasks), "/run", "/tn"]
    return {"ghz": mhz / 1000.0, "lock": [*run, "gpu-lock"], "unlock": [*run, "gpu-unlock"],
            "read": [str(smi), "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


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


def cmd_worktree(a) -> int:
    """A worktree in `workspace.worktrees`, its commit gate wired before anything builds, a pointer venv over
    `workspace.base_venv`, and its extension: copied from the main checkout when that checkout is the same commit and
    built, else built in place under the host budget (`--build` always builds)."""
    base = dev_config.get("workspace.base_venv")
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if not base or not (Path(base) / "lib" / py / "site-packages").is_dir():
        raise SystemExit(f"workspace.base_venv {base!r} has no lib/{py}/site-packages; python tools/dev.py init")
    main = Path(subprocess.run(["git", "-C", str(ROOT), "worktree", "list", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.split("\n", 1)[0].split(" ", 1)[1])
    worktree = Path(dev_config.get("workspace.worktrees")) / a.name
    venv = worktree.parent / f"venv-{a.name}"
    for p in (worktree, venv):
        if p.exists():
            raise SystemExit(f"refusing: {p} already exists")
    subprocess.run(["git", "-C", str(main), "worktree", "add", str(worktree), a.sha], check=True)
    if not os.access(worktree / "tools" / "git-hooks" / "pre-commit", os.X_OK):
        raise SystemExit(f"refusing: {worktree}/tools/git-hooks/pre-commit is missing or not executable")
    wire_hooks(worktree)
    subprocess.run(["uv", "venv", "--python", py.removeprefix("python"), "--seed", str(venv)], check=True)
    site = site_packages(venv)
    (site / f"aa_rola_{a.name}.pth").write_text(f"{worktree}\n")
    (site / "zz_rola_base.pth").write_text(f"{site_packages(Path(base))}\n")
    (site / "rola_results.pth").write_text(f"{dev_config.get('store.root')}\n")
    main_so = sorted((main / "rola").glob("_C.cpython-*.so"))
    same = subprocess.run(["git", "-C", str(main), "rev-parse", "HEAD"], capture_output=True, text=True).stdout == \
        subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    install = [str(venv / "bin" / "python"), "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"]
    if main_so and same and not a.build:
        subprocess.run(install, cwd=worktree, env={**os.environ, "ROLA_NO_EXTENSION": "1"}, check=True)
        shutil.copy(main_so[0], worktree / "rola")
    else:
        subprocess.run(install, cwd=worktree, check=True)
    finder = site / "__editable___rola_0_1_0_dev0_finder.py"
    if not finder.exists() or f"'rola': '{worktree}/rola'" not in finder.read_text():
        raise SystemExit(f"refusing: the editable finder {finder} does not map rola to {worktree}/rola")
    subprocess.run([str(venv / "bin" / "python"), "-c", "import rola; from rola._build_config import BUILD_CONFIG; "
                    "print('rola:', rola.__file__, 'archs:', BUILD_CONFIG['archs'])"], cwd=worktree, check=True)
    print(f"worktree {worktree}  venv {venv}")
    return 0


def _mounts() -> list[str]:
    """The host paths the container shares, onto `IMAGE_LAYOUT`, as `source:target[:ro]`: the checkout, the lock
    directories, the store, the worktrees folder, and under WSL the driver's libraries at the path the image's loader
    searches."""
    gpu, image_gpu = Path(dev_config.get("host.gpu_lock")), Path(IMAGE_LAYOUT["host"]["gpu_lock"])
    if gpu.name != image_gpu.name:
        raise SystemExit(f"host.gpu_lock {gpu} must be named {image_gpu.name} for the container to share it")
    mounts = [f"{ROOT}:/workspace/rola", f"{dev_config.get('host.lock_dir')}:{IMAGE_LAYOUT['host']['lock_dir']}",
              f"{gpu.parent}:{image_gpu.parent}", f"{dev_config.get('store.root')}:{IMAGE_LAYOUT['store']['root']}",
              f"{dev_config.get('workspace.worktrees')}:{IMAGE_LAYOUT['workspace']['worktrees']}",
              f"{dev_config.get('workspace.suite')}:{IMAGE_LAYOUT['workspace']['suite']}"]
    wsl_lib = dev_config.get("host.wsl_lib")
    #: the whole WSL directory, not only lib/: its libcuda is a loader that opens the host's driver store beside it
    return mounts + ([f"{Path(wsl_lib).parent}:{Path(WSL_LIB_IN_IMAGE).parent}:ro"] if wsl_lib else [])


#: run inside the container by `container check`, each a row: the environment check, the GPU and torch's flash backend,
#: a lock held by the host seen as held, and an iteration build of a copy of the checkout imported
CONTAINER_PROOFS = {
    "dev check": "python tools/dev.py check",
    "GPU, torch and its flash backend": """python - <<'PY'
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
assert torch.cuda.is_available(), "no GPU"
q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    F.scaled_dot_product_attention(q, q, q, is_causal=True)
print(torch.cuda.get_device_name(0), torch.__version__)
PY""",
    "a host-held lock is held here": """python - <<'PY'
import fcntl, sys
f = open("/run/rola/locks/rola_container_probe.lock", "w")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("held by the host: one lock governs both")
    sys.exit(0)
print("acquired: the container does not see the host's lock")
sys.exit(1)
PY""",
    "the extension builds and imports": """set -e
rm -rf /tmp/proof && mkdir /tmp/proof
tar -C /workspace/rola --exclude=./build --exclude=./.git --exclude='*.so' -cf - . | tar -C /tmp/proof -xf -
cd /tmp/proof
ROLA_CUDA_ARCHS=86 ROLA_CARRY_ARMS=0 python -m pip install -e . --no-deps --no-build-isolation -q
python -c 'import rola; from rola._build_config import BUILD_CONFIG as c; print("built", c["archs"])'""",
}


def _only_toolchain() -> toolchains.Toolchain:
    declared = toolchains.records()
    if len(declared) != 1:
        raise SystemExit(f"several toolchains are declared ({sorted(declared)}); name one: container build --toolchain NAME")
    return next(iter(declared.values()))


def cmd_container(a) -> int:
    key = env_key()
    tag = f"{IMAGE_REPO}:{key[:12]}"
    if a.action == "compose":
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode:
            raise SystemExit(f"no image {tag} for this environment: python3 tools/dev.py container build")
        local = ROOT / ".devcontainer" / "compose.local.yaml"
        doc = {"services": {"rola": {"image": tag, "volumes": _mounts()}}}
        local.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {local} ({tag})")
        return 0
    if a.action == "build":
        with tempfile.TemporaryDirectory(prefix="rola_image_") as context:
            for rel in ENV_INPUTS:
                (Path(context) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, Path(context) / rel)
            toolchain = toolchains.named(a.toolchain) if a.toolchain else _only_toolchain()
            args = {"ROLA_ENV_KEY": key, "ROLA_TOOLCHAIN": toolchain.name, "CUDA_BASE_IMAGE": toolchain.base_image,
                    "CUDA_PACKAGES": " ".join(toolchain.packages), "TORCH_INDEX": toolchain.torch_index}
            build_args = [x for name, value in args.items() for x in ("--build-arg", f"{name}={value}")]
            return subprocess.run(["docker", "build", *build_args, "-t", tag, "-t", f"{IMAGE_REPO}:latest",
                                   context]).returncode
    if a.action == "gate":
        return _container_gate()

    lock = Path(dev_config.get("host.lock_dir")) / "rola_container_probe.lock"
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, sys; f = open(sys.argv[1], 'w'); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); sys.stdin.read()", str(lock)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    holder.stdout.readline()

    from rola_results import Store, checkout, portable

    volumes = [x for mount in _mounts() for x in ("-v", mount)]
    rows = []
    try:
        for what, script in CONTAINER_PROOFS.items():
            done = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc", "host", *volumes, "-w",
                                   "/workspace/rola", tag, "bash", "-c", script], capture_output=True, text=True)
            tail = [portable(line, ROOT) for line in (done.stdout + done.stderr).strip().splitlines()[-3:]]
            rows.append({"proof": what, "ok": done.returncode == 0, "tail": tail})
            print(f"{'ok ' if done.returncode == 0 else 'RED'}  {what}" + ("" if done.returncode == 0 else f"  -- {tail}"))
    finally:
        holder.stdin.close()
        holder.wait()

    image_id = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", tag], capture_output=True, text=True).stdout
    ok = all(r["ok"] for r in rows)
    result = {"image": tag, "image_id": image_id.strip(), "proofs": rows}
    store = Store(RESULTS_ENVIRONMENT)
    if ok:
        store.put({"env_key": key}, output=result, provenance=checkout(ROOT))
    else:
        store.put({"env_key": key}, error=f"RED: {[r['proof'] for r in rows if not r['ok']]}", provenance=checkout(ROOT),
                  **result)
    print(f"container check: {'green' if ok else 'RED'}; recorded under {store.dir}")
    return 0 if ok else 1


def _container_gate() -> int:
    """KERNEL_STANDARDS §23 (3): a commit that changes the environment's inputs carries a passing container check for
    the key of what it stages."""
    staged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--name-only"], capture_output=True, text=True,
                            check=True).stdout.split()
    if not set(staged) & set(ENV_INPUTS):
        return 0

    def read(rel: str) -> bytes:
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f":{rel}"], capture_output=True)
        return blob.stdout if blob.returncode == 0 else b""

    from rola_results import Store
    from rola_results import key as record_key

    key = env_key(read)
    if Store.complete(Store(RESULTS_ENVIRONMENT).get(record_key({"env_key": key}))):
        return 0
    print(f"§23 (3): this commit changes the dev environment ({', '.join(sorted(set(staged) & set(ENV_INPUTS)))}) and "
          f"no passing container check exists for its key {key[:12]}.\n  python tools/dev.py container build && "
          f"python tools/dev.py container check   (with exactly these files in the working tree)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write this machine's dev config, install pinned tools, wire hooks, then check")
    p.add_argument("--force", action="store_true", help="overwrite detected values already in the config")
    p.add_argument("--image", action="store_true", help="inside the dev image build: its fixed layout, no hooks or clock")
    p.add_argument("--env-key", default="", help="with --image: the environment key the image is built from")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("check", help="verify every setting, tool, pin and lock")
    p.add_argument("--image", action="store_true", help="inside an image build: the toolchain, pins and venv only")
    p.set_defaults(fn=cmd_check)
    sub.add_parser("show", help="every resolved setting and its source").set_defaults(fn=cmd_show)
    p = sub.add_parser("get", help="print one value: <section.key>, or cuda-home")
    p.add_argument("key")
    p.set_defaults(fn=cmd_get)
    p = sub.add_parser("clock", help="lock this host's SM clock once and record it")
    p.add_argument("--mhz", type=int, required=True, help="a clock the card holds under the kernel's power")
    p.set_defaults(fn=cmd_clock)
    p = sub.add_parser("worktree", help="a worktree with its pointer venv, commit gate and extension")
    p.add_argument("name")
    p.add_argument("sha")
    p.add_argument("--build", action="store_true", help="build the extension even when the main checkout's can be copied")
    p.set_defaults(fn=cmd_worktree)
    p = sub.add_parser("store", help="the measurements store: commit new records")
    p.add_argument("action", choices=("commit",))
    p.add_argument("-m", "--message", default="records", help="commit: the message")
    p.set_defaults(fn=cmd_store)
    p = sub.add_parser("container", help="the dev container: compose its mounts, build its image, check it, gate a commit")
    p.add_argument("action", choices=("compose", "build", "check", "gate"))
    p.add_argument("--toolchain", help="build: the toolchain record to build the image for (default: the one declared)")
    p.set_defaults(fn=cmd_container)
    a = ap.parse_args()
    if a.cmd not in BOOTSTRAP:
        _from_base_venv()
    return a.fn(a)


#: the commands that make or read the environment itself, and so run from whichever python starts them
BOOTSTRAP = ("init", "check", "show", "get", "clock")


def _from_base_venv() -> None:
    """Every other command runs from `workspace.base_venv`, re-executing this file there when another python started it:
    the store and the container read `rola_results`, which only the venvs carry, never the host's own python."""
    base = dev_config.get("workspace.base_venv")
    python = Path(base or "") / "bin" / "python"
    if not base or not python.is_file():
        raise SystemExit(f"no base venv at workspace.base_venv ({base!r}): python3 tools/dev.py init")
    if Path(sys.prefix).resolve() != Path(base).resolve():
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
