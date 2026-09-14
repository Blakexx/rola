"""THE SCCACHE WRAPPER PIN -- closed-world rule 1, extended to the compiler cache.

**WHY A CACHE NEEDS A RULE-1 EXTENSION AT ALL.** Rule 1 (`setup.py:_gate_toolchain`,
`tools/ratify.py:check_toolchain`) refuses a build or a ratification whose `ptxas`
does not match the string the manifest was measured under, because a spill/register
profile is a property of the assembler as much as of the source. A compiler *cache*
sits in front of that assembler and can, if unpinned, serve an object nobody
measured: install a different `sccache` build with different argument handling, or
point it at a different `nvcc`, and every gate downstream of the object file would
still see a plausible-looking `.o` with no way to tell it was never compiled by the
gated toolchain at all. `tools/sccache_pin.json` pins the wrapper the way
`csrc/third_party/cutlass/PIN.json` pins a vendored codegen input, and `gate()`
below is what refuses a mismatch before the wrapper is ever invoked.

**WHY THE WRAPPER ITSELF CANNOT BE THE THING THAT GOES WRONG.** `tools/sccache_nvcc.py`
never resolves `nvcc`; it execs whatever `ROLA_REAL_NVCC` says, and this module is
the only thing that sets that variable, always to the same `cuda_bin("nvcc")` path
`setup.py`'s and `tools/ratify.py`'s own rule-1 gates already resolved and checked
`ptxas` against. So "the cache key must carry the assembler identity the manifest
pins" is true by construction, not by a runtime comparison this file would otherwise
have to perform and could get wrong: sccache's own cache key is derived from the
full `nvcc --dryrun` decomposition of the invoked compiler
(https://github.com/mozilla/sccache/blob/main/src/compiler/nvcc.rs), and the
invoked compiler is always the one `ptxas`-gated path.

**THE DEPFILE QUESTION, MEASURED ON THIS TREE (2026-08-06).** Reading the sources
alone could not determine whether torch's forced
`--generate-dependencies-with-compile --dependency-output $out.d` (always added by
`torch.utils.cpp_extension._write_ninja_file` to every nvcc invocation) breaks
sccache's cache key, and recommended treating a 0% `.cu` hit rate as the expected
starting state. Measured here with sccache 0.17.0 pinned below: it does NOT. A cold
compile of a real shard TU WITH the depfile flags, followed by the same compile
WITHOUT them (an argv difference of exactly `-MP --generate-dependencies-with-
compile --dependency-output PATH`), produced a 100% cache hit on the second
invocation and a byte-identical `.o` (`cmp` exit 0). So the flag is NOT stripped
before this file's use of the wrapper, and `tools/ratify.py`'s post-build recompile
-- which never passed those flags to begin with, because it does no incremental
compilation of its own -- lands in the SAME cache namespace `setup.py`'s ninja
build populated. That is the mechanism behind the goal this card names: the second
compile of the same ten translation units becomes a cache hit, not a second
7-minute compile.

**DEFAULT BEHAVIOUR (K39, Blake ruling 2026-08-28: "a tool that can run one of two
ways must REFUSE, not choose").** The pinned wrapper at `toolchain.sccache` (the dev
config) must match its pin, or the build REFUSES -- there is no silent uncached fallback,
so a build never differs in whether it was cached by which binaries happened to be on a
host. `toolchain.sccache_enabled: false` is the one non-refusing outcome: an EXPLICIT
opt-out that builds uncached deliberately.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev_config  # noqa: E402 -- path insert must precede this import

PIN_PATH = HERE / "sccache_pin.json"
WRAPPER_PATH = HERE / "sccache_nvcc.py"


def pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def resolve_sccache() -> str | None:
    """`toolchain.sccache` in the dev config -- no search path; `tools/dev.py init` finds it once."""
    return dev_config.get("toolchain.sccache")


def sccache_version(path: str) -> str:
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{path} --version` failed (rc={proc.returncode}); a "
                           f"closed-world build cannot gate a cache it cannot ask")
    return proc.stdout.strip()


def binary_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate() -> dict | None:
    """Verify the resolved `sccache` against the pin and return
    `{"version", "sha256", "path"}` if it may be used, else `None`.

    Runs BEFORE the wrapper is ever put in `PYTORCH_NVCC` or `ROLA_REAL_NVCC`'s
    caller path, for the same reason rules 1/3/5 run before the compile: refusing a
    mismatch after objects have already been produced under it helps nobody.

    K39 (Blake ruling, 2026-08-28): "a tool that can run one of two ways must
    REFUSE, not choose." A pin miss under the default mode used to print a
    warning and silently continue uncached -- two builds of the identical source
    could then differ in whether they were cached by nothing more than which
    binaries happened to be on a given host's PATH that day, exactly the
    reproducibility hazard closed-world rule 1 exists to close for `ptxas`. There
    is now exactly one non-refusing outcome: `toolchain.sccache_enabled: false`, an
    EXPLICIT, deliberate opt-out (a choice in the dev config, not the gate's). Every other path --
    not found, version mismatch, hash mismatch -- raises.
    """
    if not dev_config.get("toolchain.sccache_enabled"):
        print("sccache: disabled (toolchain.sccache_enabled is false)")
        return None

    p = pin()
    path = resolve_sccache()
    if path is None:
        raise RuntimeError(
            f"CLOSED-WORLD RULE 1 (sccache wrapper pin): toolchain.sccache is not set in the "
            f"dev config. Pinned build is {p['version_string']} "
            f"({p['release_asset']}, {p['release_url']}). `python tools/dev.py init` installs it, "
            f"or set toolchain.sccache_enabled false to build uncached DELIBERATELY -- a build must "
            f"not silently differ by whether a compiler cache happened to be "
            f"installed.")

    version = sccache_version(path)
    digest = binary_sha256(path)
    if version != p["version_string"] or digest != p["binary_sha256"]:
        raise RuntimeError(
            "CLOSED-WORLD RULE 1 (sccache wrapper pin): the resolved `sccache` does "
            f"not match tools/sccache_pin.json.\n"
            f"  pinned  : {p['version_string']}  sha256 {p['binary_sha256'][:16]}...\n"
            f"  resolved: {version}  sha256 {digest[:16]}...  ({path})\n"
            "A cache under an unpinned wrapper can serve an object nobody measured "
            "this ratification against. Install the pinned build, move the pin "
            "DELIBERATELY (tools/sccache_pin.json) after re-verifying the bit-"
            "identity gate (docs/build.md#sccache) under the new one, or set "
            "toolchain.sccache_enabled false to build uncached DELIBERATELY."
        )

    print(f"sccache OK  {version}  sha256 {digest[:16]}...  ({path})")
    return {"version": version, "sha256": digest, "path": path}


def wrap(real_nvcc: str, sccache: str) -> tuple[str, dict]:
    """Point env at the wrapper for `real_nvcc`, returning `(wrapper_path, env_updates)`.

    Callers `os.environ.update(env_updates)` (or pass them to `subprocess.run`'s
    `env=`) BEFORE invoking anything with the returned path as the compiler. The
    wrapper itself performs no resolution (see module docstring); `real_nvcc` must
    already be the rule-1-gated path.
    """
    return str(WRAPPER_PATH), {"ROLA_REAL_NVCC": real_nvcc, "ROLA_SCCACHE_BIN": sccache}
